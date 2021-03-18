/*
Copyright 2020 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <vector>
#include <chrono>

#include <opencv2/core.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <cuda_bundle_adjustment.h>
#include <object_creator.h>

using OptimizerCPU = g2o::SparseOptimizer;
using OptimizerGPU = cuba::CudaBundleAdjustment;

static void readGraph(const std::string& filename, OptimizerCPU& optimizerCPU, OptimizerGPU& optimizerGPU,
	std::vector<int>& poseIds, std::vector<int>& landmarkIds);

// use memory manager for vertices and edges, since CudaBundleAdjustment doesn't delete those pointers
static cuba::ObjectCreator obj;

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: sample_comparison_with_g2o input.json" << std::endl;
		return 0;
	}

	auto optimizerCPU = std::make_unique<OptimizerCPU>();
	auto optimizerGPU = OptimizerGPU::create();

	std::vector<int> poseIds, landmarkIds;

	std::cout << "Reading Graph... " << std::flush;

	readGraph(argv[1], *optimizerCPU, *optimizerGPU, poseIds, landmarkIds);

	std::cout << "Done." << std::endl << std::endl;

	std::cout << "=== Graph size : " << std::endl;
	std::cout << "num poses      : " << optimizerGPU->nposes() << std::endl;
	std::cout << "num landmarks  : " << optimizerGPU->nlandmarks() << std::endl;
	std::cout << "num edges      : " << optimizerGPU->nedges() << std::endl << std::endl;

	std::cout << "Running BA with CPU... " << std::flush;
	const auto t0 = std::chrono::steady_clock::now();
	optimizerCPU->initializeOptimization();
	optimizerCPU->optimize(10);
	const auto t1 = std::chrono::steady_clock::now();
	std::cout << "Done." << std::endl << std::endl;

	std::cout << "Running BA with GPU... " << std::flush;
	const auto t2 = std::chrono::steady_clock::now();
	optimizerGPU->initialize();
	optimizerGPU->optimize(10);
	const auto t3 = std::chrono::steady_clock::now();
	std::cout << "Done." << std::endl << std::endl;

	const auto duration01 = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
	const auto duration23 = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();

	std::cout << "=== Processing time : " << std::endl;
	std::printf("CPU : %7.2f [sec]\n", duration01);
	std::printf("GPU : %7.2f [sec]\n", duration23);
	std::cout << std::endl;

	std::cout << "=== Objective function value : " << std::endl;
	const auto& statsCPU = optimizerCPU->batchStatistics();
	const auto& statsGPU = optimizerGPU->batchStatistics();
	const size_t niterations = std::max(statsCPU.size(), statsGPU.size());
	std::printf("%10s|%10s|%10s\n", "iteration", "chi2 CPU", "chi2 GPU");
	for (size_t i = 0; i < niterations; i++)
	{
		std::printf("%10zd|", i + 1);

		if (i < statsCPU.size())
			std::printf("%10.1f|", statsCPU[i].chi2);
		else
			std::printf("%10s", "N/A|");

		if (i < statsGPU.size())
			std::printf("%10.1f", statsGPU[i].chi2);
		else
			std::printf("%10s", "N/A");

		std::puts("");
	}
	std::cout << std::endl;

	double errorSqR = 0, errorSqT = 0;
	for (int id : poseIds)
	{
		const auto vcpu = (g2o::VertexSE3Expmap*)optimizerCPU->vertex(id);
		const auto vgpu = optimizerGPU->poseVertex(id);
		const auto estcpu = vcpu->estimate();
		const auto qdiff = estcpu.rotation().coeffs() - vgpu->q.coeffs();
		const auto tdiff = estcpu.translation() - vgpu->t;
		errorSqR += qdiff.dot(qdiff);
		errorSqT += tdiff.dot(tdiff);
	}

	double errorSqP = 0;
	for (int id : landmarkIds)
	{
		const auto vcpu = (g2o::VertexPointXYZ*)optimizerCPU->vertex(id);
		const auto vgpu = optimizerGPU->landmarkVertex(id);
		const auto pdiff = vcpu->estimate() - vgpu->Xw;
		errorSqP += pdiff.dot(pdiff);
	}

	std::cout << "=== RMSE between CPU estimates and GPU estimates : " << std::endl;
	std::printf("Rotation    : %.2e\n", sqrt(errorSqR / poseIds.size()));
	std::printf("Translation : %.2e\n", sqrt(errorSqT / poseIds.size()));
	std::printf("Landmark    : %.2e\n", sqrt(errorSqP / landmarkIds.size()));

	return 0;
}

template <typename T, int N>
static inline cuba::Array<T, N> getArray(const cv::FileNode& node)
{
	cuba::Array<T, N> arr = {};
	int pos = 0;
	for (const auto& v : node)
	{
		arr[pos] = T(v);
		if (++pos >= N)
			break;
	}
	return arr;
}

static void readGraph(const std::string& filename, OptimizerCPU& optimizerCPU, OptimizerGPU& optimizerGPU,
	std::vector<int>& poseIds, std::vector<int>& landmarkIds)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	CV_Assert(fs.isOpened());

	poseIds.clear();
	landmarkIds.clear();

	auto linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
	auto blockSolver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
	auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
	optimizerCPU.setAlgorithm(algorithm);
	optimizerCPU.setComputeBatchStatistics(true);

	// read camera parameters
	cuba::CameraParams camera;
	camera.fx = fs["fx"];
	camera.fy = fs["fy"];
	camera.cx = fs["cx"];
	camera.cy = fs["cy"];
	camera.bf = fs["bf"];

	optimizerGPU.setCameraPrams(camera);

	// read pose vertices
	for (const auto& node : fs["pose_vertices"])
	{
		const int id = node["id"];
		const int fixed = node["fixed"];
		const auto q = Eigen::Quaterniond(getArray<double, 4>(node["q"]));
		const auto t = getArray<double, 3>(node["t"]);

		// add pose vertex to CPU optimizer
		auto vcpu = new g2o::VertexSE3Expmap();
		vcpu->setEstimate(g2o::SE3Quat(q, t));
		vcpu->setId(id);
		vcpu->setFixed(fixed);
		optimizerCPU.addVertex(vcpu);

		// add pose vertex to GPU optimizer
		auto vgpu = obj.create<cuba::PoseVertex>(id, q, t, fixed);
		optimizerGPU.addPoseVertex(vgpu);

		poseIds.push_back(id);
	}

	// read landmark vertices
	for (const auto& node : fs["landmark_vertices"])
	{
		const int id = node["id"];
		const int fixed = node["fixed"];
		const auto Xw = getArray<double, 3>(node["Xw"]);

		// add landmark vertex to CPU optimizer
		auto vcpu = new g2o::VertexPointXYZ();
		vcpu->setEstimate(Xw);
		vcpu->setId(id);
		vcpu->setFixed(fixed);
		vcpu->setMarginalized(true);
		optimizerCPU.addVertex(vcpu);

		// add landmark vertex to GPU optimizer
		auto vgpu = obj.create<cuba::LandmarkVertex>(id, Xw, fixed);
		optimizerGPU.addLandmarkVertex(vgpu);

		landmarkIds.push_back(id);
	}

	// read monocular edges
	for (const auto& node : fs["monocular_edges"])
	{
		const int iP = node["vertexP"];
		const int iL = node["vertexL"];
		const auto measurement = getArray<double, 2>(node["measurement"]);
		const double information = node["information"];

		// add monocular edge to CPU optimizer
		auto ecpu = new g2o::EdgeSE3ProjectXYZ();
		ecpu->setVertex(0, optimizerCPU.vertex(iL));
		ecpu->setVertex(1, optimizerCPU.vertex(iP));
		ecpu->setMeasurement(measurement);
		ecpu->setInformation(information * Eigen::Matrix2d::Identity());
		ecpu->fx = camera.fx;
		ecpu->fy = camera.fy;
		ecpu->cx = camera.cx;
		ecpu->cy = camera.cy;
		optimizerCPU.addEdge(ecpu);

		// add monocular edge to GPU optimizer
		auto vertexP = optimizerGPU.poseVertex(iP);
		auto vertexL = optimizerGPU.landmarkVertex(iL);
		auto egpu = obj.create<cuba::MonoEdge>(measurement, information, vertexP, vertexL);
		optimizerGPU.addMonocularEdge(egpu);
	}

	// read stereo edges
	for (const auto& node : fs["stereo_edges"])
	{
		const int iP = node["vertexP"];
		const int iL = node["vertexL"];
		const auto measurement = getArray<double, 3>(node["measurement"]);
		const double information = node["information"];

		// add stereo edge to CPU optimizer
		auto ecpu = new g2o::EdgeStereoSE3ProjectXYZ();
		ecpu->setVertex(0, optimizerCPU.vertex(iL));
		ecpu->setVertex(1, optimizerCPU.vertex(iP));
		ecpu->setMeasurement(measurement);
		ecpu->setInformation(information * Eigen::Matrix3d::Identity());
		ecpu->fx = camera.fx;
		ecpu->fy = camera.fy;
		ecpu->cx = camera.cx;
		ecpu->cy = camera.cy;
		ecpu->bf = camera.bf;
		optimizerCPU.addEdge(ecpu);

		// add stereo edge to GPU optimizer
		auto vertexP = optimizerGPU.poseVertex(iP);
		auto vertexL = optimizerGPU.landmarkVertex(iL);
		auto egpu = obj.create<cuba::StereoEdge>(measurement, information, vertexP, vertexL);
		optimizerGPU.addStereoEdge(egpu);
	}

	// "warm-up" to avoid overhead
	optimizerCPU.initializeOptimization();
	optimizerCPU.optimize(1);
	optimizerGPU.initialize();
	optimizerGPU.optimize(1);
}
