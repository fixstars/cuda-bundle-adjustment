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

#include "cuda_bundle_adjustment.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

#include "constants.h"
#include "sparse_block_matrix.h"
#include "device_buffer.h"
#include "device_matrix.h"
#include "cuda_block_solver.h"
#include "cuda_linear_solver.h"

namespace cuba
{

using VertexMapP = std::map<int, VertexP*>;
using VertexMapL = std::map<int, VertexL*>;
using EdgeSet2D = std::unordered_set<Edge2D*>;
using EdgeSet3D = std::unordered_set<Edge3D*>;
using time_point = decltype(std::chrono::steady_clock::now());

static inline time_point get_time_point()
{
	gpu::waitForKernelCompletion();
	return std::chrono::steady_clock::now();
}

static inline double get_duration(const time_point& from, const time_point& to)
{
	return std::chrono::duration_cast<std::chrono::duration<double>>(to - from).count();
}

template <typename T>
static constexpr Scalar ScalarCast(T v) { return static_cast<Scalar>(v); }

/** @brief Implementation of Block solver.
*/
class CudaBlockSolver
{
public:

	enum ProfileItem
	{
		PROF_ITEM_INITIALIZE,
		PROF_ITEM_BUILD_STRUCTURE,
		PROF_ITEM_COMPUTE_ERROR,
		PROF_ITEM_BUILD_SYSTEM,
		PROF_ITEM_SCHUR_COMPLEMENT,
		PROF_ITEM_DECOMP_SYMBOLIC,
		PROF_ITEM_DECOMP_NUMERICAL,
		PROF_ITEM_UPDATE,
		PROF_ITEM_NUM
	};

	struct PLIndex
	{
		int P, L;
		PLIndex(int P = 0, int L = 0) : P(P), L(L) {}
	};

	void clear()
	{
		verticesP_.clear();
		verticesL_.clear();
		baseEdges_.clear();
		HplBlockPos_.clear();
		qs_.clear();
		ts_.clear();
		Xws_.clear();
		measurements2D_.clear();
		measurements3D_.clear();
		omegas_.clear();
		edge2PL_.clear();
		edgeFlags_.clear();
	}

	void initialize(const VertexMapP& vertexMapP, const VertexMapL& vertexMapL,
		const EdgeSet2D& edgeSet2D, const EdgeSet3D& edgeSet3D, const CameraParams& camera)
	{
		const auto t0 = get_time_point();

		nedges2D_ = static_cast<int>(edgeSet2D.size());
		nedges3D_ = static_cast<int>(edgeSet3D.size());

		clear();

		verticesP_.reserve(vertexMapP.size());
		verticesL_.reserve(vertexMapL.size());
		baseEdges_.reserve(nedges2D_ + nedges3D_);
		HplBlockPos_.reserve(nedges2D_ + nedges3D_);
		qs_.reserve(vertexMapP.size());
		ts_.reserve(vertexMapP.size());
		Xws_.reserve(vertexMapL.size());
		measurements2D_.reserve(nedges2D_);
		measurements3D_.reserve(nedges3D_);
		omegas_.reserve(nedges2D_ + nedges3D_);
		edge2PL_.reserve(nedges2D_ + nedges3D_);
		edgeFlags_.reserve(nedges2D_ + nedges3D_);

		std::vector<VertexP*> fixedVerticesP_;
		std::vector<VertexL*> fixedVerticesL_;
		int numP = 0;
		int numL = 0;

		// assign pose vertex id
		// gather rotations and translations into each vector
		for (const auto& [id, vertexP] : vertexMapP)
		{
			if (!vertexP->fixed)
			{
				vertexP->iP = numP++;
				verticesP_.push_back(vertexP);
				qs_.emplace_back(vertexP->q.coeffs().data());
				ts_.emplace_back(vertexP->t.data());
			}
			else
			{
				fixedVerticesP_.push_back(vertexP);
			}
		}

		// assign landmark vertex id
		// gather 3D positions into vector
		for (const auto& [id, vertexL] : vertexMapL)
		{
			if (!vertexL->fixed)
			{
				vertexL->iL = numL++;
				verticesL_.push_back(vertexL);
				Xws_.emplace_back(vertexL->Xw.data());
			}
			else
			{
				fixedVerticesL_.push_back(vertexL);
			}
		}

		numP_ = numP;
		numL_ = numL;

		// inactive(fixed) vertices are added after active vertices
		for (auto vertexP : fixedVerticesP_)
		{
			vertexP->iP = numP++;
			verticesP_.push_back(vertexP);
			qs_.emplace_back(vertexP->q.coeffs().data());
			ts_.emplace_back(vertexP->t.data());
		}

		for (auto vertexL : fixedVerticesL_)
		{
			vertexL->iL = numL++;
			verticesL_.push_back(vertexL);
			Xws_.emplace_back(vertexL->Xw.data());
		}

		// gather each edge members into each vector
		int edgeId = 0;
		for (const auto e : edgeSet2D)
		{
			const auto vertexP = e->vertexP;
			const auto vertexL = e->vertexL;

			baseEdges_.push_back(e);

			if (!vertexP->fixed && !vertexL->fixed)
				HplBlockPos_.push_back({ vertexP->iP, vertexL->iL, edgeId });

			measurements2D_.emplace_back(e->measurement.data());
			omegas_.push_back(ScalarCast(e->information));
			edge2PL_.push_back({ vertexP->iP, vertexL->iL });
			edgeFlags_.push_back(makeEdgeFlag(vertexP->fixed, vertexL->fixed));

			edgeId++;
		}

		// gather each edge members into each vector
		for (const auto e : edgeSet3D)
		{
			const auto vertexP = e->vertexP;
			const auto vertexL = e->vertexL;

			baseEdges_.push_back(e);
			
			if (!vertexP->fixed && !vertexL->fixed)
				HplBlockPos_.push_back({ vertexP->iP, vertexL->iL, edgeId });

			measurements3D_.emplace_back(e->measurement.data());
			omegas_.push_back(ScalarCast(e->information));
			edge2PL_.push_back({ vertexP->iP, vertexL->iL });
			edgeFlags_.push_back(makeEdgeFlag(vertexP->fixed, vertexL->fixed));

			edgeId++;
		}

		nHplBlocks_ = static_cast<int>(HplBlockPos_.size());

		// upload camera parameters to constant memory
		std::vector<Scalar> cameraParams(5);
		cameraParams[0] = ScalarCast(camera.fx);
		cameraParams[1] = ScalarCast(camera.fy);
		cameraParams[2] = ScalarCast(camera.cx);
		cameraParams[3] = ScalarCast(camera.cy);
		cameraParams[4] = ScalarCast(camera.bf);
		gpu::setCameraParameters(cameraParams.data());

		// create sparse linear solver
		if (!linearSolver_)
			linearSolver_ = SparseLinearSolver::create();

		profItems_.assign(PROF_ITEM_NUM, 0);

		const auto t1 = get_time_point();
		profItems_[PROF_ITEM_INITIALIZE] += get_duration(t0, t1);
	}

	void buildStructure()
	{
		const auto t0 = get_time_point();

		// build Hpl block matrix structure
		d_Hpl_.resize(numP_, numL_);
		d_Hpl_.resizeNonZeros(nHplBlocks_);

		d_HplBlockPos_.assign(nHplBlocks_, HplBlockPos_.data());
		d_nnzPerCol_.resize(numL_ + 1);
		d_edge2Hpl_.resize(baseEdges_.size());

		gpu::buildHplStructure(d_HplBlockPos_, d_Hpl_, d_edge2Hpl_, d_nnzPerCol_);

		// build Hschur block matrix structure
		Hsc_.resize(numP_, numP_);
		Hsc_.constructFromVertices(verticesL_);
		Hsc_.convertBSRToCSR();

		d_Hsc_.resize(numP_, numP_);
		d_Hsc_.resizeNonZeros(Hsc_.nblocks());
		d_Hsc_.upload(nullptr, Hsc_.outerIndices(), Hsc_.innerIndices());

		d_HscCSR_.resize(Hsc_.nnzSymm());
		d_BSR2CSR_.assign(Hsc_.nnzSymm(), (int*)Hsc_.BSR2CSR());

		d_HscMulBlockIds_.resize(Hsc_.nmulBlocks());
		gpu::findHschureMulBlockIndices(d_Hpl_, d_Hsc_, d_HscMulBlockIds_);

		// allocate device buffers
		d_x_.resize(numP_ * PDIM + numL_ * LDIM);
		d_b_.resize(numP_ * PDIM + numL_ * LDIM);

		d_xp_.map(numP_, d_x_.data());
		d_bp_.map(numP_, d_b_.data());
		d_xl_.map(numL_, d_x_.data() + numP_ * PDIM);
		d_bl_.map(numL_, d_b_.data() + numP_ * PDIM);

		d_Hpp_.resize(numP_);
		d_Hll_.resize(numL_);

		d_HppBackup_.resize(numP_);
		d_HllBackup_.resize(numL_);

		d_bsc_.resize(numP_);
		d_invHll_.resize(numL_);
		d_Hpl_invHll_.resize(nHplBlocks_);

		// upload solutions to device memory
		d_solution_.resize(verticesP_.size() * 7 + verticesL_.size() * 3);
		d_solutionBackup_.resize(d_solution_.size());

		d_qs_.map(qs_.size(), d_solution_.data());
		d_ts_.map(ts_.size(), d_qs_.data() + d_qs_.size());
		d_Xws_.map(Xws_.size(), d_ts_.data() + d_ts_.size());
		
		d_qs_.upload(qs_.data());
		d_ts_.upload(ts_.data());
		d_Xws_.upload(Xws_.data());

		// upload edge information to device memory
		d_measurements2D_.assign(nedges2D_, measurements2D_.data());
		d_measurements3D_.assign(nedges3D_, measurements3D_.data());
		d_errors2D_.resize(nedges2D_);
		d_errors3D_.resize(nedges3D_);
		d_omegas2D_.assign(nedges2D_, omegas_.data());
		d_omegas3D_.assign(nedges3D_, omegas_.data() + nedges2D_);
		d_Xcs2D_.resize(nedges2D_);
		d_Xcs3D_.resize(nedges3D_);
		d_edge2PL2D_.assign(nedges2D_, edge2PL_.data());
		d_edge2PL3D_.assign(nedges3D_, edge2PL_.data() + nedges2D_);
		d_edgeFlags2D_.assign(nedges2D_, edgeFlags_.data());
		d_edgeFlags3D_.assign(nedges3D_, edgeFlags_.data() + nedges2D_);
		d_edge2Hpl2D_.map(nedges2D_, d_edge2Hpl_.data());
		d_edge2Hpl3D_.map(nedges3D_, d_edge2Hpl_.data() + nedges2D_);
		d_chi_.resize(1);

		const auto t1 = get_time_point();

		// analyze pattern of Hschur matrix (symbolic decomposition)
		linearSolver_->initialize(Hsc_);

		const auto t2 = get_time_point();

		profItems_[PROF_ITEM_BUILD_STRUCTURE] += get_duration(t0, t1);
		profItems_[PROF_ITEM_DECOMP_SYMBOLIC] += get_duration(t1, t2);
	}

	double computeErrors()
	{
		const auto t0 = get_time_point();

		const Scalar chi2D = gpu::computeActiveErrors(d_qs_, d_ts_, d_Xws_, d_measurements2D_,
			d_omegas2D_, d_edge2PL2D_, d_errors2D_, d_Xcs2D_, d_chi_);

		const Scalar chi3D = gpu::computeActiveErrors(d_qs_, d_ts_, d_Xws_, d_measurements3D_,
			d_omegas3D_, d_edge2PL3D_, d_errors3D_, d_Xcs3D_, d_chi_);

		const auto t1 = get_time_point();
		profItems_[PROF_ITEM_COMPUTE_ERROR] += get_duration(t0, t1);

		return chi2D + chi3D;
	}

	void buildSystem()
	{
		const auto t0 = get_time_point();

		////////////////////////////////////////////////////////////////////////////////////
		// Build linear system about solution increments Δx
		// H*Δx = -b
		// 
		// coefficient matrix are divided into blocks, and each block is calculated
		// | Hpp  Hpl ||Δxp| = |-bp|
		// | HplT Hll ||Δxl|   |-bl|
		////////////////////////////////////////////////////////////////////////////////////

		d_Hpp_.fillZero();
		d_Hll_.fillZero();
		d_bp_.fillZero();
		d_bl_.fillZero();

		gpu::constructQuadraticForm(d_Xcs2D_, d_qs_, d_errors2D_, d_omegas2D_, d_edge2PL2D_,
			d_edge2Hpl2D_, d_edgeFlags2D_, d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_);

		gpu::constructQuadraticForm(d_Xcs3D_, d_qs_, d_errors3D_, d_omegas3D_, d_edge2PL3D_,
			d_edge2Hpl3D_, d_edgeFlags3D_, d_Hpp_, d_bp_, d_Hll_, d_bl_, d_Hpl_);

		const auto t1 = get_time_point();
		profItems_[PROF_ITEM_BUILD_SYSTEM] += get_duration(t0, t1);
	}

	double maxDiagonal()
	{
		DeviceBuffer<Scalar> d_buffer(16);
		const Scalar maxP = gpu::maxDiagonal(d_Hpp_, d_buffer);
		const Scalar maxL = gpu::maxDiagonal(d_Hll_, d_buffer);
		return std::max(maxP, maxL);
	}

	void setLambda(double lambda)
	{
		gpu::addLambda(d_Hpp_, ScalarCast(lambda), d_HppBackup_);
		gpu::addLambda(d_Hll_, ScalarCast(lambda), d_HllBackup_);
	}

	void restoreDiagonal()
	{
		gpu::restoreDiagonal(d_Hpp_, d_HppBackup_);
		gpu::restoreDiagonal(d_Hll_, d_HllBackup_);
	}

	bool solve()
	{
		const auto t0 = get_time_point();

		////////////////////////////////////////////////////////////////////////////////////
		// Schur complement
		// bSc = -bp + Hpl*Hll^-1*bl
		// HSc = Hpp - Hpl*Hll^-1*HplT
		////////////////////////////////////////////////////////////////////////////////////
		gpu::computeBschure(d_bp_, d_Hpl_, d_Hll_, d_bl_, d_bsc_, d_invHll_, d_Hpl_invHll_);
		gpu::computeHschure(d_Hpp_, d_Hpl_invHll_, d_Hpl_, d_HscMulBlockIds_, d_Hsc_);
		
		const auto t1 = get_time_point();

		////////////////////////////////////////////////////////////////////////////////////
		// Solve linear equation about Δxp
		// HSc*Δxp = bp
		////////////////////////////////////////////////////////////////////////////////////
		gpu::convertHschureBSRToCSR(d_Hsc_, d_BSR2CSR_, d_HscCSR_);
		const bool success = linearSolver_->solve(d_HscCSR_, d_bsc_.values(), d_xp_.values());
		if (!success)
			return false;

		const auto t2 = get_time_point();

		////////////////////////////////////////////////////////////////////////////////////
		// Solve linear equation about Δxl
		// Hll*Δxl = -bl - HplT*Δxp
		////////////////////////////////////////////////////////////////////////////////////
		gpu::schurComplementPost(d_invHll_, d_bl_, d_Hpl_, d_xp_, d_xl_);

		const auto t3 = get_time_point();
		profItems_[PROF_ITEM_SCHUR_COMPLEMENT] += (get_duration(t0, t1) + get_duration(t2, t3));
		profItems_[PROF_ITEM_DECOMP_NUMERICAL] += get_duration(t1, t2);

		return true;
	}

	void update()
	{
		const auto t0 = get_time_point();

		gpu::updatePoses(d_xp_, d_qs_, d_ts_);
		gpu::updateLandmarks(d_xl_, d_Xws_);

		const auto t1 = get_time_point();
		profItems_[PROF_ITEM_UPDATE] += get_duration(t0, t1);
	}

	double computeScale(double lambda)
	{
		gpu::computeScale(d_x_, d_b_, d_chi_, ScalarCast(lambda));
		Scalar scale = 0;
		d_chi_.download(&scale);
		return scale;
	}

	void push()
	{
		d_solution_.copyTo(d_solutionBackup_);
	}

	void pop()
	{
		d_solutionBackup_.copyTo(d_solution_);
	}

	void finalize()
	{
		d_qs_.download(qs_.data());
		d_ts_.download(ts_.data());
		d_Xws_.download(Xws_.data());

		for (size_t i = 0; i < verticesP_.size(); i++)
		{
			qs_[i].copyTo(verticesP_[i]->q.coeffs().data());
			ts_[i].copyTo(verticesP_[i]->t.data());
		}

		for (size_t i = 0; i < verticesL_.size(); i++)
			Xws_[i].copyTo(verticesL_[i]->Xw.data());
	}

	void getTimeProfile(TimeProfile& prof) const
	{
		static const char* profileItemString[PROF_ITEM_NUM] =
		{
			"0: Initialize Optimizer",
			"1: Build Structure",
			"2: Compute Error",
			"3: Build System",
			"4: Schur Complement",
			"5: Symbolic Decomposition",
			"6: Numerical Decomposition",
			"7: Update Solution"
		};

		prof.clear();
		for (int i = 0; i < PROF_ITEM_NUM; i++)
			prof[profileItemString[i]] = profItems_[i];
	}

private:

	static inline uint8_t makeEdgeFlag(bool fixedP, bool fixedL)
	{
		uint8_t flag = 0;
		if (fixedP) flag |= EDGE_FLAG_FIXED_P;
		if (fixedL) flag |= EDGE_FLAG_FIXED_L;
		return flag;
	}

	////////////////////////////////////////////////////////////////////////////////////
	// host buffers
	////////////////////////////////////////////////////////////////////////////////////

	// graph components
	std::vector<VertexP*> verticesP_;
	std::vector<VertexL*> verticesL_;
	std::vector<BaseEdge*> baseEdges_;
	int numP_, numL_, nedges2D_, nedges3D_;

	// solution vectors
	std::vector<Vec4d> qs_;
	std::vector<Vec3d> ts_;
	std::vector<Vec3d> Xws_;

	// edge information
	std::vector<Vec2d> measurements2D_;
	std::vector<Vec3d> measurements3D_;
	std::vector<Scalar> omegas_;
	std::vector<PLIndex> edge2PL_;
	std::vector<uint8_t> edgeFlags_;

	// block matrices
	HplSparseBlockMatrix Hpl_;
	HschurSparseBlockMatrix Hsc_;
	SparseLinearSolver::Ptr linearSolver_;
	std::vector<HplBlockPos> HplBlockPos_;
	int nHplBlocks_;

	////////////////////////////////////////////////////////////////////////////////////
	// device buffers
	////////////////////////////////////////////////////////////////////////////////////

	// solution vectors
	GpuVec1d d_solution_, d_solutionBackup_;
	GpuVec4d d_qs_;
	GpuVec3d d_ts_, d_Xws_;

	// edge information
	GpuVec3d d_Xcs2D_, d_Xcs3D_;
	GpuVec1d d_omegas2D_, d_omegas3D_;
	GpuVec2d d_measurements2D_, d_errors2D_;
	GpuVec3d d_measurements3D_, d_errors3D_;
	GpuVec2i d_edge2PL2D_, d_edge2PL3D_;
	GpuVec1b d_edgeFlags2D_, d_edgeFlags3D_;
	GpuVec1i d_edge2Hpl_, d_edge2Hpl2D_, d_edge2Hpl3D_;

	// solution increments Δx = [Δxp Δxl]
	GpuVec1d d_x_;
	GpuPx1BlockVec d_xp_;
	GpuLx1BlockVec d_xl_;

	// coefficient matrix of linear system
	// | Hpp  Hpl ||Δxp| = |-bp|
	// | HplT Hll ||Δxl|   |-bl|
	GpuPxPBlockVec d_Hpp_;
	GpuLxLBlockVec d_Hll_;
	GpuHplBlockMat d_Hpl_;
	GpuVec3i d_HplBlockPos_;
	GpuVec1d d_b_;
	GpuPx1BlockVec d_bp_;
	GpuLx1BlockVec d_bl_;
	GpuPx1BlockVec d_HppBackup_;
	GpuLx1BlockVec d_HllBackup_;

	// schur complement of the H matrix
	// HSc = Hpp - Hpl*inv(Hll)*HplT
	// bSc = -bp + Hpl*inv(Hll)*bl
	GpuHscBlockMat d_Hsc_;
	GpuPx1BlockVec d_bsc_;
	GpuLxLBlockVec d_invHll_;
	GpuPxLBlockVec d_Hpl_invHll_;
	GpuVec3i d_HscMulBlockIds_;

	// conversion matrix storage format BSR to CSR
	GpuVec1d d_HscCSR_;
	GpuVec1i d_BSR2CSR_;

	// temporary buffer
	DeviceBuffer<Scalar> d_chi_;
	GpuVec1i d_nnzPerCol_;

	////////////////////////////////////////////////////////////////////////////////////
	// statistics
	////////////////////////////////////////////////////////////////////////////////////

	std::vector<double> profItems_;
};

/** @brief Implementation of CudaBundleAdjustment.
*/
class CudaBundleAdjustmentImpl : public CudaBundleAdjustment
{
public:

	void addPoseVertex(VertexP* v) override
	{
		vertexMapP_.insert({ v->id, v });
	}

	void addLandmarkVertex(VertexL* v) override
	{
		vertexMapL_.insert({ v->id, v });
	}

	void addMonocularEdge(Edge2D* e) override
	{
		edges2D_.insert(e);

		e->vertexP->edges.insert(e);
		e->vertexL->edges.insert(e);
	}

	void addStereoEdge(Edge3D* e) override
	{
		edges3D_.insert(e);

		e->vertexP->edges.insert(e);
		e->vertexL->edges.insert(e);
	}

	VertexP* poseVertex(int id) const override
	{
		return vertexMapP_.at(id);
	}

	VertexL* landmarkVertex(int id) const override
	{
		return vertexMapL_.at(id);
	}

	void removePoseVertex(PoseVertex* v) override
	{
		auto it = vertexMapP_.find(v->id);
		if (it == std::end(vertexMapP_))
			return;

		for (auto e : it->second->edges)
			removeEdge(e);

		vertexMapP_.erase(it);
	}

	void removeLandmarkVertex(LandmarkVertex* v) override
	{
		auto it = vertexMapL_.find(v->id);
		if (it == std::end(vertexMapL_))
			return;

		for (auto e : it->second->edges)
			removeEdge(e);

		vertexMapL_.erase(it);
	}

	void removeEdge(BaseEdge* e) override
	{
		auto vertexP = e->poseVertex();
		if (vertexP->edges.count(e))
			vertexP->edges.erase(e);

		auto vertexL = e->landmarkVertex();
		if (vertexL->edges.count(e))
			vertexL->edges.erase(e);

		if (e->dim() == 2)
		{
			auto edge2D = reinterpret_cast<Edge2D*>(e);
			if (edges2D_.count(edge2D))
				edges2D_.erase(edge2D);
		}

		if (e->dim() == 3)
		{
			auto edge3D = reinterpret_cast<Edge3D*>(e);
			if (edges3D_.count(edge3D))
				edges3D_.erase(edge3D);
		}
	}

	void setCameraPrams(const CameraParams& camera) override
	{
		camera_ = camera;
	}

	size_t nposes() const override
	{
		return vertexMapP_.size();
	}

	size_t nlandmarks() const override
	{
		return vertexMapL_.size();
	}

	size_t nedges() const override
	{
		return edges2D_.size() + edges3D_.size();
	}

	void initialize() override
	{
		solver_.initialize(vertexMapP_, vertexMapL_, edges2D_, edges3D_, camera_);

		stats_.clear();
	}

	void optimize(int niterations) override
	{
		const int maxq = 10;
		const double tau = 1e-5;

		double nu = 2;
		double lambda = 0;
		double F = 0;

		// Levenberg-Marquardt iteration
		for (int iteration = 0; iteration < niterations; iteration++)
		{
			if (iteration == 0)
				solver_.buildStructure();

			const double iniF = solver_.computeErrors();
			F = iniF;

			solver_.buildSystem();
			
			if (iteration == 0)
				lambda = tau * solver_.maxDiagonal();

			int q = 0;
			double rho = -1;
			for (; q < maxq && rho < 0; q++)
			{
				solver_.push();

				solver_.setLambda(lambda);

				const bool success = solver_.solve();

				solver_.update();

				const double Fhat = solver_.computeErrors();
				const double scale = solver_.computeScale(lambda) + 1e-3;
				rho = success ? (F - Fhat) / scale : -1;

				if (rho > 0)
				{
					lambda *= clamp(attenuation(rho), 1./3, 2./3);
					nu = 2;
					F = Fhat;
					break;
				}
				else
				{
					lambda *= nu;
					nu *= 2;
					solver_.restoreDiagonal();
					solver_.pop();
				}
			}

			stats_.push_back({ iteration, F });

			if (q == maxq || rho <= 0 || !std::isfinite(lambda))
				break;
		}

		solver_.finalize();

		solver_.getTimeProfile(timeProfile_);
	}

	void clear() override
	{
		vertexMapP_.clear();
		vertexMapL_.clear();
		edges2D_.clear();
		edges3D_.clear();
		stats_.clear();
	}

	const BatchStatistics& batchStatistics() const override
	{
		return stats_;
	}

	const TimeProfile& timeProfile() override
	{
		return timeProfile_;
	}

	~CudaBundleAdjustmentImpl()
	{
		clear();
	}

private:

	static inline double attenuation(double x) { return 1 - std::pow(2 * x - 1, 3); }
	static inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

	CudaBlockSolver solver_;
	VertexMapP vertexMapP_;
	VertexMapL vertexMapL_;
	EdgeSet2D edges2D_;
	EdgeSet3D edges3D_;
	CameraParams camera_;

	BatchStatistics stats_;
	TimeProfile timeProfile_;
};

CudaBundleAdjustment::Ptr CudaBundleAdjustment::create()
{
	return std::make_unique<CudaBundleAdjustmentImpl>();
}

CudaBundleAdjustment::~CudaBundleAdjustment()
{
}

} // namespace cuba
