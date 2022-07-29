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

#ifndef __CUDA_BUNDLE_ADJUSTMENT_H__
#define __CUDA_BUNDLE_ADJUSTMENT_H__

#include "cuda_bundle_adjustment_types.h"

namespace cuba
{

/** @brief CUDA implementation of Bundle Adjustment.

The class implements a Bundle Adjustment algorithm with CUDA.
It optimizes camera poses and landmarks (3D points) represented by a graph.

@attention This class doesn't take responsibility for deleting pointers to vertices and edges
added in the graph.

*/
class CudaBundleAdjustment
{
public:

	using Ptr = UniquePtr<CudaBundleAdjustment>;

	/** @brief Creates an instance of CudaBundleAdjustment.
	*/
	static Ptr create();

	/** @brief Adds a pose vertex to the graph.
	*/
	virtual void addPoseVertex(PoseVertex* v) = 0;

	/** @brief Adds a landmark vertex to the graph.
	*/
	virtual void addLandmarkVertex(LandmarkVertex* v) = 0;

	/** @brief Adds an edge with monocular observation to the graph.
	*/
	virtual void addMonocularEdge(MonoEdge* e) = 0;

	/** @brief Adds an edge with stereo observation to the graph.
	*/
	virtual void addStereoEdge(StereoEdge* e) = 0;

	/** @brief Returns the pose vertex with specified id.
	*/
	virtual PoseVertex* poseVertex(int id) const = 0;

	/** @brief Returns the landmark vertex with specified id.
	*/
	virtual LandmarkVertex* landmarkVertex(int id) const = 0;

	/** @brief Removes a pose vertex from the graph.
	*/
	virtual void removePoseVertex(PoseVertex* v) = 0;

	/** @brief Removes a landmark vertex from the graph.
	*/
	virtual void removeLandmarkVertex(LandmarkVertex* v) = 0;

	/** @brief Removes an edge from the graph.
	*/
	virtual void removeEdge(BaseEdge* e) = 0;

	/** @brief Returns the number of poses in the graph.
	*/
	virtual size_t nposes() const = 0;

	/** @brief Returns the number of landmarks in the graph.
	*/
	virtual size_t nlandmarks() const = 0;

	/** @brief Returns the total number of edges in the graph.
	*/
	virtual size_t nedges() const = 0;

	/** @brief Sets robust kernels to specified edges.
	*/
	virtual void setRobustKernels(RobustKernelType kernelType, double delta, EdgeType edgeType) = 0;

	/** @brief Initializes the graph.
	*/
	virtual void initialize() = 0;

	/** @brief Optimizes the graph.
	@param niterations number of iterations for Levenberg-Marquardt algorithm.
	*/
	virtual void optimize(int niterations) = 0;

	/** @brief Clears the graph.
	*/
	virtual void clear() = 0;

	/** @brief Returns the batch statistics.
	*/
	virtual const BatchStatistics& batchStatistics() const = 0;

	/** @brief Returns the time profile.
	*/
	virtual const TimeProfile& timeProfile() const = 0;

	/** @brief Returns the Chi-squared of the edge.
	@note Returns 0 if the edge is inactive (pose and landmark are both fixed).
	*/
	virtual double chiSquared(const BaseEdge* e) const = 0;

	/** @brief the destructor.
	*/
	virtual ~CudaBundleAdjustment();
};

} // namespace cuba

#endif // !__CUDA_BUNDLE_ADJUSTMENT_H__
