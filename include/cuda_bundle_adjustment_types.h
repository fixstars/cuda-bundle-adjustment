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

#ifndef __CUDA_BUNDLE_ADJUSTMENT_TYPES_H__
#define __CUDA_BUNDLE_ADJUSTMENT_TYPES_H__

#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cuba
{

////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

template <class T, int N>
using Array = Eigen::Matrix<T, N, 1>;

template <class T>
using Set = std::unordered_set<T>;

template <class T>
using UniquePtr = std::unique_ptr<T>;

////////////////////////////////////////////////////////////////////////////////////
// Camera parameters
////////////////////////////////////////////////////////////////////////////////////

/** @brief Camera parameters struct.
*/
struct CameraParams
{
	double fx;               //!< focal length x (pixel)
	double fy;               //!< focal length y (pixel)
	double cx;               //!< principal point x (pixel)
	double cy;               //!< principal point y (pixel)
	double bf;               //!< stereo baseline times fx

	/** @brief The constructor.
	*/
	CameraParams() : fx(0), fy(0), cx(0), cy(0), bf(0) {}
};

////////////////////////////////////////////////////////////////////////////////////
// Edge
////////////////////////////////////////////////////////////////////////////////////

struct PoseVertex;
struct LandmarkVertex;

/** @brief Base edge struct.
*/
struct BaseEdge
{
	/** @brief Returns the connected pose vertex.
	*/
	virtual PoseVertex* poseVertex() const = 0;

	/** @brief Returns the connected landmark vertex.
	*/
	virtual LandmarkVertex* landmarkVertex() const = 0;

	/** @brief Returns the dimension of measurement.
	*/
	virtual int dim() const = 0;

	/** @brief the destructor.
	*/
	virtual ~BaseEdge() {}
};

/** @brief Edge with N-dimensional measurement.
@tparam DIM dimension of the measurement vector.
*/
template <int DIM>
struct Edge : BaseEdge
{
	using Measurement = Array<double, DIM>;
	using Information = double;

	/** @brief The constructor.
	*/
	Edge() : measurement(Measurement()), information(Information()),
		vertexP(nullptr), vertexL(nullptr) {}

	/** @brief The constructor.
	@param m measurement vector.
	@param I information matrix.
	@param vertexP connected pose vertex.
	@param vertexL connected landmark vertex.
	*/
	Edge(const Measurement& m, Information I, PoseVertex* vertexP, LandmarkVertex* vertexL) :
		measurement(m), information(I), vertexP(vertexP), vertexL(vertexL) {}

	/** @brief Returns the connected pose vertex.
	*/
	PoseVertex* poseVertex() const override { return vertexP; }

	/** @brief Returns the connected landmark vertex.
	*/
	LandmarkVertex* landmarkVertex() const override { return vertexL; }

	/** @brief Returns the dimension of measurement.
	*/
	int dim() const override { return DIM; }

	Measurement measurement; //!< measurement vector.
	Information information; //!< information matrix (represented by a scalar for performance).
	PoseVertex* vertexP;     //!< connected pose vertex.
	LandmarkVertex* vertexL; //!< connected landmark vertex.
};

/** @brief Edge with 2-dimensional measurement (monocular observation).
*/
using MonoEdge = Edge<2>;

/** @brief Edge with 3-dimensional measurement (stereo observation).
*/
using StereoEdge = Edge<3>;

/** @brief Edge types
*/
enum class EdgeType
{
	MONOCULAR = 0,
	STEREO    = 1,
	COUNT     = 2
};

////////////////////////////////////////////////////////////////////////////////////
// Vertex
////////////////////////////////////////////////////////////////////////////////////

/** @brief Pose vertex struct.
*/
struct PoseVertex
{
	using Quaternion = Eigen::Quaterniond;
	using Rotation = Quaternion;
	using Translation = Array<double, 3>;

	/** @brief The constructor.
	*/
	PoseVertex() : q(Rotation()), t(Translation()), fixed(false), id(-1), iP(-1) {}

	/** @brief The constructor.
	@param id ID of the vertex.
	@param q rotational component of the pose, represented by quaternions.
	@param t translational component of the pose.
	@param camera camera parameters of the view.
	@param fixed if true, the state variables are fixed during optimization.
	*/
	PoseVertex(int id, const Rotation& q, const Translation& t, const CameraParams& camera, bool fixed = false)
		: q(q), t(t), camera(camera), fixed(fixed), id(id), iP(-1) {}

	Rotation q;              //!< rotational component of the pose, represented by quaternions.
	Translation t;           //!< translational component of the pose.
	CameraParams camera;     //!< camera parameters.
	bool fixed;              //!< if true, the state variables are fixed during optimization.
	int id;                  //!< ID of the vertex.
	int iP;                  //!< ID of the vertex (internally used).
	Set<BaseEdge*> edges;    //!< connected edges.
};

/** @brief Landmark vertex struct.
*/
struct LandmarkVertex
{
	using Point3D = Array<double, 3>;

	/** @brief The constructor.
	*/
	LandmarkVertex() : Xw(Point3D()), fixed(false), id(-1), iL(-1) {}

	/** @brief The constructor.
	@param id ID of the vertex.
	@param Xw 3D position of the landmark.
	@param fixed if true, the state variables are fixed during optimization.
	*/
	LandmarkVertex(int id, const Point3D& Xw, bool fixed = false)
		: Xw(Xw), fixed(fixed), id(id), iL(-1) {}

	Point3D Xw;              //!< 3D position of the landmark.
	bool fixed;              //!< if true, the state variables are fixed during optimization.
	int id;                  //!< ID of the vertex.
	int iL;                  //!< ID of the vertex (internally used).
	Set<BaseEdge*> edges;    //!< connected edges.
};

////////////////////////////////////////////////////////////////////////////////////
// Robust kernel
////////////////////////////////////////////////////////////////////////////////////
enum class RobustKernelType
{
	NONE  = 0,
	HUBER = 1,
	TUKEY = 2,
};

////////////////////////////////////////////////////////////////////////////////////
// Statistics
////////////////////////////////////////////////////////////////////////////////////

/** @brief information about optimization.
*/
struct BatchInfo
{
	int iteration;           //!< iteration number
	double chi2;             //!< total chi2 (objective function value)
};

using BatchStatistics = std::vector<BatchInfo>;

/** @brief Time profile.
*/
using TimeProfile = std::map<std::string, double>;

////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

using VertexP = PoseVertex;
using VertexL = LandmarkVertex;
using Edge2D = MonoEdge;
using Edge3D = StereoEdge;

} // namespace cuba

#endif // !__CUDA_BUNDLE_ADJUSTMENT_TYPES_H__
