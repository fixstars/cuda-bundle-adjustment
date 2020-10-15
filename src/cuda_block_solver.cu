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

#include "cuda_block_solver.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

namespace cuba
{
namespace gpu
{

////////////////////////////////////////////////////////////////////////////////////
// Type alias
////////////////////////////////////////////////////////////////////////////////////

template <int N>
using Vecxd = Vec<Scalar, N>;

template <int N>
using GpuVecxd = GpuVec<Vecxd<N>>;

using PxPBlockPtr = BlockPtr<Scalar, PDIM, PDIM>;
using LxLBlockPtr = BlockPtr<Scalar, LDIM, LDIM>;
using PxLBlockPtr = BlockPtr<Scalar, PDIM, LDIM>;
using Px1BlockPtr = BlockPtr<Scalar, PDIM, 1>;
using Lx1BlockPtr = BlockPtr<Scalar, LDIM, 1>;

////////////////////////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////////////////////////
constexpr int BLOCK_ACTIVE_ERRORS = 512;
constexpr int BLOCK_MAX_DIAGONAL = 512;
constexpr int BLOCK_COMPUTE_SCALE = 512;

__constant__ Scalar c_camera[5];
#define FX() c_camera[0]
#define FY() c_camera[1]
#define CX() c_camera[2]
#define CY() c_camera[3]
#define BF() c_camera[4]

////////////////////////////////////////////////////////////////////////////////////
// Type definitions
////////////////////////////////////////////////////////////////////////////////////
struct LessRowId
{
	__device__ bool operator()(const Vec3i& lhs, const Vec3i& rhs) const
	{
		if (lhs[0] == rhs[0])
			return lhs[1] < rhs[1];
		return lhs[0] < rhs[0];
	}
};

struct LessColId
{
	__device__ bool operator()(const Vec3i& lhs, const Vec3i& rhs) const
	{
		if (lhs[1] == rhs[1])
			return lhs[0] < rhs[0];
		return lhs[1] < rhs[1];
	}
};

template <typename T, int ROWS, int COLS>
struct MatView
{
	__device__ inline T& operator()(int i, int j) { return data[j * ROWS + i]; }
	__device__ inline MatView(T* data) : data(data) {}
	T* data;
};

template <typename T, int ROWS, int COLS>
struct ConstMatView
{
	__device__ inline T operator()(int i, int j) const { return data[j * ROWS + i]; }
	__device__ inline ConstMatView(const T* data) : data(data) {}
	const T* data;
};

template <typename T, int ROWS, int COLS>
struct Matx
{
	using View = MatView<T, ROWS, COLS>;
	using ConstView = ConstMatView<T, ROWS, COLS>;
	__device__ inline T& operator()(int i, int j) { return data[j * ROWS + i]; }
	__device__ inline T operator()(int i, int j) const { return data[j * ROWS + i]; }
	__device__ inline operator View() { return View(data); }
	__device__ inline operator ConstView() const { return ConstView(data); }
	T data[ROWS * COLS];
};

using MatView2x3d = MatView<Scalar, 2, 3>;
using MatView2x6d = MatView<Scalar, 2, 6>;
using MatView3x3d = MatView<Scalar, 3, 3>;
using MatView3x6d = MatView<Scalar, 3, 6>;
using ConstMatView3x3d = ConstMatView<Scalar, 3, 3>;

////////////////////////////////////////////////////////////////////////////////////
// Host functions
////////////////////////////////////////////////////////////////////////////////////
static int divUp(int total, int grain)
{
	return (total + grain - 1) / grain;
}

////////////////////////////////////////////////////////////////////////////////////
// Device functions (template matrix and verctor operation)
////////////////////////////////////////////////////////////////////////////////////

// assignment operations
using AssignOP = void(*)(Scalar*, Scalar);
__device__ inline void ASSIGN(Scalar* address, Scalar value) { *address = value; }
__device__ inline void ACCUM(Scalar* address, Scalar value) { *address += value; }
__device__ inline void DEACCUM(Scalar* address, Scalar value) { *address -= value; }
__device__ inline void ACCUM_ATOMIC(Scalar* address, Scalar value) { atomicAdd(address, value); }
__device__ inline void DEACCUM_ATOMIC(Scalar* address, Scalar value) { atomicAdd(address, -value); }

// recursive dot product for inline expansion
template <int N>
__device__ inline Scalar dot_(const Scalar* a, const Scalar* b)
{
	return dot_<N - 1>(a, b) + a[N - 1] * b[N - 1];
}

template <>
__device__ inline Scalar dot_<1>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }

// recursive dot product for inline expansion (strided access pattern)
template <int N, int S1, int S2>
__device__ inline Scalar dot_stride_(const Scalar* a, const Scalar* b)
{
	static_assert(S1 == PDIM || S1 == LDIM, "S1 must be PDIM or LDIM");
	static_assert(S2 == 1 || S2 == PDIM, "S2 must be 1 or PDIM");
	return dot_stride_<N - 1, S1, S2>(a, b) + a[S1 * (N - 1)] * b[S2 * (N - 1)];
}

template <>
__device__ inline Scalar dot_stride_<1, PDIM, 1>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }
template <>
__device__ inline Scalar dot_stride_<1, LDIM, 1>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }
template <>
__device__ inline Scalar dot_stride_<1, PDIM, PDIM>(const Scalar* a, const Scalar* b) { return a[0] * b[0]; }

// matrix(tansposed)-vector product: b = AT*x
template <int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatTMulVec(const Scalar* A, const Scalar* x, Scalar* b, Scalar omega)
{
#pragma unroll
	for (int i = 0; i < M; i++)
		OP(b + i, omega * dot_<N>(A + i * N, x));
}

// matrix(tansposed)-matrix product: C = AT*B
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatTMulMat(const Scalar* A, const Scalar* B, Scalar* C, Scalar omega)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatTMulVec<L, M, OP>(A, B + i * M, C + i * L, omega);
}

// matrix-vector product: b = A*x
template <int M, int N, int S = 1, AssignOP OP = ASSIGN>
__device__ inline void MatMulVec(const Scalar* A, const Scalar* x, Scalar* b)
{
#pragma unroll
	for (int i = 0; i < M; i++)
		OP(b + i, dot_stride_<N, M, S>(A + i, x));
}

// matrix-matrix product: C = A*B
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatMulMat(const Scalar* A, const Scalar* B, Scalar* C)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatMulVec<L, M, 1, OP>(A, B + i * M, C + i * L);
}

// matrix-matrix(tansposed) product: C = A*BT
template <int L, int M, int N, AssignOP OP = ASSIGN>
__device__ inline void MatMulMatT(const Scalar* A, const Scalar* B, Scalar* C)
{
#pragma unroll
	for (int i = 0; i < N; i++)
		MatMulVec<L, M, N, OP>(A, B + i, C + i * L);
}

// squared L2 norm
template <int N>
__device__ inline Scalar squaredNorm(const Scalar* x) { return dot_<N>(x, x); }
template <int N>
__device__ inline Scalar squaredNorm(const Vecxd<N>& x) { return squaredNorm<N>(x.data); }

// L2 norm
template <int N>
__device__ inline Scalar norm(const Scalar* x) { return sqrt(squaredNorm<N>(x)); }
template <int N>
__device__ inline Scalar norm(const Vecxd<N>& x) { return norm<N>(x.data); }

////////////////////////////////////////////////////////////////////////////////////
// Device functions
////////////////////////////////////////////////////////////////////////////////////
__device__ static inline void cross(const Vec4d& a, const Vec3d& b, Vec3d& c)
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline void rotate(const Vec4d& q, const Vec3d& Xw, Vec3d& Xc)
{
	Vec3d tmp1, tmp2;

	cross(q, Xw, tmp1);

	tmp1[0] += tmp1[0];
	tmp1[1] += tmp1[1];
	tmp1[2] += tmp1[2];

	cross(q, tmp1, tmp2);

	Xc[0] = Xw[0] + q[3] * tmp1[0] + tmp2[0];
	Xc[1] = Xw[1] + q[3] * tmp1[1] + tmp2[1];
	Xc[2] = Xw[2] + q[3] * tmp1[2] + tmp2[2];
}

__device__ inline void projectW2C(const Vec4d& q, const Vec3d& t, const Vec3d& Xw, Vec3d& Xc)
{
	rotate(q, Xw, Xc);
	Xc[0] += t[0];
	Xc[1] += t[1];
	Xc[2] += t[2];
}

template <int MDIM>
__device__ inline void projectC2I(const Vec3d& Xc, Vecxd<MDIM>& p)
{
}

template <>
__device__ inline void projectC2I<2>(const Vec3d& Xc, Vec2d& p)
{
	const Scalar invZ = 1 / Xc[2];
	p[0] = FX() * invZ * Xc[0] + CX();
	p[1] = FY() * invZ * Xc[1] + CY();
}

template <>
__device__ inline void projectC2I<3>(const Vec3d& Xc, Vec3d& p)
{
	const Scalar invZ = 1 / Xc[2];
	p[0] = FX() * invZ * Xc[0] + CX();
	p[1] = FY() * invZ * Xc[1] + CY();
	p[2] = p[0] - BF() * invZ;
}

__device__ inline void quaternionToRotationMatrix(const Vec4d& q, MatView3x3d R)
{
	const Scalar x = q[0];
	const Scalar y = q[1];
	const Scalar z = q[2];
	const Scalar w = q[3];

	const Scalar tx = 2 * x;
	const Scalar ty = 2 * y;
	const Scalar tz = 2 * z;
	const Scalar twx = tx * w;
	const Scalar twy = ty * w;
	const Scalar twz = tz * w;
	const Scalar txx = tx * x;
	const Scalar txy = ty * x;
	const Scalar txz = tz * x;
	const Scalar tyy = ty * y;
	const Scalar tyz = tz * y;
	const Scalar tzz = tz * z;

	R(0, 0) = 1 - (tyy + tzz);
	R(0, 1) = txy - twz;
	R(0, 2) = txz + twy;
	R(1, 0) = txy + twz;
	R(1, 1) = 1 - (txx + tzz);
	R(1, 2) = tyz - twx;
	R(2, 0) = txz - twy;
	R(2, 1) = tyz + twx;
	R(2, 2) = 1 - (txx + tyy);
}

template <int MDIM>
__device__ void computeJacobians(const Vec3d& Xc, const Vec4d& q,
	MatView<Scalar, MDIM, PDIM> JP, MatView<Scalar, MDIM, LDIM> JL)
{
}

template <>
__device__ void computeJacobians<2>(const Vec3d& Xc, const Vec4d& q, MatView2x6d JP, MatView2x3d JL)
{
	const Scalar X = Xc[0];
	const Scalar Y = Xc[1];
	const Scalar Z = Xc[2];
	const Scalar invZ = 1 / Z;
	const Scalar x = invZ * X;
	const Scalar y = invZ * Y;
	const Scalar fu = FX();
	const Scalar fv = FY();
	const Scalar fu_invZ = fu * invZ;
	const Scalar fv_invZ = fv * invZ;

	Matx<Scalar, 3, 3> R;
	quaternionToRotationMatrix(q, R);

	JL(0, 0) = -fu_invZ * (R(0, 0) - x * R(2, 0));
	JL(0, 1) = -fu_invZ * (R(0, 1) - x * R(2, 1));
	JL(0, 2) = -fu_invZ * (R(0, 2) - x * R(2, 2));
	JL(1, 0) = -fv_invZ * (R(1, 0) - y * R(2, 0));
	JL(1, 1) = -fv_invZ * (R(1, 1) - y * R(2, 1));
	JL(1, 2) = -fv_invZ * (R(1, 2) - y * R(2, 2));

	JP(0, 0) = +fu * x * y;
	JP(0, 1) = -fu * (1 + x * x);
	JP(0, 2) = +fu * y;
	JP(0, 3) = -fu_invZ;
	JP(0, 4) = 0;
	JP(0, 5) = +fu_invZ * x;

	JP(1, 0) = +fv * (1 + y * y);
	JP(1, 1) = -fv * x * y;
	JP(1, 2) = -fv * x;
	JP(1, 3) = 0;
	JP(1, 4) = -fv_invZ;
	JP(1, 5) = +fv_invZ * y;
}

template <>
__device__ void computeJacobians<3>(const Vec3d& Xc, const Vec4d& q, MatView3x6d JP, MatView3x3d JL)
{
	const Scalar X = Xc[0];
	const Scalar Y = Xc[1];
	const Scalar Z = Xc[2];
	const Scalar invZ = 1 / Z;
	const Scalar invZZ = invZ * invZ;
	const Scalar fu = FX();
	const Scalar fv = FY();
	const Scalar bf = BF();

	Matx<Scalar, 3, 3> R;
	quaternionToRotationMatrix(q, R);

	JL(0, 0) = -fu * R(0, 0) * invZ + fu * X * R(2, 0) * invZZ;
	JL(0, 1) = -fu * R(0, 1) * invZ + fu * X * R(2, 1) * invZZ;
	JL(0, 2) = -fu * R(0, 2) * invZ + fu * X * R(2, 2) * invZZ;

	JL(1, 0) = -fv * R(1, 0) * invZ + fv * Y * R(2, 0) * invZZ;
	JL(1, 1) = -fv * R(1, 1) * invZ + fv * Y * R(2, 1) * invZZ;
	JL(1, 2) = -fv * R(1, 2) * invZ + fv * Y * R(2, 2) * invZZ;

	JL(2, 0) = JL(0, 0) - bf * R(2, 0) * invZZ;
	JL(2, 1) = JL(0, 1) - bf * R(2, 1) * invZZ;
	JL(2, 2) = JL(0, 2) - bf * R(2, 2) * invZZ;

	JP(0, 0) = X * Y * invZZ * fu;
	JP(0, 1) = -(1 + (X * X * invZZ)) * fu;
	JP(0, 2) = Y * invZ * fu;
	JP(0, 3) = -1 * invZ * fu;
	JP(0, 4) = 0;
	JP(0, 5) = X * invZZ * fu;

	JP(1, 0) = (1 + Y * Y * invZZ) * fv;
	JP(1, 1) = -X * Y * invZZ * fv;
	JP(1, 2) = -X * invZ * fv;
	JP(1, 3) = 0;
	JP(1, 4) = -1 * invZ * fv;
	JP(1, 5) = Y * invZZ * fv;

	JP(2, 0) = JP(0, 0) - bf * Y * invZZ;
	JP(2, 1) = JP(0, 1) + bf * X * invZZ;
	JP(2, 2) = JP(0, 2);
	JP(2, 3) = JP(0, 3);
	JP(2, 4) = 0;
	JP(2, 5) = JP(0, 5) - bf * invZZ;
}

__device__ inline void Sym3x3Inv(ConstMatView3x3d A, MatView3x3d B)
{
	const Scalar A00 = A(0, 0);
	const Scalar A01 = A(0, 1);
	const Scalar A11 = A(1, 1);
	const Scalar A02 = A(2, 0);
	const Scalar A12 = A(1, 2);
	const Scalar A22 = A(2, 2);

	const Scalar det
		= A00 * A11 * A22
		+ A01 * A12 * A02
		+ A02 * A01 * A12
		- A00 * A12 * A12
		- A02 * A11 * A02
		- A01 * A01 * A22;

	const Scalar invDet = 1 / det;

	const Scalar B00 = invDet * (A11 * A22 - A12 * A12);
	const Scalar B01 = invDet * (A02 * A12 - A01 * A22);
	const Scalar B11 = invDet * (A00 * A22 - A02 * A02);
	const Scalar B02 = invDet * (A01 * A12 - A02 * A11);
	const Scalar B12 = invDet * (A02 * A01 - A00 * A12);
	const Scalar B22 = invDet * (A00 * A11 - A01 * A01);

	B(0, 0) = B00;
	B(0, 1) = B01;
	B(0, 2) = B02;
	B(1, 0) = B01;
	B(1, 1) = B11;
	B(1, 2) = B12;
	B(2, 0) = B02;
	B(2, 1) = B12;
	B(2, 2) = B22;
}

__device__ inline void skew1(Scalar x, Scalar y, Scalar z, MatView3x3d M)
{
	M(0, 0) = +0; M(0, 1) = -z; M(0, 2) = +y;
	M(1, 0) = +z; M(1, 1) = +0; M(1, 2) = -x;
	M(2, 0) = -y; M(2, 1) = +x; M(2, 2) = +0;
}

__device__ inline void skew2(Scalar x, Scalar y, Scalar z, MatView3x3d M)
{
	const Scalar xx = x * x;
	const Scalar yy = y * y;
	const Scalar zz = z * z;

	const Scalar xy = x * y;
	const Scalar yz = y * z;
	const Scalar zx = z * x;

	M(0, 0) = -yy - zz; M(0, 1) = +xy;      M(0, 2) = +zx;
	M(1, 0) = +xy;      M(1, 1) = -zz - xx; M(1, 2) = +yz;
	M(2, 0) = +zx;      M(2, 1) = +yz;      M(2, 2) = -xx - yy;
}

__device__ inline void addOmega(Scalar a1, ConstMatView3x3d O1, Scalar a2, ConstMatView3x3d O2,
	MatView3x3d R)
{
	R(0, 0) = 1 + a1 * O1(0, 0) + a2 * O2(0, 0);
	R(1, 0) = 0 + a1 * O1(1, 0) + a2 * O2(1, 0);
	R(2, 0) = 0 + a1 * O1(2, 0) + a2 * O2(2, 0);

	R(0, 1) = 0 + a1 * O1(0, 1) + a2 * O2(0, 1);
	R(1, 1) = 1 + a1 * O1(1, 1) + a2 * O2(1, 1);
	R(2, 1) = 0 + a1 * O1(2, 1) + a2 * O2(2, 1);

	R(0, 2) = 0 + a1 * O1(0, 2) + a2 * O2(0, 2);
	R(1, 2) = 0 + a1 * O1(1, 2) + a2 * O2(1, 2);
	R(2, 2) = 1 + a1 * O1(2, 2) + a2 * O2(2, 2);
}

__device__ inline void rotationMatrixToQuaternion(ConstMatView3x3d R, Vec4d& q)
{
	Scalar t = R(0, 0) + R(1, 1) + R(2, 2);
	if (t > 0)
	{
		t = sqrt(t + 1);
		q[3] = Scalar(0.5) * t;
		t = Scalar(0.5) / t;
		q[0] = (R(2, 1) - R(1, 2)) * t;
		q[1] = (R(0, 2) - R(2, 0)) * t;
		q[2] = (R(1, 0) - R(0, 1)) * t;
	}
	else
	{
		int i = 0;
		if (R(1, 1) > R(0, 0))
			i = 1;
		if (R(2, 2) > R(i, i))
			i = 2;
		int j = (i + 1) % 3;
		int k = (j + 1) % 3;

		t = sqrt(R(i, i) - R(j, j) - R(k, k) + 1);
		q[i] = Scalar(0.5) * t;
		t = Scalar(0.5) / t;
		q[3] = (R(k, j) - R(j, k)) * t;
		q[j] = (R(j, i) + R(i, j)) * t;
		q[k] = (R(k, i) + R(i, k)) * t;
	}
}

__device__ inline void multiplyQuaternion(const Vec4d& a, const Vec4d& b, Vec4d& c)
{
	c[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
	c[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
	c[1] = a[3] * b[1] + a[1] * b[3] + a[2] * b[0] - a[0] * b[2];
	c[2] = a[3] * b[2] + a[2] * b[3] + a[0] * b[1] - a[1] * b[0];
}

__device__ inline void normalizeQuaternion(const Vec4d& a, Vec4d& b)
{
	Scalar invn = 1 / norm(a);
	if (a[3] < 0)
		invn = -invn;

	for (int i = 0; i < 4; i++)
		b[i] = invn * a[i];
}

__device__ inline void updateExp(const Scalar* update, Vec4d& q, Vec3d& t)
{
	Vec3d omega(update);
	Vec3d upsilon(update + 3);

	const Scalar theta = norm(omega);

	Matx<Scalar, 3, 3> O1, O2;
	skew1(omega[0], omega[1], omega[2], O1);
	skew2(omega[0], omega[1], omega[2], O2);

	Scalar R[9], V[9];
	if (theta < Scalar(0.00001))
	{
		addOmega(Scalar(1.0), O1, Scalar(0.5), O2, R);
		addOmega(Scalar(0.5), O1, Scalar(1)/6, O2, V);
	}
	else
	{
		const Scalar a1 = sin(theta) / theta;
		const Scalar a2 = (1 - cos(theta)) / (theta * theta);
		const Scalar a3 = (theta - sin(theta)) / (pow(theta, 3));
		addOmega(a1, O1, a2, O2, R);
		addOmega(a2, O1, a3, O2, V);
	}

	rotationMatrixToQuaternion(R, q);
	MatMulVec<3, 3>(V, upsilon.data, t.data);
}

__device__ inline void updatePose(const Vec4d& q1, const Vec3d& t1, Vec4d& q2, Vec3d& t2)
{
	Vec3d u;
	rotate(q1, t2, u);

	for (int i = 0; i < 3; i++)
		t2[i] = t1[i] + u[i];

	Vec4d r;
	multiplyQuaternion(q1, q2, r);
	normalizeQuaternion(r, q2);
}

template <int N>
__device__ inline void copy(const Scalar* src, Scalar* dst)
{
	for (int i = 0; i < N; i++)
		dst[i] = src[i];
}

__device__ inline Vec3i makeVec3i(int i, int j, int k)
{
	Vec3i  vec;
	vec[0] = i;
	vec[1] = j;
	vec[2] = k;
	return vec;
}

////////////////////////////////////////////////////////////////////////////////////
// Kernel functions
////////////////////////////////////////////////////////////////////////////////////
template <int MDIM>
__global__ void computeActiveErrorsKernel(int nedges,
	const Vec4d* qs, const Vec3d* ts, const Vec3d* Xws, const Vecxd<MDIM>* measurements,
	const Scalar* omegas, const Vec2i* edge2PL, Vecxd<MDIM>* errors, Vec3d* Xcs, Scalar* chi)
{
	using Vecmd = Vecxd<MDIM>;

	const int sharedIdx = threadIdx.x;
	__shared__ Scalar cache[BLOCK_ACTIVE_ERRORS];

	Scalar sumchi = 0;
	for (int iE = blockIdx.x * blockDim.x + threadIdx.x; iE < nedges; iE += gridDim.x * blockDim.x)
	{
		const Vec2i index = edge2PL[iE];
		const int iP = index[0];
		const int iL = index[1];

		const Vec4d& q = qs[iP];
		const Vec3d& t = ts[iP];
		const Vec3d& Xw = Xws[iL];
		const Vecmd& measurement = measurements[iE];

		// project world to camera
		Vec3d Xc;
		projectW2C(q, t, Xw, Xc);

		// project camera to image
		Vecmd proj;
		projectC2I(Xc, proj);

		// compute residual
		Vecmd error;
		for (int i = 0; i < MDIM; i++)
			error[i] = proj[i] - measurement[i];

		errors[iE] = error;
		Xcs[iE] = Xc;

		sumchi += omegas[iE] * squaredNorm(error);
	}

	cache[sharedIdx] = sumchi;
	__syncthreads();

	for (int stride = BLOCK_ACTIVE_ERRORS / 2; stride > 0; stride >>= 1)
	{
		if (sharedIdx < stride)
			cache[sharedIdx] += cache[sharedIdx + stride];
		__syncthreads();
	}

	if (sharedIdx == 0)
		atomicAdd(chi, cache[0]);
}

template <int MDIM>
__global__ void constructQuadraticFormKernel(int nedges,
	const Vec3d* Xcs, const Vec4d* qs, const Vecxd<MDIM>* errors,
	const Scalar* omegas, const Vec2i* edge2PL, const int* edge2Hpl, const uint8_t* flags,
	PxPBlockPtr Hpp, Px1BlockPtr bp, LxLBlockPtr Hll, Lx1BlockPtr bl, PxLBlockPtr Hpl)
{
	using Vecmd = Vecxd<MDIM>;

	const int iE = blockIdx.x * blockDim.x + threadIdx.x;
	if (iE >= nedges)
		return;

	const Scalar omega = omegas[iE];
	const int iP = edge2PL[iE][0];
	const int iL = edge2PL[iE][1];
	const int iPL = edge2Hpl[iE];
	const int flag = flags[iE];

	const Vec4d& q = qs[iP];
	const Vec3d& Xc = Xcs[iE];
	const Vecmd& error = errors[iE];

	// compute Jacobians
	Scalar JP[MDIM * PDIM];
	Scalar JL[MDIM * LDIM];
	computeJacobians<MDIM>(Xc, q, JP, JL);

	if (!(flag & EDGE_FLAG_FIXED_P))
	{
		// Hpp += = JPT*Ω*JP
		MatTMulMat<PDIM, MDIM, PDIM, ACCUM_ATOMIC>(JP, JP, Hpp.at(iP), omega);

		// bp += = JPT*Ω*r
		MatTMulVec<PDIM, MDIM, ACCUM_ATOMIC>(JP, error.data, bp.at(iP), omega);
	}
	if (!(flag & EDGE_FLAG_FIXED_L))
	{
		// Hll += = JLT*Ω*JL
		MatTMulMat<LDIM, MDIM, LDIM, ACCUM_ATOMIC>(JL, JL, Hll.at(iL), omega);

		// bl += = JLT*Ω*r
		MatTMulVec<LDIM, MDIM, ACCUM_ATOMIC>(JL, error.data, bl.at(iL), omega);
	}
	if (!flag)
	{
		// Hpl += = JPT*Ω*JL
		MatTMulMat<PDIM, MDIM, LDIM, ASSIGN>(JP, JL, Hpl.at(iPL), omega);
	}
}

template <int DIM>
__global__ void maxDiagonalKernel(int size, const Scalar* D, Scalar* maxD)
{
	const int sharedIdx = threadIdx.x;
	__shared__ Scalar cache[BLOCK_MAX_DIAGONAL];

	Scalar maxVal = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
	{
		const int j = i / DIM;
		const int k = i % DIM;
		const Scalar* ptrBlock = D + j * DIM * DIM;
		maxVal = max(maxVal, ptrBlock[k * DIM + k]);
	}

	cache[sharedIdx] = maxVal;
	__syncthreads();

	for (int stride = BLOCK_MAX_DIAGONAL / 2; stride > 0; stride >>= 1)
	{
		if (sharedIdx < stride)
			cache[sharedIdx] = max(cache[sharedIdx], cache[sharedIdx + stride]);
		__syncthreads();
	}

	if (sharedIdx == 0)
		maxD[blockIdx.x] = cache[0];
}

template <int DIM>
__global__ void addLambdaKernel(int size, Scalar* D, Scalar lambda, Scalar* backup)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	const int j = i / DIM;
	const int k = i % DIM;
	Scalar* ptrBlock = D + j * DIM * DIM;
	backup[i] = ptrBlock[k * DIM + k];
	ptrBlock[k * DIM + k] += lambda;
}

template <int DIM>
__global__ void restoreDiagonalKernel(int size, Scalar* D, const Scalar* backup)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	const int j = i / DIM;
	const int k = i % DIM;
	Scalar* ptrBlock = D + j * DIM * DIM;
	ptrBlock[k * DIM + k] = backup[i];
}

__global__ void computeBschureKernel(int cols, LxLBlockPtr Hll, LxLBlockPtr invHll,
	Lx1BlockPtr bl, PxLBlockPtr Hpl, const int* HplColPtr, const int* HplRowInd,
	Px1BlockPtr bsc, PxLBlockPtr Hpl_invHll)
{
	const int colId = blockIdx.x * blockDim.x + threadIdx.x;
	if (colId >= cols)
		return;

	Scalar iHll[LDIM * LDIM];
	Scalar Hpl_iHll[PDIM * LDIM];

	Sym3x3Inv(Hll.at(colId), iHll);
	copy<LDIM * LDIM>(iHll, invHll.at(colId));

	for (int i = HplColPtr[colId]; i < HplColPtr[colId + 1]; i++)
	{
		MatMulMat<6, 3, 3>(Hpl.at(i), iHll, Hpl_iHll);
		MatMulVec<6, 3, 1, DEACCUM_ATOMIC>(Hpl_iHll, bl.at(colId), bsc.at(HplRowInd[i]));
		copy<PDIM * LDIM>(Hpl_iHll, Hpl_invHll.at(i));
	}
}

__global__ void initializeHschurKernel(int rows, PxPBlockPtr Hpp, PxPBlockPtr Hsc, const int* HscRowPtr)
{
	const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowId >= rows)
		return;

	copy<PDIM * PDIM>(Hpp.at(rowId), Hsc.at(HscRowPtr[rowId]));
}

__global__ void computeHschureKernel(int size, const Vec3i* mulBlockIds,
	PxLBlockPtr Hpl_invHll, PxLBlockPtr Hpl, PxPBlockPtr Hschur)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= size)
		return;

	const Vec3i index = mulBlockIds[tid];
	Scalar A[PDIM * LDIM];
	Scalar B[PDIM * LDIM];
	copy<PDIM * LDIM>(Hpl_invHll.at(index[0]), A);
	copy<PDIM * LDIM>(Hpl.at(index[1]), B);
	MatMulMatT<6, 3, 6, DEACCUM_ATOMIC>(A, B, Hschur.at(index[2]));
}

__global__ void findHschureMulBlockIndicesKernel(int cols, const int* HplColPtr, const int* HplRowInd,
	const int* HscRowPtr, const int* HscColInd, Vec3i* mulBlockIds, int* nindices)
{
	const int colId = blockIdx.x * blockDim.x + threadIdx.x;
	if (colId >= cols)
		return;

	const int i0 = HplColPtr[colId];
	const int i1 = HplColPtr[colId + 1];
	for (int i = i0; i < i1; i++)
	{
		const int iP1 = HplRowInd[i];
		int k = HscRowPtr[iP1];
		for (int j = i; j < i1; j++)
		{
			const int iP2 = HplRowInd[j];
			while (HscColInd[k] < iP2) k++;
			const int pos = atomicAdd(nindices, 1);
			mulBlockIds[pos] = makeVec3i(i, j, k);
		}
	}
}

__global__ void permuteNnzPerRowKernel(int size, const int* srcRowPtr, const int* P, int* nnzPerRow)
{
	const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowId >= size)
		return;

	nnzPerRow[P[rowId]] = srcRowPtr[rowId + 1] - srcRowPtr[rowId];
}

__global__ void permuteColIndKernel(int size, const int* srcRowPtr, const int* srcColInd, const int* P,
	int* dstColInd, int* dstMap, int* nnzPerRow)
{
	const int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	if (rowId >= size)
		return;

	const int i0 = srcRowPtr[rowId];
	const int i1 = srcRowPtr[rowId + 1];
	const int permRowId = P[rowId];
	for (int srck = i0; srck < i1; srck++)
	{
		const int dstk = nnzPerRow[permRowId]++;
		dstColInd[dstk] = P[srcColInd[srck]];
		dstMap[dstk] = srck;
	}
}

__global__ void schurComplementPostKernel(int cols, LxLBlockPtr invHll, Lx1BlockPtr bl, PxLBlockPtr Hpl,
	const int* HplColPtr, const int* HplRowInd, Px1BlockPtr xp, Lx1BlockPtr xl)
{
	const int colId = blockIdx.x * blockDim.x + threadIdx.x;
	if (colId >= cols)
		return;

	Scalar cl[LDIM];
	copy<LDIM>(bl.at(colId), cl);

	for (int i = HplColPtr[colId]; i < HplColPtr[colId + 1]; i++)
		MatTMulVec<3, 6, DEACCUM>(Hpl.at(i), xp.at(HplRowInd[i]), cl, 1);

	MatMulVec<3, 3>(invHll.at(colId), cl, xl.at(colId));
}

__global__ void updatePosesKernel(int size, Px1BlockPtr xp, Vec4d* qs, Vec3d* ts)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	Vec4d expq;
	Vec3d expt;
	updateExp(xp.at(i), expq, expt);
	updatePose(expq, expt, qs[i], ts[i]);
}

__global__ void updateLandmarksKernel(int size, Lx1BlockPtr xl, Vec3d* Xws)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;

	const Scalar* dXw = xl.at(i);
	Vec3d& Xw = Xws[i];
	Xw[0] += dXw[0];
	Xw[1] += dXw[1];
	Xw[2] += dXw[2];
}

__global__ void computeScaleKernel(const Scalar* x, const Scalar* b, Scalar* scale, Scalar lambda, int size)
{
	const int sharedIdx = threadIdx.x;
	__shared__ Scalar cache[BLOCK_COMPUTE_SCALE];

	Scalar sum = 0;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
		sum += x[i] * (lambda * x[i] + b[i]);

	cache[sharedIdx] = sum;
	__syncthreads();

	for (int stride = BLOCK_COMPUTE_SCALE / 2; stride > 0; stride >>= 1)
	{
		if (sharedIdx < stride)
			cache[sharedIdx] += cache[sharedIdx + stride];
		__syncthreads();
	}

	if (sharedIdx == 0)
		atomicAdd(scale, cache[0]);
}

__global__ void convertBSRToCSRKernel(int size, const Scalar* src, Scalar* dst, const int* map)
{
	const int dstk = blockIdx.x * blockDim.x + threadIdx.x;
	if (dstk >= size)
		return;

	dst[dstk] = src[map[dstk]];
}

__global__ void nnzPerColKernel(const Vec3i* blockpos, int nblocks, int* nnzPerCol)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nblocks)
		return;

	const int colId = blockpos[i][1];
	atomicAdd(&nnzPerCol[colId], 1);
}

__global__ void setRowIndKernel(const Vec3i* blockpos, int nblocks, int* rowInd, int* indexPL)
{
	const int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= nblocks)
		return;

	const int rowId = blockpos[k][0];
	const int edgeId = blockpos[k][2];
	rowInd[k] = rowId;
	indexPL[edgeId] = k;
}

////////////////////////////////////////////////////////////////////////////////////
// Public functions
////////////////////////////////////////////////////////////////////////////////////

void waitForKernelCompletion()
{
	CUDA_CHECK(cudaDeviceSynchronize());
}

void setCameraParameters(const Scalar* camera)
{
	CUDA_CHECK(cudaMemcpyToSymbol(c_camera, camera, sizeof(Scalar) * 5));
}

void exclusiveScan(const int* src, int* dst, int size)
{
	auto ptrSrc = thrust::device_pointer_cast(src);
	auto ptrDst = thrust::device_pointer_cast(dst);
	thrust::exclusive_scan(ptrSrc, ptrSrc + size, ptrDst);
}

void buildHplStructure(GpuVec3i& blockpos, GpuHplBlockMat& Hpl, GpuVec1i& indexPL, GpuVec1i& nnzPerCol)
{
	const int nblocks = Hpl.nnz();
	const int block = 1024;
	const int grid = divUp(nblocks, block);
	int* colPtr = Hpl.outerIndices();
	int* rowInd = Hpl.innerIndices();

	auto ptrBlockPos = thrust::device_pointer_cast(blockpos.data());
	thrust::sort(ptrBlockPos, ptrBlockPos + nblocks, LessColId());

	CUDA_CHECK(cudaMemset(nnzPerCol, 0, sizeof(int) * (Hpl.cols() + 1)));
	nnzPerColKernel<<<grid, block>>>(blockpos, nblocks, nnzPerCol);
	exclusiveScan(nnzPerCol, colPtr, Hpl.cols() + 1);
	setRowIndKernel<<<grid, block>>>(blockpos, nblocks, rowInd, indexPL);
}

void findHschureMulBlockIndices(const GpuHplBlockMat& Hpl, const GpuHscBlockMat& Hsc,
	GpuVec3i& mulBlockIds)
{
	const int block = 1024;
	const int grid = divUp(Hpl.cols(), block);

	DeviceBuffer<int> nindices(1);
	nindices.fillZero();

	findHschureMulBlockIndicesKernel<<<grid, block>>>(Hpl.cols(), Hpl.outerIndices(), Hpl.innerIndices(),
		Hsc.outerIndices(), Hsc.innerIndices(), mulBlockIds, nindices);
	CUDA_CHECK(cudaGetLastError());

	auto ptrSrc = thrust::device_pointer_cast(mulBlockIds.data());
	thrust::sort(ptrSrc, ptrSrc + mulBlockIds.size(), LessRowId());
}

template <int M>
Scalar computeActiveErrors_(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec3d& Xws,
	const GpuVecxd<M>& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	GpuVecxd<M>& errors, GpuVec3d& Xcs, Scalar* chi)
{
	const int nedges = measurements.ssize();
	const int block = BLOCK_ACTIVE_ERRORS;
	const int grid = 16;

	if (nedges <= 0)
		return 0;

	CUDA_CHECK(cudaMemset(chi, 0, sizeof(Scalar)));
	computeActiveErrorsKernel<M><<<grid, block>>>(nedges, qs, ts, Xws, measurements, omegas,
		edge2PL, errors, Xcs, chi);
	CUDA_CHECK(cudaGetLastError());

	Scalar h_chi = 0;
	CUDA_CHECK(cudaMemcpy(&h_chi, chi, sizeof(Scalar), cudaMemcpyDeviceToHost));

	return h_chi;
}

Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec3d& Xws,
	const GpuVec2d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	GpuVec2d& errors, GpuVec3d& Xcs, Scalar* chi)
{
	return computeActiveErrors_(qs, ts, Xws, measurements, omegas, edge2PL, errors, Xcs, chi);
}

Scalar computeActiveErrors(const GpuVec4d& qs, const GpuVec3d& ts, const GpuVec3d& Xws,
	const GpuVec3d& measurements, const GpuVec1d& omegas, const GpuVec2i& edge2PL,
	GpuVec3d& errors, GpuVec3d& Xcs, Scalar* chi)
{
	return computeActiveErrors_(qs, ts, Xws, measurements, omegas, edge2PL, errors, Xcs, chi);
}

template <int M>
void constructQuadraticForm_(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVecxd<M>& errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1b& flags,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl)
{
	const int nedges = errors.ssize();
	const int block = 512;
	const int grid = divUp(nedges, block);

	if (nedges <= 0)
		return;

	constructQuadraticFormKernel<M><<<grid, block>>>(nedges, Xcs, qs, errors, omegas, edge2PL,
		edge2Hpl, flags, Hpp, bp, Hll, bl, Hpl);
	CUDA_CHECK(cudaGetLastError());
}

void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec2d& errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1b& flags,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl)
{
	constructQuadraticForm_(Xcs, qs, errors, omegas, edge2PL, edge2Hpl, flags, Hpp, bp, Hll, bl, Hpl);
}

void constructQuadraticForm(const GpuVec3d& Xcs, const GpuVec4d& qs, const GpuVec3d& errors,
	const GpuVec1d& omegas, const GpuVec2i& edge2PL, const GpuVec1i& edge2Hpl, const GpuVec1b& flags,
	GpuPxPBlockVec& Hpp, GpuPx1BlockVec& bp, GpuLxLBlockVec& Hll, GpuLx1BlockVec& bl, GpuHplBlockMat& Hpl)
{
	constructQuadraticForm_(Xcs, qs, errors, omegas, edge2PL, edge2Hpl, flags, Hpp, bp, Hll, bl, Hpl);
}

template <typename T, int DIM>
Scalar maxDiagonal_(const DeviceBlockVector<T, DIM, DIM>& D, Scalar* maxD)
{
	constexpr int block = BLOCK_MAX_DIAGONAL;
	constexpr int grid = 4;
	const int size = D.size() * DIM;

	maxDiagonalKernel<DIM><<<grid, block>>>(size, D.values(), maxD);
	CUDA_CHECK(cudaGetLastError());

	Scalar tmpMax[grid];
	CUDA_CHECK(cudaMemcpy(tmpMax, maxD, sizeof(Scalar) * grid, cudaMemcpyDeviceToHost));

	Scalar maxv = 0;
	for (int i = 0; i < grid; i++)
		maxv = std::max(maxv, tmpMax[i]);

	return maxv;
}

Scalar maxDiagonal(const GpuPxPBlockVec& Hpp, Scalar* maxD)
{
	return maxDiagonal_(Hpp, maxD);
}

Scalar maxDiagonal(const GpuLxLBlockVec& Hll, Scalar* maxD)
{
	return maxDiagonal_(Hll, maxD);
}

template <typename T, int DIM>
void addLambda_(DeviceBlockVector<T, DIM, DIM>& D, Scalar lambda, DeviceBlockVector<T, DIM, 1>& backup)
{
	const int size = D.size() * DIM;
	const int block = 1024;
	const int grid = divUp(size, block);
	addLambdaKernel<DIM><<<grid, block>>>(size, D.values(), lambda, backup.values());
	CUDA_CHECK(cudaGetLastError());
}

void addLambda(GpuPxPBlockVec& Hpp, Scalar lambda, GpuPx1BlockVec& backup)
{
	addLambda_(Hpp, lambda, backup);
}

void addLambda(GpuLxLBlockVec& Hll, Scalar lambda, GpuLx1BlockVec& backup)
{
	addLambda_(Hll, lambda, backup);
}

template <typename T, int DIM>
void restoreDiagonal_(DeviceBlockVector<T, DIM, DIM>& D, const DeviceBlockVector<T, DIM, 1>& backup)
{
	const int size = D.size() * DIM;
	const int block = 1024;
	const int grid = divUp(size, block);
	restoreDiagonalKernel<DIM><<<grid, block>>>(size, D.values(), backup.values());
	CUDA_CHECK(cudaGetLastError());
}

void restoreDiagonal(GpuPxPBlockVec& Hpp, const GpuPx1BlockVec& backup)
{
	restoreDiagonal_(Hpp, backup);
}

void restoreDiagonal(GpuLxLBlockVec& Hll, const GpuLx1BlockVec& backup)
{
	restoreDiagonal_(Hll, backup);
}

void computeBschure(const GpuPx1BlockVec& bp, const GpuHplBlockMat& Hpl, const GpuLxLBlockVec& Hll,
	const GpuLx1BlockVec& bl, GpuPx1BlockVec& bsc, GpuLxLBlockVec& invHll, GpuPxLBlockVec& Hpl_invHll)
{
	const int cols = Hll.size();
	const int block = 256;
	const int grid = divUp(cols, block);

	bp.copyTo(bsc);
	computeBschureKernel<<<grid, block>>>(cols, Hll, invHll, bl, Hpl, Hpl.outerIndices(), Hpl.innerIndices(),
		bsc, Hpl_invHll);
	CUDA_CHECK(cudaGetLastError());
}

void computeHschure(const GpuPxPBlockVec& Hpp, const GpuPxLBlockVec& Hpl_invHll,
	const GpuHplBlockMat& Hpl, const GpuVec3i& mulBlockIds, GpuHscBlockMat& Hsc)
{
	const int nmulBlocks = mulBlockIds.ssize();
	const int block = 256;
	const int grid1 = divUp(Hsc.rows(), block);
	const int grid2 = divUp(nmulBlocks, block);

	Hsc.fillZero();
	initializeHschurKernel<<<grid1, block>>>(Hsc.rows(), Hpp, Hsc, Hsc.outerIndices());
	computeHschureKernel<<<grid2, block>>>(nmulBlocks, mulBlockIds, Hpl_invHll, Hpl, Hsc);
	CUDA_CHECK(cudaGetLastError());
}

void convertHschureBSRToCSR(const GpuHscBlockMat& HscBSR, const GpuVec1i& BSR2CSR, GpuVec1d& HscCSR)
{
	const int size = HscCSR.ssize();
	const int block = 1024;
	const int grid = divUp(size, block);
	convertBSRToCSRKernel<<<grid, block>>>(size, HscBSR.values(), HscCSR, BSR2CSR);
}

void twistCSR(int size, int nnz, const int* srcRowPtr, const int* srcColInd, const int* P,
	int* dstRowPtr, int* dstColInd, int* dstMap, int* nnzPerRow)
{
	const int block = 512;
	const int grid = divUp(size, block);

	permuteNnzPerRowKernel<<<grid, block>>>(size, srcRowPtr, P, nnzPerRow);
	exclusiveScan(nnzPerRow, dstRowPtr, size + 1);
	CUDA_CHECK(cudaMemcpy(nnzPerRow, dstRowPtr, sizeof(int) * (size + 1), cudaMemcpyDeviceToDevice));
	permuteColIndKernel<<<grid, block>>>(size, srcRowPtr, srcColInd, P, dstColInd, dstMap, nnzPerRow);
}

void permute(int size, const Scalar* src, Scalar* dst, const int* P)
{
	auto ptrSrc = thrust::device_pointer_cast(src);
	auto ptrDst = thrust::device_pointer_cast(dst);
	auto ptrMap = thrust::device_pointer_cast(P);
	thrust::gather(ptrMap, ptrMap + size, ptrSrc, ptrDst);
}

void schurComplementPost(const GpuLxLBlockVec& invHll, const GpuLx1BlockVec& bl,
	const GpuHplBlockMat& Hpl, const GpuPx1BlockVec& xp, GpuLx1BlockVec& xl)
{
	const int block = 1024;
	const int grid = divUp(Hpl.cols(), block);

	schurComplementPostKernel<<<grid, block>>>(Hpl.cols(), invHll, bl, Hpl,
		Hpl.outerIndices(), Hpl.innerIndices(),xp, xl);
	CUDA_CHECK(cudaGetLastError());
}

void updatePoses(const GpuPx1BlockVec& xp, GpuVec4d& qs, GpuVec3d& ts)
{
	const int block = 256;
	const int grid = divUp(xp.size(), block);
	updatePosesKernel<<<grid, block>>>(xp.size(), xp, qs, ts);
	CUDA_CHECK(cudaGetLastError());
}

void updateLandmarks(const GpuLx1BlockVec& xl, GpuVec3d& Xws)
{
	const int block = 1024;
	const int grid = divUp(xl.size(), block);
	updateLandmarksKernel<<<grid, block>>>(xl.size(), xl, Xws);
	CUDA_CHECK(cudaGetLastError());
}

void computeScale(const GpuVec1d& x, const GpuVec1d& b, Scalar* scale, Scalar lambda)
{
	const int block = BLOCK_COMPUTE_SCALE;
	const int grid = 4;

	CUDA_CHECK(cudaMemset(scale, 0, sizeof(Scalar)));
	computeScaleKernel<<<grid, block>>>(x, b, scale, lambda, x.ssize());
	CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace cuba
