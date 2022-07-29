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

#ifndef __FIXED_VECTOR_H__
#define __FIXED_VECTOR_H__

#include <cstdint>
#include <cuda_runtime.h>

#include "scalar.h"

namespace cuba
{

#define HOST_DEVICE __host__ __device__ inline

template <typename T, int N>
struct Vec
{
	HOST_DEVICE Vec() {}
	HOST_DEVICE Vec(const T* values) { for (int i = 0; i < N; i++) data[i] = values[i]; }

	template <typename U>
	HOST_DEVICE Vec(const U* values) { for (int i = 0; i < N; i++) data[i] = T(values[i]); }

	HOST_DEVICE T& operator[](int i) { return data[i]; }
	HOST_DEVICE const T& operator[](int i) const { return data[i]; }

	HOST_DEVICE void copyTo(T* rhs) const { for (int i = 0; i < N; i++) rhs[i] = data[i]; }

	template <typename U>
	HOST_DEVICE void copyTo(U* rhs) const { for (int i = 0; i < N; i++) rhs[i] = U(data[i]); }

	T data[N];
};

using Vec2d = Vec<Scalar, 2>;
using Vec3d = Vec<Scalar, 3>;
using Vec4d = Vec<Scalar, 4>;
using Vec5d = Vec<Scalar, 5>;

using Vec2i = Vec<int, 2>;
using Vec3i = Vec<int, 3>;
using Vec4i = Vec<int, 4>;

} // namespace cuba

#endif // !__FIXED_VECTOR_H__
