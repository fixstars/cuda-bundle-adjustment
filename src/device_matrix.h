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

#ifndef __DEVICE_MATRIX_H__
#define __DEVICE_MATRIX_H__

#include "scalar.h"
#include "constants.h"
#include "fixed_vector.h"
#include "device_buffer.h"

namespace cuba
{

template <typename T, int BLOCK_ROWS, int BLOCK_COLS>
class BlockPtr
{
public:

	static const int BLOCK_AREA = BLOCK_ROWS * BLOCK_COLS;
	__device__ BlockPtr(T* data) : data_(data) {}
	__device__ T* at(int i) { return data_ + i * BLOCK_AREA; }
	__device__ const T* at(int i) const { return data_ + i * BLOCK_AREA; }

private:

	T* data_;
};

template <typename T, int BLOCK_ROWS, int BLOCK_COLS, int ORDER>
class DeviceBlockMatrix
{
public:

	static const int BLOCK_AREA = BLOCK_ROWS * BLOCK_COLS;
	using BlockPtrT = BlockPtr<T, BLOCK_ROWS, BLOCK_COLS>;

	DeviceBlockMatrix() : rows_(0), cols_(0), nnz_(0), outerSize_(0), innerSize_(0) {}
	DeviceBlockMatrix(int rows, int cols) : nnz_(0) { resize(rows, cols); }

	void resize(int rows, int cols)
	{
		rows_ = rows;
		cols_ = cols;
		outerSize_ = ORDER == ROW_MAJOR ? rows : cols;
		innerSize_ = ORDER == ROW_MAJOR ? cols : rows;
		outerIndices_.resize(outerSize_ + 1);
	}

	void resizeNonZeros(int nnz)
	{
		nnz_ = nnz;
		values_.resize(nnz * BLOCK_AREA);
		innerIndices_.resize(nnz);
	}

	void upload(const T* values, const int* outerIndices, const int* innerIndices)
	{
		if (values)
			values_.upload(values);
		if (outerIndices)
			outerIndices_.upload(outerIndices);
		if (innerIndices)
			innerIndices_.upload(innerIndices);
	}

	void download(T* values, int* outerIndices, int* innerIndices) const
	{
		if (values)
			values_.download(values);
		if (outerIndices)
			outerIndices_.download(outerIndices);
		if (innerIndices)
			innerIndices_.download(innerIndices);
	}

	void fillZero()
	{
		values_.fillZero();
	}

	T* values() { return values_.data(); }
	int* outerIndices() { return outerIndices_.data(); }
	int* innerIndices() { return innerIndices_.data(); }

	const T* values() const { return values_.data(); }
	const int* outerIndices() const { return outerIndices_.data(); }
	const int* innerIndices() const { return innerIndices_.data(); }

	int rows() const { return rows_; }
	int cols() const { return cols_; }
	int nnz() const { return nnz_; }

	operator BlockPtrT() const { return BlockPtrT((T*)values_.data()); }

private:

	DeviceBuffer<T> values_;
	DeviceBuffer<int> outerIndices_, innerIndices_;
	int rows_, cols_, nnz_, outerSize_, innerSize_;
};

template <typename T, int BLOCK_ROWS, int BLOCK_COLS>
class DeviceBlockVector
{
public:

	static const int BLOCK_AREA = BLOCK_ROWS * BLOCK_COLS;
	using BlockPtrT = BlockPtr<T, BLOCK_ROWS, BLOCK_COLS>;

	DeviceBlockVector() : size_(0) {}
	DeviceBlockVector(int size) { resize(size); }

	void resize(int size)
	{
		size_ = size;
		values_.resize(size * BLOCK_AREA);
	}

	void map(int size, T* data)
	{
		size_ = size;
		values_.map(size * BLOCK_AREA, data);
	}

	void fillZero()
	{
		values_.fillZero();
	}

	void copyTo(DeviceBlockVector& rhs) const
	{
		values_.copyTo(rhs.values());
	}

	T* values() { return values_.data(); }
	const T* values() const { return values_.data(); }

	int size() const { return size_; }
	int elemSize() const { return size_ * BLOCK_AREA; }

	operator BlockPtrT() const { return BlockPtrT((T*)values_.data()); }

private:

	DeviceBuffer<T> values_;
	int size_;
};

template <typename T>
using GpuVec = DeviceBuffer<T>;

using GpuVec1d = GpuVec<Scalar>;
using GpuVec2d = GpuVec<Vec2d>;
using GpuVec3d = GpuVec<Vec3d>;
using GpuVec4d = GpuVec<Vec4d>;
using GpuVec5d = GpuVec<Vec5d>;

using GpuVec1i = GpuVec<int>;
using GpuVec2i = GpuVec<Vec2i>;
using GpuVec3i = GpuVec<Vec3i>;
using GpuVec4i = GpuVec<Vec4i>;

using GpuVec1b = GpuVec<uint8_t>;

using GpuHplBlockMat = DeviceBlockMatrix<Scalar, PDIM, LDIM, COL_MAJOR>;
using GpuHscBlockMat = DeviceBlockMatrix<Scalar, PDIM, PDIM, ROW_MAJOR>;

using GpuPxPBlockVec = DeviceBlockVector<Scalar, PDIM, PDIM>;
using GpuLxLBlockVec = DeviceBlockVector<Scalar, LDIM, LDIM>;
using GpuPxLBlockVec = DeviceBlockVector<Scalar, PDIM, LDIM>;
using GpuPx1BlockVec = DeviceBlockVector<Scalar, PDIM, 1>;
using GpuLx1BlockVec = DeviceBlockVector<Scalar, LDIM, 1>;

class GpuVecAny
{
public:

	GpuVecAny() : ptr_(nullptr) {}
	template <typename T> GpuVecAny(const GpuVec<T>& vec) : ptr_((void*)&vec) {}
	template <typename T> GpuVec<T>& getRef() const { return *((GpuVec<T>*)ptr_); }
	template <typename T> const GpuVec<T>& getCRef() const { return *((GpuVec<T>*)ptr_); }

private:

	void* ptr_;
};

} // namespace cuba

#endif // !__DEVICE_MATRIX_H__
