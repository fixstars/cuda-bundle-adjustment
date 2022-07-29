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

#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#include <cuda_runtime.h>

#include "macro.h"

namespace cuba
{

template <typename T>
class DeviceBuffer
{
public:

	DeviceBuffer() : data_(nullptr), size_(0), capacity_(0), allocated_(false) {}
	DeviceBuffer(size_t size) : data_(nullptr), size_(0), capacity_(0), allocated_(false) { resize(size); }
	~DeviceBuffer() { destroy(); }

	void allocate(size_t count)
	{
		if (data_ && capacity_ >= count)
			return;

		destroy();
		CUDA_CHECK(cudaMalloc(&data_, sizeof(T) * count));
		capacity_ = count;
		allocated_ = true;
	}

	void destroy()
	{
		if (allocated_ && data_)
			CUDA_CHECK(cudaFree(data_));
		data_ = nullptr;
		size_ = 0;
		allocated_ = false;
	}

	void resize(size_t size)
	{
		allocate(size);
		size_ = size;
	}

	void map(size_t size, void* data)
	{
		data_ = (T*)data;
		size_ = size;
		allocated_ = false;
	}

	void assign(size_t size, const void* h_data)
	{
		resize(size);
		upload((T*)h_data);
	}

	void upload(const T* h_data)
	{
		CUDA_CHECK(cudaMemcpy(data_, h_data, sizeof(T) * size_, cudaMemcpyHostToDevice));
	}

	void download(T* h_data) const
	{
		CUDA_CHECK(cudaMemcpy(h_data, data_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
	}

	void copyTo(T* rhs) const
	{
		CUDA_CHECK(cudaMemcpy(rhs, data_, sizeof(T) * size_, cudaMemcpyDeviceToDevice));
	}

	void fillZero()
	{
		CUDA_CHECK(cudaMemset(data_, 0, sizeof(T) * size_));
	}

	T* data() { return data_; }
	const T* data() const { return data_; }

	size_t size() const { return size_; }
	int ssize() const { return static_cast<int>(size_); }

	operator T* () { return data_; }
	operator const T* () const { return data_; }

private:

	T* data_;
	size_t size_, capacity_;
	bool allocated_;
};

} // namespace cuba

#endif // !__DEVICE_BUFFER_H__
