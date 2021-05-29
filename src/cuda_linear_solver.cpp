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

#include "cuda_linear_solver.h"

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "device_buffer.h"
#include "cuda_block_solver.h"

namespace cuba
{

template <typename T>
static constexpr bool is_value_type_32f() { return std::is_same_v<T, float>; }
template <typename T>
static constexpr bool is_value_type_64f() { return std::is_same_v<T, double>; }

struct CusparseHandle
{
	CusparseHandle() { init(); }
	~CusparseHandle() { destroy(); }
	void init() { cusparseCreate(&handle); }
	void destroy() { cusparseDestroy(handle); }
	operator cusparseHandle_t() const { return handle; }
	CusparseHandle(const CusparseHandle&) = delete;
	CusparseHandle& operator=(const CusparseHandle&) = delete;
	cusparseHandle_t handle;
};

struct CusolverHandle
{
	CusolverHandle() { init(); }
	~CusolverHandle() { destroy(); }
	void init() { cusolverSpCreate(&handle); }
	void destroy() { cusolverSpDestroy(handle); }
	operator cusolverSpHandle_t() const { return handle; }
	CusolverHandle(const CusolverHandle&) = delete;
	CusolverHandle& operator=(const CusolverHandle&) = delete;
	cusolverSpHandle_t handle;
};

struct CusparseMatDescriptor
{
	CusparseMatDescriptor() { init(); }
	~CusparseMatDescriptor() { destroy(); }

	void init()
	{
		cusparseCreateMatDescr(&desc);
		cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);
	}

	void destroy() { cusparseDestroyMatDescr(desc); }
	operator cusparseMatDescr_t() const { return desc; }
	CusparseMatDescriptor(const CusparseMatDescriptor&) = delete;
	CusparseMatDescriptor& operator=(const CusparseMatDescriptor&) = delete;
	cusparseMatDescr_t desc;
};

template <typename T>
class SparseSquareMatrixCSR
{
public:

	SparseSquareMatrixCSR() : size_(0), nnz_(0) {}

	void resize(int size)
	{
		size_ = size;
		rowPtr_.resize(size + 1);
	}

	void resizeNonZeros(int nnz)
	{
		nnz_ = nnz;
		values_.resize(nnz);
		colInd_.resize(nnz);
	}

	void upload(const T* values = nullptr, const int* rowPtr = nullptr, const int* colInd = nullptr)
	{
		if (values)
			values_.upload(values);
		if (rowPtr)
			rowPtr_.upload(rowPtr);
		if (colInd)
			colInd_.upload(colInd);
	}

	void download(T* values = nullptr, int* rowPtr = nullptr, int* colInd = nullptr) const
	{
		if (values)
			values_.download(values);
		if (rowPtr)
			rowPtr_.download(rowPtr);
		if (colInd)
			colInd_.download(colInd);
	}

	T* val() { return values_.data(); }
	int* rowPtr() { return rowPtr_.data(); }
	int* colInd() { return colInd_.data(); }

	const T* val() const { return values_.data(); }
	const int* rowPtr() const { return rowPtr_.data(); }
	const int* colInd() const { return colInd_.data(); }

	int size() const { return size_; }
	int nnz() const { return nnz_; }

	cusparseMatDescr_t desc() const { return desc_; }

private:

	DeviceBuffer<T> values_;
	DeviceBuffer<int> rowPtr_;
	DeviceBuffer<int> colInd_;
	int size_, nnz_;
	CusparseMatDescriptor desc_;
};

template <typename T>
class SparseCholesky
{
public:

	void init(cusolverSpHandle_t handle)
	{
		handle_ = handle;

		// create info
		cusolverSpCreateCsrcholInfo(&info_);
	}

	void allocateBuffer(const SparseSquareMatrixCSR<T>& A)
	{
		size_t internalData, workSpace;

		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholBufferInfo(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, &internalData, &workSpace);

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholBufferInfo(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, &internalData, &workSpace);

		buffer_.resize(workSpace);
	}

	bool hasZeroPivot(int* position = nullptr) const
	{
		const T tol = static_cast<T>(1e-14);
		int singularity = -1;

		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholZeroPivot(handle_, info_, tol, &singularity);

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholZeroPivot(handle_, info_, tol, &singularity);

		if (position)
			*position = singularity;
		return singularity >= 0;
	}

	bool analyze(const SparseSquareMatrixCSR<T>& A)
	{
		cusolverSpXcsrcholAnalysis(handle_, A.size(), A.nnz(), A.desc(), A.rowPtr(), A.colInd(), info_);
		allocateBuffer(A);
		return true;
	}

	bool factorize(SparseSquareMatrixCSR<T>& A)
	{
		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholFactor(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, buffer_.data());

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholFactor(handle_, A.size(), A.nnz(), A.desc(),
				A.val(), A.rowPtr(), A.colInd(), info_, buffer_.data());

		return !hasZeroPivot();
	}

	void solve(int size, const T* b, T* x)
	{
		if constexpr (is_value_type_32f<T>())
			cusolverSpScsrcholSolve(handle_, size, b, x, info_, buffer_.data());

		if constexpr (is_value_type_64f<T>())
			cusolverSpDcsrcholSolve(handle_, size, b, x, info_, buffer_.data());
	}

	void destroy()
	{
		cusolverSpDestroyCsrcholInfo(info_);
	}

	~SparseCholesky() { destroy(); }

private:

	cusolverSpHandle_t handle_;
	csrcholInfo_t info_;
	DeviceBuffer<unsigned char> buffer_;
};

template <typename T>
class CuSparseCholeskySolver
{
public:

	enum Info
	{
		SUCCESS,
		NUMERICAL_ISSUE
	};

	CuSparseCholeskySolver(int size = 0)
	{
		init();

		if (size > 0)
			resize(size);
	}

	void init()
	{
		cholesky.init(cusolver);
		doOrdering = false;
		information = Info::SUCCESS;
	}

	void resize(int size)
	{
		Acsr.resize(size);
		d_y.resize(size);
		d_z.resize(size);
	}

	void setPermutaion(int size, const int* P)
	{
		h_PT.resize(size);
		for (int i = 0; i < size; i++)
			h_PT[P[i]] = i;

		d_P.assign(size, P);
		d_PT.assign(size, h_PT.data());
		doOrdering = true;
	}

	void analyze(int nnz, const int* csrRowPtr, const int* csrColInd)
	{
		const int size = Acsr.size();
		Acsr.resizeNonZeros(nnz);

		if (doOrdering)
		{
			d_tmpRowPtr.assign(size + 1, csrRowPtr);
			d_tmpColInd.assign(nnz, csrColInd);
			d_nnzPerRow.resize(size + 1);
			d_map.resize(nnz);

			gpu::twistCSR(size, nnz, d_tmpRowPtr, d_tmpColInd, d_PT,
				Acsr.rowPtr(), Acsr.colInd(), d_map, d_nnzPerRow);
		}
		else
		{
			Acsr.upload(nullptr, csrRowPtr, csrColInd);
		}

		cholesky.analyze(Acsr);
	}

	void factorize(const T* d_A)
	{
		if (doOrdering)
		{
			permute(Acsr.nnz(), d_A, Acsr.val(), d_map);
		}
		else
		{
			cudaMemcpy(Acsr.val(), d_A, sizeof(Scalar) * Acsr.nnz(), cudaMemcpyDeviceToDevice);
		}

		// M = L * LT
		if (!cholesky.factorize(Acsr))
			information = Info::NUMERICAL_ISSUE;
	}

	void solve(const T* d_b, T* d_x)
	{
		if (doOrdering)
		{
			// y = P * b
			permute(Acsr.size(), d_b, d_y, d_P);

			// solve A * z = y
			cholesky.solve(Acsr.size(), d_y, d_z);

			// x = PT * z
			permute(Acsr.size(), d_z, d_x, d_PT);
		}
		else
		{
			// solve A * x = b
			cholesky.solve(Acsr.size(), d_b, d_x);
		}
	}

	void permute(int size, const T* src, T* dst, const int* P)
	{
		gpu::permute(size, src, dst, P);
	}

	void reordering(int size, int nnz, const int* csrRowPtr, const int* csrColInd, int* P) const
	{
		//cusolverSpXcsrsymrcmHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
		//cusolverSpXcsrsymamdHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
		//cusolverSpXcsrsymmdqHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, P);
		cusolverSpXcsrmetisndHost(cusolver, size, nnz, Acsr.desc(), csrRowPtr, csrColInd, nullptr, P);
	}

	Info info() const
	{
		return information;
	}

	void downloadCSR(int* csrRowPtr, int* csrColInd)
	{
		Acsr.download(nullptr, csrRowPtr, csrColInd);
	}

private:

	SparseSquareMatrixCSR<T> Acsr;
	DeviceBuffer<T> d_y, d_z, d_tmp;
	DeviceBuffer<int> d_P, d_PT, d_map;
	DeviceBuffer<int> d_tmpRowPtr, d_tmpColInd, d_nnzPerRow;

	CusparseHandle cusparse;
	CusolverHandle cusolver;

	SparseCholesky<T> cholesky;

	std::vector<int> h_PT;

	Info information;
	bool doOrdering;
};

class SparseLinearSolverImpl : public SparseLinearSolver
{
public:

	using SparseMatrixCSR = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;
	using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;
	using Cholesky = CuSparseCholeskySolver<Scalar>;

	void initialize(const HschurSparseBlockMatrix& Hsc) override
	{
		const int size = Hsc.rows();
		const int nnz = Hsc.nnzSymm();

		cholesky_.resize(size);

		// set permutation
		P_.resize(size);
		cholesky_.reordering(size, nnz, Hsc.rowPtr(), Hsc.colInd(), P_.data());
		cholesky_.setPermutaion(size, P_.data());

		// analyze
		cholesky_.analyze(nnz, Hsc.rowPtr(), Hsc.colInd());
	}

	bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) override
	{
		cholesky_.factorize(d_A);

		if (cholesky_.info() != Cholesky::SUCCESS)
		{
			std::cerr << "factorize failed" << std::endl;
			return false;
		}

		cholesky_.solve(d_b, d_x);

		return true;
	}

private:

	std::vector<int> P_;
	Cholesky cholesky_;
};

SparseLinearSolver::Ptr SparseLinearSolver::create()
{
	return std::make_unique<SparseLinearSolverImpl>();
}

SparseLinearSolver::~SparseLinearSolver()
{
}

} // namespace cuba
