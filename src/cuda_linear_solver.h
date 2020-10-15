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

#ifndef __CUDA_LINEAR_SOLVER_H__
#define __CUDA_LINEAR_SOLVER_H__

#include <memory>

#include "scalar.h"
#include "sparse_block_matrix.h"

namespace cuba
{

class SparseLinearSolver
{
public:

	using Ptr = std::unique_ptr<SparseLinearSolver>;
	static Ptr create();

	virtual void initialize(const HschurSparseBlockMatrix& Hsc) = 0;
	virtual bool solve(const Scalar* d_A, const Scalar* d_b, Scalar* d_x) = 0;

	virtual ~SparseLinearSolver();
};

} // namespace cuba

#endif // !__CUDA_LINEAR_SOLVER_H__
