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

#include "sparse_block_matrix.h"

#include <algorithm>

namespace cuba
{

void HplSparseBlockMatrix::constructFromBlockPos(std::vector<HplBlockPos>& blockpos)
{
	Eigen::VectorXi& bcolPtr = outerIndices_;
	Eigen::VectorXi& browInd = innerIndices_;
	Eigen::VectorXi nnzPerCol;

	nnzPerCol.resize(bcols_);
	nnzPerCol.setZero();

	std::sort(std::begin(blockpos), std::end(blockpos), [](const HplBlockPos& lhs, const HplBlockPos& rhs)
	{
		return lhs.row < rhs.row;
	});

	for (const auto& pos : blockpos)
		nnzPerCol[pos.col]++;

	// set colPtr
	bcolPtr.resize(bcols_ + 1);
	bcolPtr[0] = 0;
	for (int c = 0; c < bcols_; c++)
		bcolPtr[c + 1] = bcolPtr[c] + nnzPerCol[c];
	nblocks_ = bcolPtr[bcols_];

	// set rowInd
	nnzPerCol = bcolPtr;
	browInd.resize(nblocks_);
	for (const auto& pos : blockpos)
		browInd[nnzPerCol[pos.col]++] = pos.row;
}

void HschurSparseBlockMatrix::constructFromVertices(const std::vector<VertexL*>& verticesL)
{
	struct BlockPos { int row, col; };

	Eigen::VectorXi& browPtr_ = outerIndices_;
	Eigen::VectorXi& bcolInd_ = innerIndices_;

	std::vector<uint8_t> map(brows_ * bcols_, 0);
	std::vector<int> indices;

	std::vector<BlockPos> blockpos;
	blockpos.reserve(brows_ * bcols_);

	int countmul = 0;
	for (auto vL : verticesL)
	{
		if (vL->fixed)
			continue;

		indices.clear();
		for (const auto e : vL->edges)
		{
			const auto vP = e->poseVertex();
			if (!vP->fixed)
				indices.push_back(vP->iP);
		}

		std::sort(std::begin(indices), std::end(indices));
		const int nindices = static_cast<int>(indices.size());
		for (int i = 0; i < nindices; i++)
		{
			const int rowId = indices[i];
			uint8_t* ptrMap = map.data() + rowId * bcols_;
			for (int j = i; j < nindices; j++)
			{
				const int colId = indices[j];
				if (!ptrMap[colId])
				{
					blockpos.push_back({ rowId, colId });
					ptrMap[colId] = 1;
				}

				countmul++;
			}
		}
	}

	nmultiplies_ = countmul;

	// set nonzero blocks
	nblocks_ = static_cast<int>(blockpos.size());

	std::sort(std::begin(blockpos), std::end(blockpos), [](const BlockPos& lhs, const BlockPos& rhs)
	{
		return lhs.col < rhs.col;
	});

	// set rowPtr
	nnzPerRow_.resize(brows_);
	nnzPerRow_.setZero();
	for (int i = 0; i < nblocks_; i++)
		nnzPerRow_[blockpos[i].row]++;

	browPtr_.resize(brows_ + 1);
	browPtr_[0] = 0;
	for (int r = 0; r < brows_; r++)
		browPtr_[r + 1] = browPtr_[r] + nnzPerRow_[r];

	// set colInd
	nnzPerRow_ = browPtr_;
	bcolInd_.resize(nblocks_);
	for (int i = 0; i < nblocks_; i++)
	{
		const int rowId = blockpos[i].row;
		const int colId = blockpos[i].col;
		const int k = nnzPerRow_[rowId]++;
		bcolInd_[k] = colId;
	}
}

void HschurSparseBlockMatrix::convertBSRToCSR()
{
	const int PDIM = BLOCK_ROWS;
	const int nnz = nnzSymm();
	const int drows = rows();

	Eigen::VectorXi& browPtr_ = outerIndices_;
	Eigen::VectorXi& bcolInd_ = innerIndices_;

	rowPtr_.resize(drows + 1);
	colInd_.resize(nnz);
	BSR2CSR_.resize(nnz);

	nnzPerRow_.resize(drows);
	nnzPerRow_.setZero();

	for (int blockRowId = 0; blockRowId < brows_; blockRowId++)
	{
		for (int i = browPtr_[blockRowId]; i < browPtr_[blockRowId + 1]; i++)
		{
			const int blockColId = bcolInd_[i];
			const int dstColId0 = blockColId * PDIM;
			const int dstRowId0 = blockRowId * PDIM;
			if (blockRowId == blockColId)
			{
				for (int dr = 0; dr < PDIM; dr++)
					nnzPerRow_[dstRowId0 + dr] += PDIM;
			}
			else
			{
				for (int dr = 0; dr < PDIM; dr++)
					nnzPerRow_[dstRowId0 + dr] += PDIM;
				for (int dc = 0; dc < PDIM; dc++)
					nnzPerRow_[dstColId0 + dc] += PDIM;
			}
		}
	}

	rowPtr_[0] = 0;
	for (int r = 0; r < drows; r++)
		rowPtr_[r + 1] = rowPtr_[r] + nnzPerRow_[r];

	for (int r = 0; r < drows; r++)
		nnzPerRow_[r] = rowPtr_[r];

	for (int blockRowId = 0; blockRowId < brows_; blockRowId++)
	{
		for (int i = browPtr_[blockRowId]; i < browPtr_[blockRowId + 1]; i++)
		{
			const int blockColId = bcolInd_[i];
			const int dstColId0 = blockColId * PDIM;
			const int dstRowId0 = blockRowId * PDIM;
			int srck = i * PDIM * PDIM;

			if (blockRowId == blockColId)
			{
				for (int dc = 0; dc < PDIM; ++dc)
				{
					for (int dr = 0; dr < PDIM; ++dr)
					{
						const int colId = dstColId0 + dc;
						const int rowId = dstRowId0 + dr;
						const int dstk = nnzPerRow_[rowId]++;
						colInd_[dstk] = colId;
						BSR2CSR_[dstk] = srck;
						srck++;
					}
				}
			}
			else
			{
				for (int dc = 0; dc < PDIM; ++dc)
				{
					for (int dr = 0; dr < PDIM; ++dr)
					{
						const int colId0 = dstColId0 + dc;
						const int rowId0 = dstRowId0 + dr;

						const int colId1 = rowId0;
						const int rowId1 = colId0;

						const int dstk0 = nnzPerRow_[rowId0]++;
						const int dstk1 = nnzPerRow_[rowId1]++;

						colInd_[dstk0] = colId0;
						colInd_[dstk1] = colId1;
						BSR2CSR_[dstk0] = srck;
						BSR2CSR_[dstk1] = srck;
						srck++;
					}
				}
			}
		}
	}
}

} // namespace cuba
