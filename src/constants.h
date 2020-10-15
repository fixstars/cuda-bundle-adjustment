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

#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

namespace cuba
{

static constexpr int PDIM = 6;
static constexpr int LDIM = 3;

enum StorageOrder
{
	ROW_MAJOR,
	COL_MAJOR
};

enum EdgeFlag
{
	EDGE_FLAG_FIXED_L = 1,
	EDGE_FLAG_FIXED_P = 2
};

} // namespace cuba

#endif // !__CONSTANTS_H__
