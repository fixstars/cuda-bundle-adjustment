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

#ifndef __ROBUST_KERNEL_H__
#define __ROBUST_KERNEL_H__

#include "scalar.h"

namespace cuba
{

struct RobustKernel
{
	RobustKernel(int type = 0, Scalar delta = 0) : type(type), delta(delta) {}
	int type;
	Scalar delta;
};

} // namespace cuba

#endif // !__ROBUST_KERNEL_H__
