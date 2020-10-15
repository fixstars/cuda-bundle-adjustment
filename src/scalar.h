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

#ifndef __SCALAR_H__
#define __SCALAR_H__

#include <type_traits>

namespace cuba
{

#ifdef USE_FLOAT32
using Scalar = float;
#else
using Scalar = double;
#endif // USE_FLOAT32

static_assert(std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value,
	"Scalar must be float or double.");

} // namespace cuba

#endif // !__SCALAR_H__
