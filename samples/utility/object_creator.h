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

#ifndef __OBJECT_CREATOR_H__
#define __OBJECT_CREATOR_H__

#include <vector>

namespace cuba
{

struct BaseObject { virtual ~BaseObject() {} };

template <class T>
struct Object : public BaseObject
{
	Object(T* ptr = nullptr) : ptr(ptr) {}
	~Object() { delete ptr; }
	T* ptr;
};

class ObjectCreator
{
public:

	ObjectCreator() {}
	~ObjectCreator() { release(); }

	template <class T, class... Args>
	T* create(Args&&... args)
	{
		T* ptr = new T(std::forward<Args>(args)...);
		objects_.push_back(new Object(ptr));
		return ptr;
	}

	void release()
	{
		for (auto ptr : objects_)
			delete ptr;
		objects_.clear();
	}

private:

	std::vector<BaseObject*> objects_;
};

} // namespace cuba

#endif // !__OBJECT_CREATOR_H__
