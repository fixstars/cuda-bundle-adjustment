cmake_minimum_required(VERSION 3.7)

project(CudaBundleAdjustment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_SAMPLES "Build samples" ON)
option(WITH_G2O       "Build sample with g2o (g2o needs to be installed)" OFF)
option(USE_FLOAT32    "Use 32bit float in internal floating-point operations (default is 64bit float)" OFF)
set(CUDA_ARCHS "61;72;75;86" CACHE STRING "List of architectures to generate device code for")

add_subdirectory(src)

if(ENABLE_SAMPLES)
	add_subdirectory(samples/sample_ba_from_file)
	if(WITH_G2O)
		add_subdirectory(samples/sample_comparison_with_g2o)
	endif()
endif()
