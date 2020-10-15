cmake_minimum_required(VERSION 3.7)

project(sample_ba_from_file)

set(SAMPLE_UTILITY_DIR ../utility)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

if (WIN32)
	add_compile_options(/wd4819)
	add_definitions(-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${SAMPLE_UTILITY_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB srcs ./*.cpp ./*.h* ${SAMPLE_UTILITY_DIR}/*.cpp ${SAMPLE_UTILITY_DIR}/*.h*)
add_executable(sample_ba_from_file ${srcs})
target_link_libraries(sample_ba_from_file ${OpenCV_LIBS} cuda_bundle_adjustment)
