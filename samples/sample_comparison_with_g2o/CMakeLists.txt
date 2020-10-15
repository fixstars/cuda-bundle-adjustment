cmake_minimum_required(VERSION 3.7)

project(sample_comparison_with_g2o)

set(SAMPLE_UTILITY_DIR ../utility)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

if (WIN32)
	add_compile_options(/wd4819)
	add_definitions(-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING)

	# g2o warning suppressions
	add_compile_options(/wd4244 /wd4251 /wd4267)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${SAMPLE_UTILITY_DIR})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# g2o
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIR}/g2o/EXTERNAL/csparse)

file(GLOB srcs ./*.cpp ./*.h* ${SAMPLE_UTILITY_DIR}/*.cpp ${SAMPLE_UTILITY_DIR}/*.h*)
add_executable(sample_comparison_with_g2o ${srcs})
set(G2O_LIBS ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_EIGEN} ${G2O_TYPES_SBA} ${G2O_TYPES_SLAM3D})
target_link_libraries(sample_comparison_with_g2o ${OpenCV_LIBS} ${G2O_LIBS} cuda_bundle_adjustment)
