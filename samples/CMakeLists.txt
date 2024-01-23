cmake_minimum_required(VERSION 3.18)

project(samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(SRCS_COMMON object_creator.h)

add_executable(sample_ba_from_file sample_ba_from_file.cpp ${SRCS_COMMON})
target_include_directories(sample_ba_from_file PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(sample_ba_from_file cuda_bundle_adjustment ${OpenCV_LIBS})
target_compile_options(sample_ba_from_file PRIVATE
$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
)

if (WITH_G2O)
	find_package(g2o REQUIRED)
	set(G2O_LIBS ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_EIGEN} ${G2O_TYPES_SBA} ${G2O_TYPES_SLAM3D})
	
	add_executable(sample_comparison_with_g2o sample_comparison_with_g2o.cpp ${SRCS_COMMON})
	target_include_directories(sample_comparison_with_g2o PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIR}/g2o/EXTERNAL/csparse)
	target_link_libraries(sample_comparison_with_g2o cuda_bundle_adjustment ${OpenCV_LIBS} ${G2O_LIBS})
	target_compile_options(sample_comparison_with_g2o PRIVATE
	$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -O3>
	$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819 /wd4244 /wd4251 /wd4267>
	)
	target_compile_definitions(sample_comparison_with_g2o PRIVATE G2O_USE_VENDORED_CERES)
endif()
