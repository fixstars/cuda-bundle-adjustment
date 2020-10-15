cmake_minimum_required(VERSION 3.7)

project(cuda_bundle_adjustment)

set(CUBA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
include_directories(${CUBA_INCLUDE_DIR})

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo")

message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Flags for each platform
if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

if (WIN32)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd 4819\" -Xcompiler \"/wd 4244\"")
	add_compile_options(/wd4819)
	add_definitions(-D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING)
endif()

if (USE_FLOAT32)
	add_definitions(-DUSE_FLOAT32)
endif()

file(GLOB srcs ./*.cpp ./*.cu ./*.h ${CUBA_INCLUDE_DIR}/*.h)
cuda_add_library(cuda_bundle_adjustment ${srcs})
target_link_libraries(cuda_bundle_adjustment ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY})

install(
	TARGETS cuda_bundle_adjustment
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

install(
	DIRECTORY ${CUBA_INCLUDE_DIR}
	DESTINATION ${CMAKE_INSTALL_PREFIX}
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
