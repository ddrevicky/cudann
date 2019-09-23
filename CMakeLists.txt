cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

if (WIN32)
    project (cudann LANGUAGES CXX)		# Listing CUDA results in error in CMake on Windows
elseif (UNIX)
    project (cudann LANGUAGES CXX CUDA)
endif ()

find_package(CUDA)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

include_directories(cudann 
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_TOOLKIT_INCLUDE})

# Collect source files
file(GLOB PROJECT_SOURCEFILES
    ${PROJECT_SOURCE_DIR}/src/*.h
    ${PROJECT_SOURCE_DIR}/src/*.hxx
    ${PROJECT_SOURCE_DIR}/src/*.cu
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.c
)

# Specify build type
set(CMAKE_BUILD_TYPE Release)

add_executable (${PROJECT_NAME} ${PROJECT_SOURCEFILES})
if (WIN32) 
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)              # Explicitly link with 64-bit version of the CUDA libs
    target_link_libraries(${PROJECT_NAME} cudart.lib cuda.lib)
elseif (UNIX)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif ()
target_compile_features(cudann PUBLIC cxx_std_11)