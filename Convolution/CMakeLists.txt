cmake_minimum_required(VERSION 2.4.6)
FIND_PACKAGE(CUDA)
FIND_PACKAGE(OpenCV REQUIRED)
#FIND_PACKAGE(OpenMP)

SET(PROJECT_NAME convolution)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CUDA_ARCH_FLAGS 
#-DGT400 -arch sm_20
)

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("$ENV{CUDADIR}/include")
include_directories("$ENV{CUDASDKDIR}/common/inc")
link_directories("$ENV{CUDASDKDIR}/lib")
link_directories("$ENV{CUDASDKDIR}/../shared/lib")
link_directories("$ENV{CUDADRVDIR}")


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

add_executable(${PROJECT_NAME}
src/convolution_main.cpp
src/convolution_cpu.cpp
)

SET(CUDA_SOURCES
src/convolution_gpu.cu
)

CUDA_ADD_LIBRARY(cudaobjects ${CUDA_SOURCES} OPTIONS --ptxas-options=-v -DUNIX -D_DEBUG 
${CUDA_ARCH_FLAGS}
)

target_link_libraries(${PROJECT_NAME} 
${OpenCV_LIBS}
cudaobjects cuda cudart
)