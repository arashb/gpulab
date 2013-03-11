/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: convolution
* file:    convolution_gpu.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef __CONVOLUTION_GPU_CUH
#define __CONVOLUTION_GPU_CUH


#include "convolution_cpu.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <time.h>

#ifndef cutilSafeCall
#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
  if(cudaSuccess != err) {
    fprintf(stderr,"%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}
#endif

const char* gpu_getStudentLogin();
const char* gpu_getStudentName();
int         gpu_getStudentID();
bool        gpu_checkStudentData();
bool        gpu_checkStudentNameAndID();


__global__ void gpu_convolutionGrayImage_gm_d(const float *inputImage, const float *kernel, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch, size_t kPitch);
                                              
__global__ void gpu_convolutionGrayImage_gm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch);
                                             
__global__ void gpu_convolutionGrayImage_sm_d(const float *inputImage, const float *kernel, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch, size_t kPitch);

__global__ void gpu_convolutionGrayImage_sm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch);
                                              
__global__ void gpu_convolutionGrayImage_dsm_cm_d(const float *inputImage, float *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitch);
                                             
__global__ void gpu_convolutionGrayImage_tex_cm_d(float *outputImage,
    int iWidth, int iHeight, int kRadiusX, int kRadiusY,
    size_t iPitch);

void gpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode=1);

void gpu_convolutionRGB(const float *inputImage, const float *kernel, float *outputImage,
                        int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode=1);


__global__ void gpu_convolutionInterleavedRGB_dsm_cm_d(const float3 *inputImage, float3 *outputImage,
                                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                              size_t iPitchBytes);

void gpu_convolutionInterleavedRGB(const float *inputImage, const float *kernel, float *outputImage,
                                   int iWidth, int iHeight, int kRadiusX, int kRadiusY, int mode=5);

__global__ void gpu_ImageFloat3ToFloat4_d(const float3 *inputImage, float4 *outputImage, int iWidth, int iHeight, int iPitchBytes, size_t oPitchBytes);

__global__ void gpu_convolutionInterleavedRGB_tex_cm_d(float3 *outputImage,
    int iWidth, int iHeight, int kRadiusX, int kRadiusY, size_t oPitchBytes);


void gpu_convolutionKernelBenchmarkGrayImage(const float *inputImage, const float *kernel, float *outputImage,
                                             int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                             int numKernelTestCalls);

void gpu_convolutionKernelBenchmarkInterleavedRGB(const float *inputImage, const float *kernel, float *outputImage,
                                                  int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                                                  int numKernelTestCalls);


void gpu_bindConstantMemory(const float *kernel, int size);

void gpu_bindTextureMemory(float *d_inputImage, int iWidth, int iHeight, size_t iPitchBytes);

void gpu_unbindTextureMemory();

void gpu_bindTextureMemoryF4(float4 *d_inputImageF4, int iWidth, int iHeight, size_t iPitchBytesF4);

void gpu_unbindTextureMemoryF4();



#endif // #ifndef __CONVOLUTION_GPU_CUH
