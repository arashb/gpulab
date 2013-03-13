/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: gradient
* file:    gradient.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef GRADIENT_CUH
#define GRADIENT_CUH


#include <stdio.h>
#include <cuda_runtime.h>
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



const char* getStudentLogin();
const char* getStudentName();
int         getStudentID();
bool        checkStudentData();
bool        checkStudentNameAndID();



void gpu_derivative_sm_d(const float *inputImage, float *outputImage,
                          int iWidth, int iHeight, int iSpectrum, int mode=0);



#endif // #ifndef GRADIENT_CUH
