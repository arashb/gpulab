/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: diffusion
* file:    diffusion.cuh
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef __DIFFUSION_CUH
#define __DIFFUSION_CUH

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


__global__ void diffuse_linear_isotrop_shared(const float *d_input,float *d_output,float timeStep, int nx,int ny, size_t pitch);

__global__ void diffuse_linear_isotrop_shared(const float3 *d_input,float3 *d_output,float timeStep,int nx, int ny, size_t pitchBytes);
 
__global__ void diffuse_nonlinear_isotrop_shared(const float *d_input,const float *d_diffusivity,float *d_output,float timeStep,int nx,int ny, size_t pitch);
 
__global__ void diffuse_nonlinear_isotrop_shared(const float3 *d_input,const float3 *d_diffusivity,float3 *d_output,float timeStep, int nx, int ny, size_t pitchBytes);

__global__ void compute_tv_diffusivity_shared(const float *d_input,float *d_output,int nx,int ny,size_t pitch);

__global__ void compute_tv_diffusivity_joined_shared(const float3 *d_input,float3 *d_output,int nx,int ny,size_t pitchBytes);

__global__ void compute_tv_diffusivity_separate_shared(const float3 *d_input,float3 *d_output,int nx,int ny,size_t pitchBytes);
 
__global__ void jacobi_shared(float *d_output,const float *d_input,const float *d_original,const float *d_diffusivity,float weight,int nx,int ny,size_t pitch);
 
__global__ void jacobi_shared(float3 *d_output,const float3 *d_input,const float3 *d_original,const float3 *d_diffusivity,float weight,int nx,int ny,size_t pitchBytes);
 
__global__ void sor_shared(float *d_output,const float *d_input,const float *d_original,const float *d_diffusivity,float weight,float overrelaxation,int nx,int ny,size_t pitch,int red);
 
__global__ void sor_shared(float3 *d_output,const float3 *d_input,const float3 *d_original,const float3 *d_diffusivity,float weight,float overrelaxation,int nx,int ny,size_t pitchBytes,int red);
 
void gpu_diffusion(const float *input,float *output,int nx, int ny, int nc,float timeStep,int iterations,float weight,int lagged_iterations,float overrelaxation,int mode,bool jointDiffusivity=false);


#endif // #ifndef __DIFFUSION_CUH
