/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: convolution
* file:    convolution_cpu.h
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#ifndef __CONVOLUTION_CPU_H
#define __CONVOLUTION_CPU_H


const char* cpu_getStudentLogin();
const char* cpu_getStudentName();
int         cpu_getStudentID();
bool        cpu_checkStudentData();
bool        cpu_checkStudentNameAndID();


float *makeGaussianKernel(int radiusX, int radiusY, float sigmaX, float sigmaY);

float *cpu_normalizeGaussianKernel_cv(const float *kernel, float *kernelImgdata, int kRadiusX, int kRadiusY, int step);

void cpu_convolutionGrayImage(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY);

void cpu_convolutionRGB(const float *inputImage, const float *kernel, float *outputImage,
                        int iWidth, int iHeight, int kRadiusX, int kRadiusY);

void cpu_convolutionBenchmark(const float *inputImage, const float *kernel, float *outputImage,
                              int iWidth, int iHeight, int kRadiusX, int kRadiusY,
                              int numKernelTestCalls);

#endif // #ifndef __CONVOLUTION_CPU_H
