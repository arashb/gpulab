/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: convolution
* file:    main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "convolution_cpu.h"
#include "convolution_gpu.cuh"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main(int argc,char **argv) {
  //----------------------------------------------------------------------------
  // Initialization
  //----------------------------------------------------------------------------
  if (argc <= 1) {
    cout << "Too few arguments!\n" << endl;
    cout << "Usage: " << argv[0] << " [ImageName] [Mode=0] [Radius=5] [Sigma=10] [Interleaved=0] [NumBenchmarkCycles=0]" << endl;
    cout << "   or: " << argv[0] << " [camera] [Mode=5] [Radius=5] [Sigma=10] [Width=640] [Height=480]" <<endl;
    cout << endl << "Available modes:" << endl;
    cout << "0 - cpu only" << endl;
    cout << "1 - global memory only" << endl;
    cout << "2 - global memory for image & constant memory for kernel access" << endl;
    cout << "3 - shared memory for image & global memory for kernel access" << endl;
    cout << "4 - shared memory for image & constant memory for kernel access" << endl;
    cout << "5 - dyn. shared memory for image & const memory for kernel access" << endl;
    cout << "6 - texture memory for image & const memory for kernel access" << endl;
    cout << "interleaved=1 - for modes 5 and 6 only" << endl;
    return 1;
  }
  // check for student name and ID
  if (!cpu_checkStudentData()) std::cout << "WARNING: Please enter your correct student name and ID in file convolution_cpu.cpp!" << std::endl;
  if (!gpu_checkStudentData()) std::cout << "WARNING: Please enter your correct student name and ID in file convolution_gpu.cu!"  << std::endl;
  if (strcmp(cpu_getStudentName(), gpu_getStudentName()) != 0) std::cout << "WARNING: Student name mismatch in files convolution_cpu.cpp and convolution_gpu.cu" << std::endl;
  if (cpu_getStudentID() != gpu_getStudentID()) std::cout << "WARNING: Student ID mismatch in files convolution_cpu.cpp and convolution_gpu.cu" << std::endl;

  // read program arguments
  const int kRadiusX = (argc > 3) ? atoi(argv[3]) : 5;
  const int kRadiusY = kRadiusX;
  const float sigma = (argc > 4) ? (float)atof(argv[4]) : 10.0f;
  int mode, numBenchmarkCycles, cameraWidth, cameraHeight;
  bool interleavedMode;  
  if (strcmp(argv[1],"camera") != 0) {
    mode = (argc > 2) ? atoi(argv[2]) : 0;
    interleavedMode = ((argc > 5) ? atoi(argv[5]) != 0 : false);
    numBenchmarkCycles = (argc > 6) ? atoi(argv[6]) : 0;
  }
  else{
    mode = (argc > 2) ? atoi(argv[2]) : 5;
    if (mode != 5 && mode != 6) {
      std::cout << "camera - for modes 5 or 6 only" << std::endl;
      return 1;
    }
    interleavedMode = true;
    cameraWidth = (argc > 5) ? atoi(argv[5]) : 640;
    cameraHeight = (argc > 6) ? atoi(argv[6]) : 480;
  }

  // Compute Gaussian Kernel in CPU method  
  float *kernel = makeGaussianKernel(kRadiusX, kRadiusY, sigma, sigma);

  //----------------------------------------------------------------------------
  // Image File Mode
  //----------------------------------------------------------------------------
  if (strcmp(argv[1],"camera") != 0) {


  	cv::Mat inputImg = cv::imread(argv[1], -1);
  	if (inputImg.data == NULL) { fprintf(stderr,"\nERROR: Could not load image %s",argv[1]); return 1; }

  	inputImg.convertTo(inputImg,(inputImg.type() == CV_8UC1) ? CV_32FC1 : CV_32FC3);

    fprintf(stderr,"\nNumber of Image Channels: %i\n",inputImg.channels());

    cv::Mat outputImg(inputImg.rows,inputImg.cols,inputImg.type());

    //---------- gray values images ----------
    if (inputImg.channels() == 1) {

      if (mode == 0){
        cpu_convolutionGrayImage((float*)inputImg.data, kernel, (float*)outputImg.data,
        		inputImg.cols, inputImg.rows, kRadiusX, kRadiusY);
      }
      else{
        gpu_convolutionGrayImage((float*)inputImg.data, kernel, (float*)outputImg.data,
        		inputImg.cols, inputImg.rows, kRadiusX, kRadiusY, mode);
      }

  	} // endif gray image
    //---------- RGB color images ----------
    else if (inputImg.channels() == 3) {

      if (interleavedMode) {
      	if(mode <= 0){
      		fprintf(stderr,"\nInterleaved Color Mode only available for GPU!");
      		return -1;
      	}
        gpu_convolutionInterleavedRGB((float*)inputImg.data, kernel, (float*)outputImg.data,
        		inputImg.cols, inputImg.rows, kRadiusX, kRadiusY, mode);
      } // endif interleaved mode
      else {
        float *imgData =       new float[inputImg.cols*inputImg.rows*inputImg.channels()];
        float *outputImgData = new float[inputImg.cols*inputImg.rows*inputImg.channels()];

        for (int i=0;i<inputImg.rows;i++){
          for (int j=0;j<inputImg.cols;j++){
            for (int s=0;s<inputImg.channels();s++){
              imgData[s*inputImg.cols*inputImg.rows+i*inputImg.cols+j] =
              		((float*)inputImg.data)[(i*inputImg.cols+j)*inputImg.channels()+s];
            }
          }
        }

        if (mode == 0){
          cpu_convolutionRGB(imgData, kernel, outputImgData, inputImg.cols, inputImg.rows, kRadiusX, kRadiusY);
        }
        else{
          gpu_convolutionRGB(imgData, kernel, outputImgData, inputImg.cols, inputImg.rows, kRadiusX, kRadiusY, mode);
        }

        for (int i=0;i<inputImg.rows;i++){
          for (int j=0;j<inputImg.cols;j++){
            for (int s=0;s<inputImg.channels();s++){
            	((float*)outputImg.data)[(i*inputImg.cols+j)*inputImg.channels()+s] =
              		outputImgData[s*inputImg.cols*inputImg.rows+i*inputImg.cols+j];
            }
          }
        }

        delete[] imgData;
        delete[] outputImgData;

      } // endif non-interleaved mode

    } // endif RGB image (spectrum == 3)
    else {
      std::cout << "Error: Unsupported image type!" << std::endl;
      return 1;
    }

    //TODO: BENCHMARK NICHT VERGESSEN, UND DANN Kamera

    cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Original Image", 100, 100);
    cv::imshow("Original Image",inputImg/255.0f);
    cv::namedWindow("Output Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Output Image", 500, 100);
    cv::imshow("Output Image",outputImg/255.0f);

    cv::Mat kernelImg(kRadiusY*2+1,kRadiusX*2+1,CV_32FC1);
    cpu_normalizeGaussianKernel_cv(kernel,(float*)kernelImg.data,kRadiusX,kRadiusY,kernelImg.cols);
    cv::namedWindow("Kernel Image",CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Kernel Image", 150, 150);cv::imshow("Kernel Image",kernelImg);

    //---------- BENCHMARK ----------
    if (numBenchmarkCycles > 0 && inputImg.channels() == 3){
    	fprintf(stderr,"\nRunning Benchmark ...\n");

      cpu_convolutionBenchmark((float*)inputImg.data, kernel, (float*)outputImg.data, inputImg.cols, inputImg.rows, kRadiusX, kRadiusY, numBenchmarkCycles);

      gpu_convolutionKernelBenchmarkGrayImage((float*)inputImg.data, kernel, (float*)outputImg.data, inputImg.cols, inputImg.rows, kRadiusX, kRadiusY, numBenchmarkCycles);

      gpu_convolutionKernelBenchmarkInterleavedRGB((float*)inputImg.data, kernel, (float*)outputImg.data, inputImg.cols, inputImg.rows, kRadiusX, kRadiusY, numBenchmarkCycles);

      fprintf(stderr,"\nBenchmark done.");
    }// end benchmark

    fprintf(stderr,"\nPress any key on the image to exit...\n\n");
    cv::waitKey(0);
  } // endif image file



  //----------------------------------------------------------------------------
  // Camera Mode
  //----------------------------------------------------------------------------
  else {

  	fprintf(stderr,"\nReading Images from Camera");
  	cv::VideoCapture capture(0);
  	if(!capture.isOpened()){
  		std::cerr << std::endl << "Could not Open Capture Device";
  		return -1;
  	}
  	capture.set(CV_CAP_PROP_FRAME_WIDTH,cameraWidth);
  	capture.set(CV_CAP_PROP_FRAME_HEIGHT,cameraHeight);
  	cv::Mat inputImg(cv::Size(cameraWidth,cameraHeight),CV_8UC3);
  	cv::namedWindow("Input",CV_WINDOW_AUTOSIZE);
  	cv::namedWindow("Output",CV_WINDOW_AUTOSIZE);
		cvMoveWindow("Input", 50, 100);
		cvMoveWindow("Output", 750, 100);

//    img = cvQueryFrame(capture);
  	capture >> inputImg;

  	cv::Mat outputImg(inputImg.rows,inputImg.cols,CV_32FC3);


    fprintf(stderr,"\nPress Esc on the image to exit...\n");

    while (cv::waitKey(30) < 0){
    	capture >> inputImg;
    	inputImg.convertTo(inputImg,CV_32FC3);

      gpu_convolutionInterleavedRGB((float*)inputImg.data, kernel,
      		(float*)outputImg.data, inputImg.cols, inputImg.rows, kRadiusX, kRadiusY, mode);

      cv::imshow("Input", inputImg/255.0f);
      cv::imshow("Output", outputImg/255.0f);
    }

    cvDestroyAllWindows();
  } //endif "camera"

  return 0;
}
