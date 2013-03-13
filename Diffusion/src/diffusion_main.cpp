/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: diffusion
* file:    main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "diffusion.cuh"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;

int main(int argc,char **argv) {
  //----------------------------------------------------------------------------
  // Initialization
  //----------------------------------------------------------------------------
  if (argc <= 1) {
    cout << "Too few arguments!\n" << endl;
    cout << "Usage: " << argv[0] << " [ImageName] [mode=0] [timeStepWidth=0.05] [Iterations=100] [regWeight=10] [l_iter=3] [overrelaxation=1] [jointDiffusivity=0]" << endl << endl;
    cout << "   or: " << argv[0] << " [camera] [mode=3] [W=320] [H=240] [timeStepWidth=0.05] [Iterations=50] [regWeight=10] [l_iter=3] [overrelaxation=1] [jointDiffusivity=0]" << endl;
    cout << endl << "Available modes:" << endl;
    cout << "0 - linear isotropic diffusion - gradient descent" << endl;
    cout << "1 - non-linear isotropic diffusion - explicit" << endl;
    cout << "2 - non-linear isotropic diffusion - jacobi" << endl;
    cout << "3 - non-linear isotropic diffusion - SOR" << endl;
    return 1;
  }
  // check for student data
  if (!checkStudentData()) std::cout << "WARNING: Please enter your correct student login, name and ID in header of diffusion.cu!" << std::endl;

  int mode, numIterations, lagged_iterations, cameraWidth, cameraHeight;
  float timeStepWidth, regWeight, overrelaxation;
  bool jointDiffusivity;
  if (strcmp(argv[1], "camera") != 0) {
    mode = (argc > 2) ? atoi(argv[2]) : 0;
    timeStepWidth = (argc > 3) ? (float)atof(argv[3]) : 0.05f;
    numIterations = (argc > 4) ? atoi(argv[4]) : 100;
    regWeight = (argc > 5) ? (float)atof(argv[5]) : 10.0f;
    lagged_iterations = (argc > 6) ? atoi(argv[6]) : 3;
    overrelaxation = (argc > 7) ? (float)atof(argv[7]) : 1.0f;
    jointDiffusivity = (argc > 8) ? (bool)(atoi(argv[8])!=0) : false;
  }
  else {
    mode = (argc > 2) ? atoi(argv[2]) : 3;
    cameraWidth = (argc > 3) ? atoi(argv[3]) : 320;
    cameraHeight = (argc > 4) ? atoi(argv[4]) : 240;
    timeStepWidth = (argc > 5) ? (float)atof(argv[5]) : 0.05f;
    numIterations = (argc > 6) ? atoi(argv[6]) : 50;
    regWeight = (argc > 7) ? (float)atof(argv[7]) : 10.0f;
    lagged_iterations = (argc > 8) ? atoi(argv[8]) : 3;
    overrelaxation = (argc > 9) ? (float)atof(argv[9]) : 1.0f;
    jointDiffusivity = (argc > 10) ? (bool)(atoi(argv[10])!=0) : false;
  }

  //----------------------------------------------------------------------------
  // Image File Mode
  //----------------------------------------------------------------------------
  if (strcmp(argv[1], "camera") != 0) {  

    // Read Input Picture
    cv::Mat inputImg = cv::imread(argv[1], -1);
    if(inputImg.data == NULL){ fprintf(stderr,"\nERROR: Could not load image %s",argv[1]); return 1; }

    inputImg.convertTo(inputImg,(inputImg.type() == CV_8UC1) ? CV_32FC1 : CV_32FC3);

    fprintf(stderr,"\nNumber of Image Channels: %i",inputImg.channels());

    // Initializa Output Picture
    cv::Mat outputImg(inputImg.rows,inputImg.cols,inputImg.type());

    gpu_diffusion((float*)inputImg.data, (float*)outputImg.data, inputImg.cols, inputImg.rows, inputImg.channels(),
      timeStepWidth, numIterations, regWeight, lagged_iterations, overrelaxation, mode, jointDiffusivity);

    cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Original Image", 100, 100);
    cv::imshow("Original Image",inputImg/255.0f);
    cv::namedWindow("Output Image", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Output Image", 500, 100);
    cv::imshow("Output Image",outputImg/255.0f);
    std::cerr << std::endl << "Press any key on the image to exit..." << std::endl;
    cv::waitKey(0);

  }  //endif image file
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
    cv::namedWindow("Input Image",CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Output Image",CV_WINDOW_AUTOSIZE);

    cv::Mat outputImg(inputImg.rows,inputImg.cols,CV_32FC3);

    while (cv::waitKey(30) < 0){
    capture >> inputImg;
    inputImg.convertTo(inputImg,(inputImg.type() == CV_8UC1) ? CV_32FC1 : CV_32FC3);
    gpu_diffusion((float*)inputImg.data, (float*)outputImg.data,
      inputImg.cols, inputImg.rows, inputImg.channels(), timeStepWidth, numIterations, 
      regWeight, lagged_iterations, overrelaxation, mode, jointDiffusivity);

    cv::imshow("Input Image",inputImg/255.0f);
    cv::imshow("Output Image",outputImg/255.0f);
    }

  } // endif "camera"

  return 0;
} 
