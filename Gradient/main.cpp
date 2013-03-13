/****************************************************************************\
*      --- Practical Course: GPU Programming in Computer Vision ---
*
* time:    winter term 2012/13 / March 11-18, 2013
*
* project: gradient
* file:    main.cpp
*
*
*             !THIS FILE IS SUPPOSED TO REMAIN UNCHANGED!
\****************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "gradient.cuh"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;




int main(int argc,char **argv) {
  if (argc <= 1) {
    cout << "Too few arguments!\n" << endl;
    cout << "Usage: " << argv[0] << "[ImageName] [mode=3]" << endl;
    cout << endl << "Available modes:" << endl;
    cout << "0 - X Derivative" << endl;
    cout << "1 - Y Derivative" << endl;
    cout << "2 - Gradient Magnitude" << endl;
    cout << "3 - ALL" << endl;
    
    return 1;
  }

  const int mode = (argc > 2) ? atoi(argv[2]) : 3;
  
  if (!checkStudentNameAndID()) std::cout << "WARNING: Please enter your correct student name and ID in file gradient.cu!"  << std::endl;

  // Read Input Picture
//  IplImage* img = cvLoadImage(argv[1],-1);
  cv::Mat inputImg = cv::imread(argv[1], -1);
  if(inputImg.data == NULL){ fprintf(stderr,"\nERROR: Could not load image %s",argv[1]); return 1; }
  inputImg.convertTo(inputImg,(inputImg.type() == CV_8UC1) ? CV_32FC1 : CV_32FC3);

  cv::namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
  cvMoveWindow("Original Image", 100, 100);
  cv::imshow("Original Image",inputImg/255.0f);
  cv::Mat derXImg(inputImg.rows,inputImg.cols,inputImg.type());
  cv::Mat derYImg(inputImg.rows,inputImg.cols,inputImg.type());
  cv::Mat magnImg(inputImg.rows,inputImg.cols,inputImg.type());

  if(mode == 0) {

		gpu_derivative_sm_d((float*)inputImg.data, (float*)derXImg.data, inputImg.cols, inputImg.rows, inputImg.channels(), 0);

		cv::namedWindow("Derivative X", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative X", 500, 100);
    cv::imshow("Derivative X", (derXImg+128.0f)/255.0f);
  }
  else if (mode == 1) {
		
		gpu_derivative_sm_d((float*)inputImg.data, (float*)derYImg.data, inputImg.cols, inputImg.rows, inputImg.channels(), 1);

		cv::namedWindow("Derivative Y", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative Y", 500, 100);
    cv::imshow("Derivative Y", (derYImg+128.0f)/255.0f);
  }
  else if(mode == 2) {

		gpu_derivative_sm_d((float*)inputImg.data, (float*)magnImg.data, inputImg.cols, inputImg.rows, inputImg.channels(), 2);

		cv::namedWindow("Magnitude", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Magnitude", 500, 100);
    cv::imshow("Magnitude", magnImg/255.0f);
  }
  else if(mode == 3) {

		gpu_derivative_sm_d((float*)inputImg.data, (float*)derXImg.data, inputImg.cols, inputImg.rows, inputImg.channels(), 0);
		gpu_derivative_sm_d((float*)inputImg.data, (float*)derYImg.data, inputImg.cols, inputImg.rows, inputImg.channels(), 1);
		gpu_derivative_sm_d((float*)inputImg.data, (float*)magnImg.data, inputImg.cols, inputImg.rows, inputImg.channels(), 2);

		cv::namedWindow("Derivative X", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative X", 500, 100);
    cv::imshow("Derivative X", (derXImg+128.0f)/255.0f);

    cv::namedWindow("Derivative Y", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Derivative Y", 500, 100);
    cv::imshow("Derivative Y", (derYImg+128.0f)/255.0f);

    cv::namedWindow("Magnitude", CV_WINDOW_AUTOSIZE);
    cvMoveWindow("Magnitude", 500, 100);
    cv::imshow("Magnitude", magnImg/255.0f);
  }

  std::cerr << std::endl << "Press any key on the image to exit..." << std::endl;
  cv::waitKey(0);


  return 0;
} 
