/*
 * face.h
 *
 *  Created on: Dec 23, 2013
 *      Author: jorge
 */

#ifndef FACE_H_
#define FACE_H_

#include <stdio.h>
#include <stdlib.h>
#include "opencv/highgui.h"
#include "stasm_lib.h"
#include "opencv2/core/core.hpp"
#include <iostream>     // std::cout
#include <algorithm>    // std::min
#include <cmath>        // std::abs
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <errno.h>
#include <ctime>
#include <LibFace.h>

using namespace cv;
using namespace std;

class Face
{
	private:


	public:
		Face() { } // private default constructor
		//libface lib functions
		IplImage* Rgb2Gray(IplImage *src);
		CvMat* IplImage2Mat(IplImage *inp_img);
		IplImage* Mat2IplImage(CvMat *inp_mat, int type);
		int Scale_Mat(CvMat *input, double scale);
		CvMat* Weighted_Gaussian(CvMat *inp, CvMat *gaussian);
		CvMat* Get_Mat(CvPoint a, int width, int height, IplImage *image);
		CvMat* Conv_Weighted_Gaussian(IplImage *inp_img, CvMat *kernel);
		CvMat* Gaussian(int size);
		IplImage* SQI(IplImage* inp);

		//FACE functions
		Mat rotateImage(const Mat& source, double angle);
		Mat correctPerpective(Mat src, Point pt1, Point pt2, Point pt3);
		double sigmoid(double x);
		double calculateMean(double value[]);
		double calculateStd(double value[]);
		double getMassCenter(std::string const& name, Mat1b const& image);
		static void error(const char* s1, const char* s2);
		Point midpoint(double x1, double y1, double x2, double y2);
		vector<Point> getStasmPts(char* imgPath, int shape);
		vector<Point> getNewStasmPts(Mat src, int shape);
		double pixelsMean(Mat img);
		vector<Mat> divideIntoSubRegions(Mat region);
		double localCorrelation(Mat rA, Mat rB);
		double globalCorrelation(Mat A, Mat B);

};

#endif /* FACE_H_ */
