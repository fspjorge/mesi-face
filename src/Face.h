/*
 * face.h
 *
 *  Created on: Dec 23, 2013
 *      Author: Jorge Silva Pereira
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
#include <string>
#include <math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <errno.h>
#include <ctime>
#include <unistd.h>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"

using namespace cv;
using namespace std;

class Face {
private:
	vector<Point> stasmPts;
	Mat_<unsigned char> face;
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
	//aux functions
	double computePixelsMean(Mat img);
	vector<Mat> divideIntoSubRegions(Mat region, int size);
	Point rotatePoint(Point pt, double angle);
	double computePointsAngle(Point pt1, Point pt2);
	Mat rotateImage(const Mat& source, double angle);
	Mat correctBandPerpective(Mat src, Point pt1, Point pt2, Point pt3);
	double computeSigmoid(double x);
	double computeMean(double value[]);
	double computeStdDev(double value[]);
	double computeMassCenter(std::string const& name, Mat1b const& image);
	static void error(const char* s1, const char* s2);
	Point calcMidpoint(double x1, double y1, double x2, double y2);
	vector<Point> getStasmPts(char* imgPath, int shape);
public:
	Face(const char* imgPath);
	const Mat_<unsigned char>& loadMat() const {
		return face;
	}
	double computeLocalCorrelation(Mat rA, Mat rB);
	double computeGlobalCorrelation(Mat A, Mat B);
	vector<Point> getStasmPts();
	double computeSp(Point LPupil, Point RPupil, Point LEyebrowInner,
			Point CNoseTip, Point CNoseBase, Point CTipOfChin);
	double computeSi(Point LPupil, Point RPupil, Point LEyebrowInner,
			Point CNoseTip, Point CNoseBase, Point CTipOfChin);
	Mat normalizePose(Mat face, Point LPupil, Point RPupil, Point LEyebrowInner,
			Point CNoseTip, Point CNoseBase, Point CTipOfChin);
	Mat normalizeIllumination(Mat face);
	double computeRelativeDistance(Mat p);
	double computeQls(double x, double xmax);
	double computeSRR(string imgPath, string modelsFolder);
};

#endif /* FACE_H_ */
