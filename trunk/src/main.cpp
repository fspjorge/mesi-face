// code based on minimal2.cpp: Display the landmarks of possibly multiple faces in an image.
/**
 * Author: Jorge Pereira
 */

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

#define C_TEXT( text ) ((char*)std::string( text ).c_str())

using namespace cv;
using namespace std;

// http://stackoverflow.com/questions/2289690/opencv-how-to-rotate-iplimage
Mat rotateImage(const Mat& source, double angle) {
	Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	Mat dst;
	warpAffine(source, dst, rot_mat, source.size());

	return dst;
}

IplImage* rotateImage(const IplImage* src, int angleDegrees) {
	//take the dimention of original image
	int w = src->width;
	int h = src->height;

	// Make a new image for the result
	CvSize newSize;
	newSize.width = cvRound(w);
	newSize.height = cvRound(h);
	IplImage *imageRotated = cvCreateImage(newSize, src->depth, src->nChannels);

	// Create a map_matrix, where the left 2x2 matrix is the transform and the right 2x1 is the dimensions.
	float m[6];
	CvMat M = cvMat(2, 3, CV_32F, m);

	float angleRadians = angleDegrees * ((float) CV_PI / 180.0f);
	m[0] = (float) (cos(angleRadians));
	m[1] = (float) (sin(angleRadians));
	m[3] = -m[1];
	m[4] = m[0];
	m[2] = w * 0.5f;
	m[5] = h * 0.5f;

	// Transform the image
	cvGetQuadrangleSubPix(src, imageRotated, &M);

	return imageRotated;
}

/**
 * http://stackoverflow.com/questions/7838487/executing-cvwarpperspective-for-a-fake-deskewing-on-a-set-of-cvpoint
 */
Mat deskewing(Mat src, Point pt1, Point pt2, Point pt3, Point pt4) {

	cv::Point2f src_vertices[4];
	src_vertices[0] = pt1;
	src_vertices[1] = pt2;
	src_vertices[2] = pt3;
	src_vertices[3] = pt4;

	cv::Point2f dst_vertices[4];
	dst_vertices[0] = Point(src.cols * 0.5, pt1.y);
	dst_vertices[1] = Point(src.cols * 0.5, pt2.y);
	dst_vertices[2] = Point(src.cols, pt2.y);
	dst_vertices[3] = Point(src.cols, 0);

	Mat warpAffineMatrix = getAffineTransform(src_vertices, dst_vertices);

	cv::Mat rotated;
	cv::Size size(src.cols, src.rows);
	warpAffine(src, rotated, warpAffineMatrix, size, INTER_LINEAR,
			BORDER_CONSTANT);

	return rotated;
}

// http://cboard.cprogramming.com/contests-board/91606-fastest-sigmoid-function-2.html
/* For an array value, or most values of x, citizen_sig will return the resulting
 * value for that x. citizen_sig does not draw the resulting curve for several
 * values of x.  citizen_sig also returns EDOM if x is outside its domain.
 */
double sigmoid(double x) {
	double square_of_x, div;

	errno = 0;
	square_of_x = pow(x, 2.);
	div = sqrt(square_of_x + 1.);
	if (errno == EDOM)
		return EDOM;
	else
		return x / div;
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
// http://answers.opencv.org/question/9511/how-to-find-the-intersection-point-of-two-lines/
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

// http://stackoverflow.com/questions/15771512/compare-histograms-of-grayscale-images-in-opencv
double getMassCenter(std::string const& name, Mat1b const& image) {
	// Set histogram bins count
	int bins = 256;
	int histSize[] = { bins };

	// Set ranges for histogram bins
	float lranges[] = { 0, 256 };
	const float* ranges[] = { lranges };

	// create matrix for histogram
	cv::Mat hist;
	int channels[] = { 0 };

	// create matrix for histogram visualization
	int const hist_height = 256;
	cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

	cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges,
			true, false);

	double max_val = 0;
	minMaxLoc(hist, 0, &max_val);
	int sum1 = 0;
	int sum2 = -256;

	// visualize each bin
	for (int b = 0; b < bins; b++) {
		float const binVal = hist.at<float>(b);
		int const height = cvRound(binVal * hist_height / max_val);
		cv::line(hist_image, cv::Point(b, hist_height - height),
				cv::Point(b, hist_height), cv::Scalar::all(255));
		// mc
		sum1 += b * height;
		//printf("sum1=%d\n", sum1);
		sum2 += height;
		//printf("sum2=%d\n", sum2);
	}
	printf("sum1=%d / sum2=%d | mc = %d\n", sum1, sum2, sum1 / sum2);
	// mc formula
	int mc = sum1 / sum2;

	// show and save histograms
//    circle(hist_image,Point(mc, 255),5,cvScalar(255,0,0),-1,8);
//    cv::imwrite(std::string("histograms/") + name + ".png",hist_image); // save
//    cv::imshow(name, hist_image);

	return mc;
}

static void error(const char* s1, const char* s2) {
	printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
	exit(1);
}

Point midpoint(double x1, double y1, double x2, double y2) {
	return Point((x1 + x2) / 2, (y1 + y2) / 2);
}

// get STASM points
std::vector<cv::Point> getStasmArray(char* imgPath, int shape) {

	std::vector<cv::Point> stasmArray;

	if (!stasm_init("data", 0 /*trace*/))
		error("stasm_init failed: ", stasm_lasterr());

	static const char* path = imgPath; //"data/copy of feret_1.jpg";

	cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));

	if (!img.data)
		error("Cannot load", path);

	if (!stasm_open_image((const char*) img.data, img.cols, img.rows, path,
			1 /*multiface*/, 10 /*minwidth*/))
		error("stasm_open_image failed: ", stasm_lasterr());

	int foundface;
	float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

	while (1) {
		if (!stasm_search_auto(&foundface, landmarks))
			error("stasm_search_auto failed: ", stasm_lasterr());

		if (!foundface)
			break;      // note break

		// for demonstration, convert from Stasm 77 points to XM2VTS 68 points
		stasm_convert_shape(landmarks, shape);

		// draw the landmarks on the image as white dots
		stasm_force_points_into_image(landmarks, img.cols, img.rows);
		for (int i = 0; i < shape; i++)
			stasmArray.push_back(
					Point(cvRound(landmarks[i * 2]),
							cvRound(landmarks[i * 2 + 1])));

	}
	return stasmArray;
}

int main() {
	if (!stasm_init("data", 0 /*trace*/))
		error("stasm_init failed: ", stasm_lasterr());

	static const char* path = "scarlett_johansson_face.jpg";

	cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));

	if (!img.data)
		error("Cannot load", path);

	if (!stasm_open_image((const char*) img.data, img.cols, img.rows, path,
			1 /*multiface*/, 10 /*minwidth*/))
		error("stasm_open_image failed: ", stasm_lasterr());

	int foundface;
	float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

	int nfaces = 0;
	while (1) {
		if (!stasm_search_auto(&foundface, landmarks))
			error("stasm_search_auto failed: ", stasm_lasterr());

		if (!foundface)
			break;      // note break

		// for demonstration, convert from Stasm 77 points to XM2VTS 68 points
		stasm_convert_shape(landmarks, 77);

//        // draw the landmarks on the image as white dots
//        stasm_force_points_into_image(landmarks, img.cols, img.rows);
//        for (int i = 0; i < stasm_NLANDMARKS; i++)
//        	img(cvRound(landmarks[38*2+1]), cvRound(landmarks[38*2])) = 255;
//      		img(cvRound(landmarks[39*2+1]), cvRound(landmarks[39*2])) = 255;

		Point LPupil = Point(cvRound(landmarks[38 * 2]),
				cvRound(landmarks[38 * 2 + 1]));
		Point RPupil = Point(cvRound(landmarks[39 * 2]),
				cvRound(landmarks[39 * 2 + 1]));
		Point CNoseTip = Point(cvRound(landmarks[52 * 2]),
				cvRound(landmarks[52 * 2 + 1]));
		Point LEyebrowInner = Point(cvRound(landmarks[21 * 2]),
				cvRound(landmarks[21 * 2 + 1]));
		Point CNoseBase = Point(cvRound(landmarks[56 * 2]),
				cvRound(landmarks[56 * 2 + 1]));
		Point CTipOfChin = Point(cvRound(landmarks[6 * 2]),
				cvRound(landmarks[6 * 2 + 1]));

		/*        // draw a line between the two eyes
		 line(img, RPupil, LPupil, cvScalar(255,0,255), 1);

		 // draw a line between the right eye pupil and the nose tip
		 line(img, RPupil, CNoseTip, cvScalar(255,0,255), 1);

		 // draw a line between the left eye pupil and the nose tip
		 line(img, LPupil, CNoseTip, cvScalar(255,0,255), 1);

		 // draw a line between the right eye pupil and the nose tip
		 line(img, LEyebrowInner, CNoseTip, cvScalar(255,0,255), 1);

		 // draw a line between the left eye pupil and the nose tip
		 line(img, CNoseBase, CTipOfChin, cvScalar(255,0,255), 1);*/

		// roll
		double theta = atan2((double) LPupil.y - RPupil.y, LPupil.x - RPupil.x); //deg = * 180 / CV_PI;
		printf("theta = %f degrees\n", theta);

		double roll = min(2 * theta / CV_PI, 1.0); // rad
		printf("roll = %f\n", roll);

		// yaw ()
		// cálculo do dl e dr (http://answers.opencv.org/question/14188/calc-eucliadian-distance-between-two-single-point/)
		double dl = cv::norm(LPupil - CNoseTip);
		double dr = cv::norm(RPupil - CNoseTip);

		double yaw = (max(dl, dr) - min(dl, dr)) / max(dl, dr);
		printf("yaw = %f\n", yaw);

		// pitch
		double eu = cv::norm(LEyebrowInner - CNoseTip);
		double cd = cv::norm(CNoseBase - CTipOfChin);

		double pitch = (max(eu, cd) - min(eu, cd)) / max(eu, cd);
		printf("pitch = %f\n", pitch);

		// SP
		// being alpha = 0.1, beta = 0.6 and gamma = 0.3 | article page 153
		double alpha = 0.1;
		double beta = 0.6;
		double gamma = 0.3;

		double sp = alpha * (1 - roll) + beta * (1 - yaw) + gamma * (1 - pitch);
		printf("sp = %f\n######################\n", sp);

		// SI
		// SI = 1 - F(std(mc))

		// Finding the 8 points
		Point p1 = midpoint((double) cvRound(landmarks[0 * 2]),
				(double) cvRound(landmarks[0 * 2 + 1]),
				(double) cvRound(landmarks[58 * 2]),
				(double) cvRound(landmarks[58 * 2 + 1]));
		Point p2 = midpoint((double) cvRound(landmarks[3 * 2]),
				(double) cvRound(landmarks[3 * 2 + 1]),
				(double) cvRound(landmarks[58 * 2]),
				(double) cvRound(landmarks[58 * 2 + 1]));
		Point p3 = LEyebrowInner;
		Point p4 = midpoint((double) LEyebrowInner.x, (double) LEyebrowInner.y,
				(double) CNoseTip.x, (double) CNoseTip.y);
		Point p5 = CNoseTip;
		Point p6 = CTipOfChin;
		Point p7 = midpoint((double) cvRound(landmarks[54 * 2]),
				(double) cvRound(landmarks[54 * 2 + 1]),
				(double) cvRound(landmarks[12 * 2]),
				(double) cvRound(landmarks[12 * 2 + 1]));
		Point p8 = midpoint((double) cvRound(landmarks[54 * 2]),
				(double) cvRound(landmarks[54 * 2 + 1]),
				(double) cvRound(landmarks[9 * 2]),
				(double) cvRound(landmarks[9 * 2 + 1]));

		// For each such reference point, we select the corresponding region w of the image, whose size is proportional to the square containing the whole face.
		// http://stackoverflow.com/questions/12369697/access-sub-matrix-of-a-multidimensional-mat-in-opencv
		// Parameters:
		// x – x-coordinate of the top-left corner.
		// y – y-coordinate of the top-left corner (sometimes bottom-left corner).
		// width – width of the rectangle.
		// height – height of the rectangle.

		int length;

		if (img.rows > img.cols)
			length = img.cols;
		else
			length = img.rows;

		Mat subMatPt1 = img(
				cv::Rect(p1.x - (cvRound(length * 0.1) * 0.5),
						p1.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));
		Mat subMatPt2 = img(
				cv::Rect(p2.x - (cvRound(length * 0.1) * 0.5),
						p2.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));
		Mat subMatPt3 = img(
				cv::Rect(p3.x - (cvRound(length * 0.1) * 0.5),
						p3.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));
		Mat subMatPt4 = img(
				cv::Rect(p4.x - (cvRound(length * 0.1) * 0.5),
						p4.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));
		Mat subMatPt5 = img(
				cv::Rect(p5.x - (cvRound(length * 0.1) * 0.5),
						p5.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));
		Mat subMatPt6 = img(
				cv::Rect(p6.x - (cvRound(length * 0.1) * 0.5),
						p6.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));
		Mat subMatPt7 = img(
				cv::Rect(p7.x - (cvRound(length * 0.1) * 0.5),
						p7.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));
		Mat subMatPt8 = img(
				cv::Rect(p8.x - (cvRound(length * 0.1) * 0.5),
						p8.y - (cvRound(length * 0.1) * 0.5),
						cvRound(length * 0.1), cvRound(length * 0.1)));

		/*
		 // rectangle draws
		 rectangle( img,
		 Point( p1.x - cvRound(length*0.1/2), p1.y - (cvRound(length*0.1/2))),
		 Point( p1.x + cvRound(length*0.1/2), p1.y + cvRound(length*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,
		 1 );

		 rectangle( img,
		 Point( p2.x - cvRound(img.cols*0.1/2), p2.y - (cvRound(img.rows*0.1/2))),
		 Point( p2.x + cvRound(img.cols*0.1/2), p2.y + cvRound(img.rows*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,
		 1 );

		 rectangle( img,
		 Point( p3.x - cvRound(img.cols*0.1/2), p3.y - (cvRound(img.rows*0.1/2))),
		 Point( p3.x + cvRound(img.cols*0.1/2), p3.y + cvRound(img.rows*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,
		 1 );

		 rectangle( img,
		 Point( p4.x - cvRound(img.cols*0.1/2), p4.y - (cvRound(img.rows*0.1/2))),
		 Point( p4.x + cvRound(img.cols*0.1/2), p4.y + cvRound(img.rows*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,
		 1 );

		 rectangle( img,
		 Point( p5.x - cvRound(img.cols*0.1/2), p5.y - (cvRound(img.rows*0.1/2))),
		 Point( p5.x + cvRound(img.cols*0.1/2), p5.y + cvRound(img.rows*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,
		 1 );

		 rectangle( img,
		 Point( p6.x - cvRound(img.cols*0.1/2), p6.y - (cvRound(img.rows*0.1/2))),
		 Point( p6.x + cvRound(img.cols*0.1/2), p6.y + cvRound(img.rows*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,
		 1 );

		 rectangle( img,
		 Point( p7.x - cvRound(img.cols*0.1/2), p7.y - (cvRound(img.rows*0.1/2))),
		 Point( p7.x + cvRound(img.cols*0.1/2), p7.y + cvRound(img.rows*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,lipTop
		 1 );

		 rectangle( img,
		 Point( p8.x - cvRound(img.cols*0.1/2), p8.y - (cvRound(img.rows*0.1/2))),
		 Point( p8.x + cvRound(img.cols*0.1/2), p8.y + cvRound(img.rows*0.1/2)),
		 Scalar( 0, 255, 255 ),
		 1,
		 1 );
		 */

		cv::imwrite("histograms/w1.png", subMatPt1); // save
		cv::imwrite("histograms/w2.png", subMatPt2); // save
		cv::imwrite("histograms/w3.png", subMatPt3); // save
		cv::imwrite("histograms/w4.png", subMatPt4); // save
		cv::imwrite("histograms/w5.png", subMatPt5); // save
		cv::imwrite("histograms/w6.png", subMatPt6); // save
		cv::imwrite("histograms/w7.png", subMatPt7); // save
		cv::imwrite("histograms/w8.png", subMatPt8); // save

		// histograms
		// ver http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
		double mc_w1 = (double) getMassCenter("h1", subMatPt1);
		double mc_w2 = (double) getMassCenter("h2", subMatPt2);
		double mc_w3 = (double) getMassCenter("h3", subMatPt3);
		double mc_w4 = (double) getMassCenter("h4", subMatPt4);
		double mc_w5 = (double) getMassCenter("h5", subMatPt5);
		double mc_w6 = (double) getMassCenter("h6", subMatPt6);
		double mc_w7 = (double) getMassCenter("h7", subMatPt7);
		double mc_w8 = (double) getMassCenter("h8", subMatPt8);

		/*vector<int> vector_mc;

		 vector_mc.push_back(mc_w1);
		 vector_mc.push_back(mc_w2);
		 vector_mc.push_back(mc_w3);
		 vector_mc.push_back(mc_w4);
		 vector_mc.push_back(mc_w5);
		 vector_mc.push_back(mc_w6);
		 vector_mc.push_back(mc_w7);
		 vector_mc.push_back(mc_w8);*/

		double mc[8] =
				{ mc_w1, mc_w2, mc_w3, mc_w4, mc_w5, mc_w6, mc_w7, mc_w8 };

		//int variance = gsl_stats_variance(mc, 0, 8);

		double variance;

		// VARIANCIA | ver http://www.gnu.org/software/gsl/manual/html_node/Example-statistical-programs.html
		variance = gsl_stats_mean(mc, 1, 5);

		printf("The dataset is %g, %g, %g, %g, %g, %g, %g, %g\n", mc[0], mc[1],
				mc[2], mc[3], mc[4], mc[5], mc[6], mc[7]);

		printf("variance %f: \n", variance);

		// SI = 1 − F (std(mc))

		double si = 1 - sigmoid(variance); // MELHORAR COM CPPNETLIB?

		printf("sigmoid %f: \n", sigmoid(variance));

		printf("si %f: \n", si);

		// C. POSE NORMALIZATION ###########################################
		Mat poseNormImg(img.rows, img.cols, CV_8UC1);

		// 4.(a) rotation
		double theta_deg = theta * 180 / CV_PI;
		poseNormImg = rotateImage(img, -180 + theta_deg);
		//cv::imshow("rotatedImg", poseNormImg);

		// 4.(b) horizontal flip if dr smaller than dl
		if (gsl_fcmp(dl, dr, DBL_EPSILON) == 1) // x = y returns 0; if x < y returns -1; x > y returns +1;
			flip(poseNormImg, poseNormImg, 1);
		else
			// do nothing
			//flip(rotatedImg,rotatedImg,1); // http://www.technolabsz.com/2012/08/how-to-flip-image-in-opencv.html

			//cv::imshow("flippedImg", poseNormImg);
			imwrite("poseNormImg.jpg", poseNormImg);

		// 4. (c) stretching

		std::vector<Point> stasmVector = getStasmArray("poseNormImg.jpg", 68);

		Point topCenter = Point(poseNormImg.cols * 0.5, 0);
		Point noseTop = midpoint(stasmVector.at(24).x, stasmVector.at(24).y,
				stasmVector.at(18).x, stasmVector.at(18).y);
		Point noseTip = stasmVector.at(67);
		Point noseBase = stasmVector.at(41);
		Point lipTop = stasmVector.at(51);
		Point lipBottom = stasmVector.at(57);
		Point chinTip = stasmVector.at(7);
		Point bottomCenter = Point(poseNormImg.cols * 0.5, poseNormImg.rows);
		Point bottomRight = Point(poseNormImg.cols, poseNormImg.rows);
		Point topRight = Point(poseNormImg.cols, 0);

		line(poseNormImg, topCenter, noseTop, 255, 1, 8, 0);
		line(poseNormImg, noseTop, noseTip, 255, 1, 8, 0);
		line(poseNormImg, noseTip, noseBase, 255, 1, 8, 0);
		line(poseNormImg, noseBase, lipTop, 255, 1, 8, 0);
		line(poseNormImg, lipTop, lipBottom, 255, 1, 8, 0);
		line(poseNormImg, lipBottom, chinTip, 255, 1, 8, 0);
		line(poseNormImg, chinTip, bottomCenter, 255, 1, 8, 0);
		line(poseNormImg, bottomCenter, bottomRight, 255, 1, 8, 0);
		line(poseNormImg, bottomRight, topRight, 255, 1, 8, 0);
		line(poseNormImg, topRight, topCenter, 255, 1, 8, 0);

		imshow("poseNormImg", poseNormImg);

		cv::Mat out(poseNormImg.rows, poseNormImg.cols, CV_8U);
		cv::Mat out2(poseNormImg.rows, poseNormImg.cols, CV_8U);

		Mat d1 = deskewing(poseNormImg, topCenter, noseTop,
				Point(poseNormImg.cols, noseTop.y), topRight);
		Mat d2 = deskewing(poseNormImg, noseTop, noseTip,
				Point(poseNormImg.cols, noseTip.y),
				Point(poseNormImg.cols, noseTop.y));
		Mat d3 = deskewing(poseNormImg, noseTip, noseBase,
				Point(poseNormImg.cols, noseBase.y),
				Point(poseNormImg.cols, noseTip.y));
		Mat d4 = deskewing(poseNormImg, noseBase, lipTop,
				Point(poseNormImg.cols, lipTop.y),
				Point(poseNormImg.cols, noseBase.y));
		Mat d5 = deskewing(poseNormImg, lipTop, lipBottom,
				Point(poseNormImg.cols, lipBottom.y),
				Point(poseNormImg.cols, lipTop.y));
		Mat d6 = deskewing(poseNormImg, lipBottom, chinTip,
				Point(poseNormImg.cols, chinTip.y),
				Point(poseNormImg.cols, lipBottom.y));
		Mat d7 = deskewing(poseNormImg, chinTip, bottomCenter, bottomRight,
				Point(poseNormImg.cols, chinTip.y));

		for (int row = 0; row < poseNormImg.rows; row++) {
			if (row < noseTop.y) {
				d1.row(row).copyTo(out.row(row));
			} else if (row < noseTip.y) {
				d2.row(row).copyTo(out.row(row));
			} else if (row < noseBase.y) {
				d3.row(row).copyTo(out.row(row));
			} else if (row < lipTop.y) {
				d4.row(row).copyTo(out.row(row));
			} else if (row < lipBottom.y) {
				d5.row(row).copyTo(out.row(row));
			} else if (row < chinTip.y) {
				d6.row(row).copyTo(out.row(row));
			} else {
				d7.row(row).copyTo(out.row(row));
			}
		}

		imshow("stretched", out);

		for (int row = 0; row < poseNormImg.rows; row++) {
			for (int col = 0; col < poseNormImg.cols; col++) {
				if (col < poseNormImg.cols * 0.5)
					out2.at<uchar>(row, col) = out.at<uchar>(row, -col);
				else
					out2.at<uchar>(row, col) = out.at<uchar>(row, col);
			}
		}

		imshow("mirror", out2);

		// 4.(f)
		Mat illumNorn;
		equalizeHist(out2, illumNorn);

		imshow("illumNorn", illumNorn);

		nfaces++;
	}

	printf("%s: %d face(s)\n", path, nfaces);
	fflush(stdout);

//    cv::imwrite("minimal2.bmp", img);
	cv::waitKey(0);

	return 0;
}
