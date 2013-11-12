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
#include <LibFace.h>

#define C_TEXT( text ) ((char*)std::string( text ).c_str())

using namespace cv;
using namespace std;

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

// http://stackoverflow.com/questions/2289690/opencv-how-to-rotate-iplimage
Mat rotateImage(const Mat& source, double angle) {
	Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	Mat dst;
	warpAffine(source, dst, rot_mat, source.size());

	return dst;
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

	static const char* path = "2013-11-12-001446.jpg";

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

		imshow("Original img", img);

		// C. POSE NORMALIZATION ###########################################

		Mat poseNormImg(img.rows, img.cols, CV_8UC1);

		// 4.(a) rotation
		double theta_deg = theta * 180 / CV_PI;
		img = rotateImage(img, -180 + theta_deg);
		//cv::imshow("rotatedImg", poseNormImg);

		// 4.(b) horizontal flip if dr smaller than dl
		if (gsl_fcmp(dl, dr, DBL_EPSILON) == 1) {
			flip(img, img, 1); // x = y returns 0; if x < y returns -1; x > y returns +1;
		}

		// imagem rodada theta graus, nova verificação das coordenadas dos pontos devido à rotação
		imwrite("img.jpg", img);

		std::vector<Point> roi_vector = getStasmArray("img.jpg", 77);

		Point CForehead = roi_vector.at(14);
		Point LJawNoseline = roi_vector.at(2);
		Point RJaw11 = roi_vector.at(11);

		int width = RJaw11.x - LJawNoseline.x;
		int height = CTipOfChin.y - CForehead.y;

		//crop face
		Mat roi = img(
				Rect(LJawNoseline.x - 5, CForehead.y, width + 5, height + 5));

		imwrite("img.jpg", roi);
		roi_vector.clear();

		// 4. (c) stretching

		std::vector<Point> stasm_vector = getStasmArray("img.jpg", 68);

		Point topCenter = Point(roi.cols * 0.5, 0);
		Point noseTop = midpoint(stasm_vector.at(24).x, stasm_vector.at(24).y,
				stasm_vector.at(18).x, stasm_vector.at(18).y);
		Point noseTip = stasm_vector.at(67);
		Point noseBase = stasm_vector.at(41);
		Point lipTop = stasm_vector.at(51);
		Point lipBottom = stasm_vector.at(57);
		Point chinTip = stasm_vector.at(7);
		Point bottomCenter = Point(roi.cols * 0.5, roi.rows);
		Point bottomRight = Point(roi.cols, roi.rows);
		Point topRight = Point(roi.cols, 0);

		/*
		int thickness = -1;
		int lineType = 8;

		circle(roi, topCenter, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, noseTop, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, noseTip, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, noseBase, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, lipTop, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, lipBottom, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, chinTip, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, bottomCenter, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, bottomRight, 2, Scalar(0, 0, 255), thickness, lineType);
		circle(roi, topRight, 2, Scalar(0, 0, 255), thickness, lineType);

		line(roi, topCenter, noseTop, 255, 1, 8, 0);
		line(roi, noseTop, noseTip, 255, 1, 8, 0);
		line(roi, noseTip, noseBase, 255, 1, 8, 0);
		line(roi, noseBase, lipTop, 255, 1, 8, 0);
		line(roi, lipTop, lipBottom, 255, 1, 8, 0);
		line(roi, lipBottom, chinTip, 255, 1, 8, 0);
		line(roi, chinTip, bottomCenter, 255, 1, 8, 0);
		line(roi, bottomCenter, bottomRight, 255, 1, 8, 0);
		line(roi, bottomRight, topRight, 255, 1, 8, 0);
		line(roi, topRight, topCenter, 255, 1, 8, 0);
		*/

		imshow("roi - lines", roi);

		cv::Mat out(roi.rows, roi.cols, CV_8U);
		cv::Mat out2(roi.rows, roi.cols, CV_8U);

		Mat d1 = deskewing(roi, topCenter, noseTop, Point(roi.cols, noseTop.y),
				topRight);
		Mat d2 = deskewing(roi, noseTop, noseTip, Point(roi.cols, noseTip.y),
				Point(roi.cols, noseTop.y));
		Mat d3 = deskewing(roi, noseTip, noseBase, Point(roi.cols, noseBase.y),
				Point(roi.cols, noseTip.y));
		Mat d4 = deskewing(roi, noseBase, lipTop, Point(roi.cols, lipTop.y),
				Point(roi.cols, noseBase.y));
		Mat d5 = deskewing(roi, lipTop, lipBottom, Point(roi.cols, lipBottom.y),
				Point(roi.cols, lipTop.y));
		Mat d6 = deskewing(roi, lipBottom, chinTip, Point(roi.cols, chinTip.y),
				Point(roi.cols, lipBottom.y));
		Mat d7 = deskewing(roi, chinTip, bottomCenter, bottomRight,
				Point(roi.cols, chinTip.y));

		for (int row = 0; row < roi.rows; row++) {
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

		imshow("rows stretched equally", out);

		for (int row = 0; row < roi.rows; row++) {
			for (int col = 0; col < roi.cols; col++) {
				if (col < roi.cols * 0.5)
					out2.at<uchar>(row, col) = out.at<uchar>(row, -col);
				else
					out2.at<uchar>(row, col) = out.at<uchar>(row, col);
			}
		}

		imshow("Mirror right to left", out2);

		// 4.(f)
		Mat illumNorn;

		IplImage copy = out2;

		illumNorn = Mat(SQI(&copy));

		/*Mat imageSQItest;
		imageSQItest = imread("Screenshot - 09-11-2013 - 16:20:17.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file

		if(! imageSQItest.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the image" << std::endl ;
			return -1;
		}

		IplImage copy = imageSQItest;

		illumNorn = Mat(SQI(&copy));*/

		imshow("illumination norm with SQI", illumNorn);

		nfaces++;
	}

	printf("%s: %d face(s)\n", path, nfaces);
	fflush(stdout);

//    cv::imwrite("minimal2.bmp", img);
	cv::waitKey(0);

	return 0;
}

IplImage* Rgb2Gray(IplImage *src) {
	IplImage *result;
	int i, j;

	int step_src, step_res;

	result = cvCreateImage(cvSize(src->width, src->height), src->depth, 1);

	unsigned char *src_data;
	unsigned char *res_data;

	src_data = (unsigned char*) src->imageData;
	res_data = (unsigned char*) result->imageData;

	step_src = src->widthStep;
	step_res = result->widthStep;

	for (i = 0; i < src->height; i = i + 1) {
		for (j = 0; j < (src->width * src->nChannels); j = j + src->nChannels) {
			res_data[j / src->nChannels] = (unsigned char) ((0.3
					* (double) src_data[j + 2])
					+ (0.59 * (double) src_data[j + 1])
					+ (0.11 * (double) src_data[j]));
			//res_data[j+2]=src_data[j+2]; BGR format gray
		}
		src_data += step_src;
		res_data += step_res;
	}

	return result;
}

CvMat* IplImage2Mat(IplImage *inp_img) {

	CvMat *result;
	IplImage *temp;
	int i, j;
	unsigned char tmp_char;
	temp = Rgb2Gray(inp_img);
	result = cvCreateMat(temp->height, temp->width, CV_64FC1);

	for (i = 0; i < temp->height; i++) {
		for (j = 0; j < temp->width; j++) {
			tmp_char =
					(unsigned char) temp->imageData[(i * temp->widthStep) + j];
			cvmSet(result, i, j, (double) tmp_char);
		}
	}

	cvReleaseImage(&temp);
	return result;
}

IplImage* Mat2IplImage(CvMat *inp_mat, int type) {

	IplImage *result;
	int i, j;
	double tmp_val;

	if (type == 0) {
		result = cvCreateImage(cvSize(inp_mat->cols, inp_mat->rows),
				IPL_DEPTH_8U, 1);
	} else if (type == 1) {
		result = cvCreateImage(cvSize(inp_mat->cols, inp_mat->rows),
				IPL_DEPTH_32F, 1);
	} else {
		return 0;
	}

	for (i = 0; i < result->height; i++) {
		for (j = 0; j < result->width; j++) {
			tmp_val = cvmGet(inp_mat, i, j);
			result->imageData[(i * result->widthStep) + j] =
					(unsigned char) tmp_val;
		}
	}

	return result;
}

int Scale_Mat(CvMat *input, double scale) {
	double tmp;
	double val;
	double min;
	double max;
	min = 20000.0;
	max = -20000.0;
	int i, j;
	for (i = 0; i < input->rows; i++) {
		for (j = 0; j < input->cols; j++) {
			tmp = cvmGet(input, i, j);
			//if(tmp==-INF)
			//      printf("%d--%d\n",i,j);
			if (tmp < min)
				min = tmp;
			if (tmp > max)
				max = tmp;
		}
	}

//printf("%g - %g\n",min,max);

	for (i = 0; i < input->rows; i++) {
		for (j = 0; j < input->cols; j++) {
			tmp = cvmGet(input, i, j);
			val = scale * ((tmp - min) / (max - min));
			//      printf("%g * ",val);
			cvmSet(input, i, j, val);
		}
	}

	/*max=0.0;
	 for(i=0;i<input->rows;i++)
	 {
	 for(j=0;j<input->cols;j++)
	 {
	 tmp=cvmGet(input,i,j);
	 if(max<tmp)
	 max=tmp;
	 }
	 }

	 printf("max =%g\n",max);
	 */
	return 1;
}

CvMat* Weighted_Gaussian(CvMat *inp, CvMat *gaussian) {
	double sum;
	double threshold;
	int i, j;
	double tmp1, tmp2;
	double scl_factor;
	double lt_cnt;
	double gt_cnt;
	bool mr_t_thr;

	CvMat* w_gauss = cvCreateMat(gaussian->rows, gaussian->cols, CV_64FC1);
	cvSetZero(w_gauss);
	sum = 0.0;
	for (i = 0; i < inp->rows; i++) {
		for (j = 0; j < inp->cols; j++) {
			sum = sum + cvmGet(inp, i, j);
		}
	}

	threshold = sum / (inp->cols * inp->rows);
	lt_cnt = 0;
	gt_cnt = 0;
	for (i = 0; i < inp->rows; i++) {
		for (j = 0; j < inp->cols; j++) {
			tmp1 = cvmGet(inp, i, j);
			if (tmp1 > threshold) {
				gt_cnt = gt_cnt + 1;
			} else {
				lt_cnt = lt_cnt + 1;
			}
		}
	}

	if (gt_cnt > lt_cnt) {
		mr_t_thr = true;
	} else {
		mr_t_thr = false;
	}

	scl_factor = 0.0;
	for (i = 0; i < inp->rows; i++) {
		for (j = 0; j < inp->cols; j++) {
			tmp1 = cvmGet(inp, i, j);
			if (((tmp1 > threshold) && mr_t_thr == false)
					|| ((tmp1 < threshold) && mr_t_thr == true)) {

				cvmSet(w_gauss, i, j, 0.0);
			} else {
				tmp2 = cvmGet(gaussian, i, j);
				scl_factor = scl_factor + tmp2;
				cvmSet(w_gauss, i, j, tmp2);
			}
		}
	}

	/*Normalizing the weighted gaussian matrix*/

	for (i = 0; i < inp->rows; i++) {
		for (j = 0; j < inp->cols; j++) {
			tmp1 = cvmGet(w_gauss, i, j);
			tmp2 = tmp1 / scl_factor;
			cvmSet(w_gauss, i, j, tmp2);
		}
	}

	return w_gauss;
}

CvMat* Get_Mat(CvPoint a, int width, int height, IplImage *image) {

	CvMat *fea_ar;

	unsigned char t_val;
	int h_i, w_i;

	fea_ar = cvCreateMat(height, width, CV_64FC1);

	cvSetZero(fea_ar);

	int i, j;

	for (i = a.y; i < (a.y + height); i++) {
		for (j = a.x; j < (a.x + width); j++) {
			if ((i >= 0) && (j >= 0) && (i < (image->height))
					&& (j < (image->width))) {
				t_val = (unsigned char) image->imageData[(i * image->widthStep)
						+ j];
				cvmSet(fea_ar, i - a.y, j - a.x, (double) t_val);
			} else {
				if (j < 0) {
					w_i = image->width + j;
				} else if (j >= image->width) {
					w_i = j - image->width;
				} else {
					w_i = j;
				}

				if (i < 0) {
					h_i = -i;
				} else if (i >= image->height) {
					h_i = image->height - (i - image->height);
				} else {
					h_i = i;
				}

				t_val =
						(unsigned char) image->imageData[(h_i * image->widthStep)
								+ w_i];
				cvmSet(fea_ar, i - a.y, j - a.x, (double) t_val);
			}
		}

	}

	return (fea_ar);
}

/**
 *
 */
CvMat* Conv_Weighted_Gaussian(IplImage *inp_img, CvMat *kernel) {
//IplImage *result;
	int i, j;
	CvPoint start;
	CvMat *tmp;
	CvMat *ddd;
	CvMat *w_gauss;
//result=cvCloneImage(inp_img);

	double val;
	ddd = cvCreateMat(inp_img->height, inp_img->width, CV_64FC1);

	for (i = 0; i < inp_img->height; i++) {
		for (j = 0; j < inp_img->width; j++) {
			start.x = j - (kernel->cols / 2);
			start.y = i - (kernel->rows / 2);

			tmp = Get_Mat(start, kernel->cols, kernel->rows, inp_img);

			w_gauss = Weighted_Gaussian(tmp, kernel);

			val = cvDotProduct(w_gauss, tmp);

			cvmSet(ddd, i, j, val);

			cvReleaseMat(&tmp);
			cvReleaseMat(&w_gauss);

		}
	}

	/*
	 cvNamedWindow("fdf",1);
	 cvShowImage("fdf",ddd);
	 cvWaitKey(0);
	 */

	return ddd;

}

CvMat* Gaussian(int size) {
	CvMat *res;
	int i, j;
	int x, y;
	double tmp;
	double sigma;
	int halfsize;

	sigma = (double) size / 5;
	res = cvCreateMat(size, size, CV_64FC1);
	halfsize = size / 2;

	for (i = 0; i < res->rows; i++) {
		for (j = 0; j < res->cols; j++) {
			x = j - halfsize;
			y = i - halfsize;
			tmp = exp(-(double) (x * x + y * y) / sigma);
			cvmSet(res, i, j, tmp);
		}
	}

	return res;
}

IplImage* SQI(IplImage* inp) {
	int num_filters;
	int size[3];
	IplImage *ttt;
	CvMat *filtered_image[3];
	CvMat *qi[3];
	CvMat *inp_mat;
	CvMat *res;
	IplImage *res_img;
	int i, j, k;
	double tmp1, tmp2, tmp3;
	CvMat *g_ker;

	num_filters = 3;

	size[0] = 3;
	size[1] = 9;
	size[2] = 15;

	inp_mat = IplImage2Mat(inp);

// cvNamedWindow("ttt",1);

//cvShowImage("ttt",inp);
//cvWaitKey(0);

	for (i = 0; i < num_filters; i++) {
		g_ker = Gaussian(size[i]);
		filtered_image[i] = Conv_Weighted_Gaussian(inp, g_ker);

		/*  ttt=Mat2IplImage(filtered_image[i],0);
		 cvShowImage("ttt",ttt);
		 cvWaitKey(0);
		 cvReleaseImage(&ttt);
		 */
		cvReleaseMat(&g_ker);
		qi[i] = cvCreateMat(inp_mat->rows, inp_mat->cols, CV_64FC1);
		for (j = 0; j < inp_mat->rows; j++) {
			for (k = 0; k < inp_mat->cols; k++) {
				tmp1 = cvmGet(inp_mat, j, k);
				tmp2 = cvmGet(filtered_image[i], j, k);

				//  if(tmp1==0.0 || tmp2==0.0 )
				//{
				//  tmp3=0.0;
				//}
				// else{
				tmp3 = log10((tmp1 + 1.0) / (tmp2 + 1.0));
				//}
				// printf("%g *",tmp3);
				cvmSet(qi[i], j, k, tmp3);

			}
		}
		cvReleaseMat(&filtered_image[i]);
	}

	res = cvCreateMat(inp_mat->rows, inp_mat->cols, CV_64FC1);
	cvSetZero(res);
	for (i = 0; i < num_filters; i++) {
		for (j = 0; j < inp_mat->rows; j++) {
			for (k = 0; k < inp_mat->cols; k++) {
				tmp1 = cvmGet(qi[i], j, k);
				tmp2 = cvmGet(res, j, k);
#ifdef DEBUG
				//       printf("%g * ",tmp1+tmp2);
#endif
				cvmSet(res, j, k, tmp1 + tmp2);
			}
		}
		cvReleaseMat(&qi[i]);
	}

	Scale_Mat(res, 255);
	res_img = Mat2IplImage(res, 0);

	cvReleaseMat(&res);

	return res_img;
}
