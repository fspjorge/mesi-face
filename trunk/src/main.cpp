/**
 * Author: Jorge Pereira
 * Code based on the example "minimal2.cpp" provided by Stephen Milborrow along with the STASM library.
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

/**
 * Rotate image depending on the angle.
 * From http://stackoverflow.com/questions/2289690/opencv-how-to-rotate-iplimage
 */
Mat rotateImage(const Mat& source, double angle) {
	Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	Mat dst;
	warpAffine(source, dst, rot_mat, source.size());

	return dst;
}

/**
 * Affine transformations to correct the band perspective.
 * Code based on http://stackoverflow.com/questions/7838487/executing-cvwarpperspective-for-a-fake-deskewing-on-a-set-of-cvpoint
 * 1-----3
 * |  -
 * 2-
 */
Mat correctPerpective(Mat src, Point pt1, Point pt2, Point pt3) {
	Mat warp_dst;
	Size size(src.cols, src.rows);
	Point2f src_vertices[3];
	Point2f dst_vertices[3];

	src_vertices[0] = pt1;
	src_vertices[1] = pt2;
	src_vertices[2] = pt3;

	dst_vertices[0] = Point(src.cols * 0.5 - 1, pt1.y);
	dst_vertices[1] = Point(src.cols * 0.5 - 1, pt2.y);
	dst_vertices[2] = Point(src.cols, pt2.y);

	Mat warpAffineMatrix = getAffineTransform(src_vertices, dst_vertices);

	warpAffine(src, warp_dst, warpAffineMatrix, size, INTER_LINEAR,
			BORDER_CONSTANT);

	return warp_dst;
}

/**
 * Sigmoid function.
 */
double sigmoid(double x) {

	double s = 0.0;
	s = 1.0 / (1.0 + exp(-x / 160));

	return s;
}

/**
 * Calculate mean. http://www.softwareandfinance.com/CPP/MeanVarianceStdDevi.html
 */
double calculateMean(double value[]) {
	double max = 8;

	double sum = 0;
	for (int i = 0; i < max; i++)
		sum += value[i];

	return (sum / max);
}

/**
 * Calculate Variance. From http://www.softwareandfinance.com/CPP/MeanVarianceStdDevi.html
 */
double calculateStd(double value[]) {
	int max = 8;
	double mean;
	mean = calculateMean(value);

	double temp = 0;
	for (int i = 0; i < max; i++) {
		temp += (value[i] - mean) * (value[i] - mean);
	}
	double deviance = temp / max;

	return sqrt(deviance);
}

/**
 * Returns the value of the center of mass for each submatrix.
 * From http://stackoverflow.com/questions/15771512/compare-histograms-of-grayscale-images-in-opencv
 */
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
	double sum1 = 0;
	double sum2 = -256;

	// visualize each bin
	for (int b = 0; b < bins; b++) {
		float const binVal = hist.at<float>(b);
		int const height = cvRound(binVal * hist_height / max_val);
		cv::line(hist_image, cv::Point(b, hist_height - height),
				cv::Point(b, hist_height), cv::Scalar::all(255));
		// mc
		sum1 += b * (hist_height - height);
		//printf("sum1=%d\n", sum1);
		sum2 += hist_height - height;
		//printf("sum2=%d\n", sum2);
	}
	printf("sum1=%f / sum2=%f | mc = %f\n", sum1, sum2, sum1 / sum2);
	// mc formula
	double mc = sum1 / sum2;

	// show and save histograms
	circle(hist_image, Point(mc, 255), 5, cvScalar(255, 0, 0), -1, 8);
	cv::imwrite(std::string("histograms/") + name + ".png", hist_image); // save
//    cv::imshow(name, hist_image);

	return mc;
}

static void error(const char* s1, const char* s2) {
	printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
	exit(1);
}

/**
 * Returns the midpoint coordinates.
 */
Point midpoint(double x1, double y1, double x2, double y2) {
	return Point((x1 + x2) / 2, (y1 + y2) / 2);
}

/**
 * Get STASM points array from image.
 */
vector<Point> getStasmPts(char* imgPath, int shape) {

	vector<Point> pts_array;

	if (!stasm_init("data", 0 /*trace*/))
		error("stasm_init failed: ", stasm_lasterr());

	static const char* path = imgPath;

	Mat_<unsigned char> img(imread(path, CV_LOAD_IMAGE_GRAYSCALE));

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
			pts_array.push_back(
					Point(cvRound(landmarks[i * 2]),
							cvRound(landmarks[i * 2 + 1])));

	}
	return pts_array;
}

/**
 * Corrects the STASM points coordinates caused by the rotation and horizontal flip on the original image.
 */
vector<Point> getNewStasmPts(Mat src, int shape) {
	imwrite("tmp.jpg", src);
	char *tmp = new char[10];
	strcpy(tmp, "tmp.jpg");
	vector<Point> pts = getStasmPts(tmp, shape);

	if (remove("tmp.jpg") != 0)
		perror("Error deleting file tmp.jpg");
	else
		puts("File tmp.jpg successfully deleted");

	return pts;
}

/**
 * Calculates the average value of the pixels of a matrix.
 */
double pixelsMean(Mat img) {
	vector<Mat> channels;
	split(img, channels);
	Scalar m = mean(channels[0]);

	return m[0];
}

/**
 * Divides a region in 8x8 square sub-regions and puts them in a vector.
 * Returns a vector of sub-regions (cv::Mat).
 */
vector<Mat> divideIntoSubRegions(Mat region) {

	vector<Mat> subRegions;
	int roiSize = 8;

	for (int i = 0; i < region.cols/roiSize; ++i) {
		for (int j = 0; j < region.rows/roiSize; ++j) {

			try {
				if (region.at<uchar>(i, j) > 0) {
					Mat subRegion = region(Rect(i, j, roiSize, roiSize));
					subRegions.push_back(subRegion);
					cout << "i = " << i << " j = " << j << endl;
				}
			} catch (cv::Exception& e) {
				cout << e.msg << endl;
			}
		}
	}

	cout << "Dividing a region with size " << region.cols << "x" << region.rows
			<< " results in a total of " << subRegions.size()
			<< " sub-regions, each one with size " << subRegions.at(0).cols
			<< "x" << subRegions.at(0).rows << endl;

	return subRegions;
}

/**
 * Calculates the correlation between two cv::Mat.
 */
double localCorrelation(Mat rA, Mat rB) {

	double rAPixMean = pixelsMean(rA);
	double rBPixMean = pixelsMean(rB);

	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;

	for (int j = 0; j < rA.rows; j++) {
		for (int i = 0; i < rA.cols; i++) {

			sum1 += (rA.at<uchar>(j, i) - rAPixMean)
					* (rB.at<uchar>(j, i) - rBPixMean);
			sum2 += pow(rA.at<uchar>(j, i) - rAPixMean, 2.0);
			sum3 += pow(rB.at<uchar>(j, i) - rBPixMean, 2.0);
		}
	}

	double corr = sum1 / (sqrt(sum2 * sum3));
	cout << "corrLocal = " << corr << endl;

	return corr;
}

/**
 * Soma dos máximos das subregiões (WORK IN PROGRESS).
 */
double globalCorrelation(Mat A, Mat B) {
	double SumAB = 0.0;

	cv::resize(A, B, A.size(), 0, 0, cv::INTER_CUBIC);

	vector<Mat> subRegionsOfA = divideIntoSubRegions(A);
	vector<Mat> subRegionsOfB = divideIntoSubRegions(B);

	unsigned int regionsPerLine = div(A.cols, 8).quot;

	cout << "regionsPerLine = " << regionsPerLine << endl;

	int count = 0;

	for (unsigned int i = 0; i < subRegionsOfA.size(); i++) {

		if (i == 0) {
			/**
			 * ____
			 * |__|
			 *
			 */
			cout << "i = 0 " << endl;
			cout << "A comparar a região i(" << i << ") com a região i(" << i << ")"<< endl;
			localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i));
		} else if (i > 0 && i < regionsPerLine) {

			/**
			 * ______
			 * |__|__|
			 *
			 */
			cout << "i > 0 && i < regionsPerLine " << endl;
			cout << "A comparar a região i(" << i << ") com a região i(" << i << ")"<< endl;
			cout << "A comparar a região i(" << i << ") com a região i(" << (i-1) << ")"<< endl;

			localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i));
			localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i - 1));
		} else if (i % regionsPerLine == 0) {
			/**
			 * ____
			 * |__|
			 * |__|
			 *
			 */
			cout << "i > 0 && i < regionsPerLine " << endl;
			cout << "A comparar a região i(" << i << ") com a região i(" << i << ")"<< endl;
			cout << "A comparar a região i(" << i << ") com a região i-regionsPerLine(" << (i-regionsPerLine) << ")"<< endl;

			localCorrelation(subRegionsOfA.at(i),
					subRegionsOfB.at(i - regionsPerLine));
			localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i));
		} else {
			/**
			 * ______
			 * |__|__|
			 * |__|__|
			 *
			 */

			cout << "i > 0 && i < regionsPerLine " << endl;
			cout << "A comparar a região " << i << " com a região i(" << i << ")" << endl;
			cout << "A comparar a região " << i << " com a região i-1(" << (i - 1) << ")" << endl;
			cout << "A comparar a região " << i << " com a região regionsPerLine(" << regionsPerLine << ")" << endl;
			cout << "A comparar a região " << i << " com a região regionsPerLine+1(" << (regionsPerLine + 1) << ")" << endl;

			localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i));
			localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i - 1));
			localCorrelation(subRegionsOfA.at(i),
					subRegionsOfB.at(regionsPerLine));
			localCorrelation(subRegionsOfA.at(i),
					subRegionsOfB.at(regionsPerLine + 1));
		}
		waitKey(0);
		count++;
	}

	return SumAB;
}

int main() {

	clock_t time = clock();

	if (!stasm_init("data", 0 /*trace*/))
		error("stasm_init failed: ", stasm_lasterr());

	static const char* path = "2013-11-18-173422.jpg";

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

		// roll
		double theta = atan2((double) LPupil.y - RPupil.y, LPupil.x - RPupil.x); //deg = * 180 / CV_PI;
		printf("theta = %f degrees\n", theta);
		double roll = min(abs((2 * theta) / CV_PI), 1.0); // rad
		printf("roll = %f\n", roll);

		// yaw ()
		// cálculo do dl e dr (http://answers.opencv.org/question/14188/calc-eucliadian-distance-between-two-single-point/)
		double dl = cv::norm(LPupil - CNoseTip);
		double dr = cv::norm(RPupil - CNoseTip);
		double yaw = (max(dl, dr) - min(dl, dr)) / max(dl, dr);
		printf("yaw = %f\n", yaw);

		// pitch
		double eu = cv::norm(LEyebrowInner - CNoseTip);
		double ed = cv::norm(CNoseBase - CTipOfChin);
		double pitch = (max(eu, ed) - min(eu, ed)) / max(eu, ed);
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
		Point p1, p2, p3, p4, p5, p6, p7, p8;

		// Finding the 8 points
		try {
			p1 = midpoint((double) cvRound(landmarks[0 * 2]),
					(double) cvRound(landmarks[0 * 2 + 1]),
					(double) cvRound(landmarks[58 * 2]),
					(double) cvRound(landmarks[58 * 2 + 1]));
			p2 = midpoint((double) cvRound(landmarks[3 * 2]),
					(double) cvRound(landmarks[3 * 2 + 1]),
					(double) cvRound(landmarks[58 * 2]),
					(double) cvRound(landmarks[58 * 2 + 1]));
			p3 = LEyebrowInner;
			p4 = midpoint((double) LEyebrowInner.x, (double) LEyebrowInner.y,
					(double) CNoseTip.x, (double) CNoseTip.y);
			p5 = CNoseTip;
			p6 = CTipOfChin;
			p7 = midpoint((double) cvRound(landmarks[54 * 2]),
					(double) cvRound(landmarks[54 * 2 + 1]),
					(double) cvRound(landmarks[12 * 2]),
					(double) cvRound(landmarks[12 * 2 + 1]));
			p8 = midpoint((double) cvRound(landmarks[54 * 2]),
					(double) cvRound(landmarks[54 * 2 + 1]),
					(double) cvRound(landmarks[9 * 2]),
					(double) cvRound(landmarks[9 * 2 + 1]));
		} catch (Exception & e) {
			cout << e.msg << endl;
			break;
		}

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

		double mc[8] =
				{ mc_w1, mc_w2, mc_w3, mc_w4, mc_w5, mc_w6, mc_w7, mc_w8 };
		printf("The dataset is %g, %g, %g, %g, %g, %g, %g, %g\n", mc[0], mc[1],
				mc[2], mc[3], mc[4], mc[5], mc[6], mc[7]);

		double std = calculateStd(mc);
		printf("Std deviation = %f: \n", std);

		double si = 1 - sigmoid(std);

		printf("Sigmoid = %f: \n", sigmoid(std));

		printf("SI = %f: \n", si);

		imshow("Original img", img);

		// C. POSE NORMALIZATION ###########################################

		// 4.(a) rotation
		double theta_deg = theta * 180 / CV_PI;
		img = rotateImage(img, -180 + theta_deg);
//		cv::imshow("rotatedImg", img);
//		cv::waitKey(0);

		// 4.(b) horizontal flip if dr smaller than dl
		if (gsl_fcmp(dl, dr, DBL_EPSILON) > 0) { // x = y returns 0; if x < y returns -1; x > y returns +1;
			flip(img, img, 1);
		}

		std::vector<Point> roi_vector = getNewStasmPts(img, 68);

		// image crop for better results
		int x1, y1, x2, y2;
		try {
			x1 = roi_vector.at(1).x - 5;
			y1 = roi_vector.at(23).y - 40;
			x2 = roi_vector.at(13).x + 5;
			y2 = roi_vector.at(7).y + 5;
		} catch (const std::out_of_range& oor) {
			std::cerr << "Unable to crop image! Reason: Out of Range error: "
					<< oor.what() << '\n';
			break;
		}
		int width = x2 - x1;
		int height = y2 - y1;
		Mat crop = img(Rect(x1, y1, width, height));

		// 4. (c) stretching
		std::vector<Point> stasm_vector = getNewStasmPts(crop, 68);

		Point noseTop = midpoint(stasm_vector.at(24).x, stasm_vector.at(24).y,
				stasm_vector.at(18).x, stasm_vector.at(18).y);
		Point topCenter = Point(noseTop.x, 0);
		Point noseTip = stasm_vector.at(67);
		Point noseBase = stasm_vector.at(41);
		Point lipTop = stasm_vector.at(51);
		Point lipBottom = stasm_vector.at(57);
		Point chinTip = stasm_vector.at(7);
		Point bottomCenter = Point(chinTip.x, crop.rows);
//		Point bottomRight = Point(crop.cols, crop.rows);
//		Point topRight = Point(crop.cols, 0);
//
//		int thickness = -1;
//		int lineType = 8;

//		circle(crop, topCenter, 2, Scalar(0, 255, 255), thickness, lineType);
//		circle(crop, noseTop, 2, Scalar(0, 0, 255), thickness, lineType);
//		circle(crop, noseTip, 2, Scalar(0, 255, 255), thickness, lineType);
//		circle(crop, noseBase, 2, Scalar(0, 0, 255), thickness, lineType);
//		circle(crop, lipTop, 2, Scalar(0, 255, 255), thickness, lineType);
//		circle(crop, lipBottom, 2, Scalar(0, 255, 255), thickness, lineType);
//		circle(crop, chinTip, 2, Scalar(0, 255, 255), thickness, lineType);
//		circle(crop, bottomCenter, 2, Scalar(0, 0, 255), thickness, lineType);
//		circle(crop, bottomRight, 2, Scalar(0, 255, 255), thickness, lineType);
//		circle(crop, topRight, 2, Scalar(0, 255, 255), thickness, lineType);
//
//		line(crop, topCenter, noseTop, 255, 1, 8, 0);
//		line(crop, noseTop, noseTip, 255, 1, 8, 0);
//		line(crop, noseTip, noseBase, 255, 1, 8, 0);
//		line(crop, noseBase, lipTop, 255, 1, 8, 0);
//		line(crop, lipTop, lipBottom, 255, 1, 8, 0);
//		line(crop, lipBottom, chinTip, 255, 1, 8, 0);
//		line(crop, chinTip, bottomCenter, 255, 1, 8, 0);
//		line(crop, bottomCenter, bottomRight, 255, 1, 8, 0);
//		line(crop, bottomRight, topRight, 255, 1, 8, 0);
//		line(crop, topRight, topCenter, 255, 1, 8, 0);
//
//		line(crop, Point(0, noseTop.y), Point(crop.cols, noseTop.y), 255, 1, 8, 0);
//		line(crop, Point(0, noseTip.y), Point(crop.cols, noseTip.y), 255, 1, 8, 0);
//		line(crop, Point(0, noseBase.y), Point(crop.cols, noseBase.y), 255, 1, 8, 0);
//		line(crop, Point(0, lipTop.y), Point(crop.cols, lipTop.y), 255, 1, 8, 0);
//		line(crop, Point(0, lipBottom.y), Point(crop.cols, lipBottom.y), 255, 1, 8, 0);
//		line(crop, Point(0, chinTip.y), Point(crop.cols, chinTip.y), 255, 1, 8, 0);

		imshow("roi - lines", crop);

		cv::Mat out(crop.rows, crop.cols, CV_8U);
		cv::Mat out2(crop.rows, crop.cols, CV_8U);

		Mat band1 = correctPerpective(crop, topCenter, noseTop,
				Point(crop.cols, noseTop.y));
		Mat band2 = correctPerpective(crop, noseTop, noseTip,
				Point(crop.cols - (abs(noseTop.x - noseTip.x)), noseTip.y));
		Mat band3 = correctPerpective(crop, noseTip, noseBase,
				Point(crop.cols - (abs(noseTip.x - noseBase.x)), noseBase.y));
		Mat band4 = correctPerpective(crop, noseBase, lipTop,
				Point(crop.cols - (abs(noseBase.x - lipTop.x) + 2), lipTop.y));
		Mat band5 = correctPerpective(crop, lipTop, lipBottom,
				Point(crop.cols - (abs(lipTop.x - lipBottom.x)), lipBottom.y));
		Mat band6 = correctPerpective(crop, lipBottom, chinTip,
				Point(crop.cols - (abs(lipBottom.x - chinTip.x)), chinTip.y));
		Mat band7 = correctPerpective(crop, chinTip, bottomCenter,
				Point(crop.cols - (abs(chinTip.x - bottomCenter.x)),
						bottomCenter.y));

		for (int r = 0; r < crop.rows; r++) {
			if (r < noseTop.y) {
				band1.row(r).copyTo(out.row(r));
			} else if (r < noseTip.y) {
				band2.row(r).copyTo(out.row(r));
			} else if (r < noseBase.y) {
				band3.row(r).copyTo(out.row(r));
			} else if (r < lipTop.y) {
				band4.row(r).copyTo(out.row(r));
			} else if (r < lipBottom.y) {
				band5.row(r).copyTo(out.row(r));
			} else if (r < chinTip.y) {
				band6.row(r).copyTo(out.row(r));
			} else {
				band7.row(r).copyTo(out.row(r));
			}
		}

		imshow("rows stretched equally", out);

		for (int row = 0; row < crop.rows; row++) {
			for (int col = 0; col < crop.cols; col++) {
				if (col < crop.cols * 0.5)
					out2.at<uchar>(row, col) = out.at<uchar>(row + 1, -col);
				else
					out2.at<uchar>(row, col) = out.at<uchar>(row, col - 1);
			}
		}

		std::vector<Point> stasm_vector2 = getNewStasmPts(out2, 68);

		imshow("Mirror right to left", out2);

		// 4.(f) função sqi
		Mat illumNorn;

		IplImage copy = out2;

		illumNorn = Mat(SQI(&copy));

		localCorrelation(out2, out2);

		divideIntoSubRegions(out2);

		globalCorrelation(out2, out2);

		imshow("illumination norm with SQI", illumNorn);

		nfaces++;
	}

	time = clock() - time;

	int ms = double(time) / CLOCKS_PER_SEC * 1000;

	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	printf("%s: %d face(s)\n", path, nfaces);
	fflush(stdout);
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
//	IplImage *ttt;
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
