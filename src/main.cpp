/**
 * Author: Jorge Pereira
 * MESI-IPBEJA
 */

#include "Face.h"

#define C_TEXT( text ) ((char*)std::string( text ).c_str())

using namespace cv;
using namespace std;

int main() {

	clock_t time = clock();
	Face face = Face("1530524_702769799755645_598776919_n.jpg");
	Mat img = face.loadMat();
	vector<Point> stasmPts = face.getStasmPts();

	Point LPupil = stasmPts.at(31);
	Point RPupil = stasmPts.at(36);
	Point CNoseTip = stasmPts.at(67);
	Point LEyebrowInner = stasmPts.at(24);
	Point CNoseBase = stasmPts.at(41);
	Point CTipOfChin = stasmPts.at(7);

	double sp = face.calcSp(LPupil, RPupil, LEyebrowInner, CNoseBase, CNoseBase, CTipOfChin);

	// SI = 1 - F(std(mc))
	Point p1, p2, p3, p4, p5, p6, p7, p8;

	// Finding the 8 points
	try {
		p1 = face.calcMidpoint(stasmPts.at(0).x, stasmPts.at(0).y, stasmPts.at(39).x, stasmPts.at(39).y);
		p2 = face.calcMidpoint(stasmPts.at(3).x, stasmPts.at(3).y, stasmPts.at(39).x, stasmPts.at(39).y);
		p3 = LEyebrowInner;
		p4 = face.calcMidpoint((double) LEyebrowInner.x, (double) LEyebrowInner.y,
				(double) CNoseTip.x, (double) CNoseTip.y);
		p5 = CNoseTip;
		p6 = CTipOfChin;
		p7 = face.calcMidpoint(stasmPts.at(14).x, stasmPts.at(14).y, stasmPts.at(43).x, stasmPts.at(43).y);
		p8 = face.calcMidpoint(stasmPts.at(10).x, stasmPts.at(10).y, stasmPts.at(43).x, stasmPts.at(43).y);

	} catch (Exception & e) {
		cout << e.msg << endl;
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
	double mc_w1 = (double) face.getMassCenter("h1", subMatPt1);
	double mc_w2 = (double) face.getMassCenter("h2", subMatPt2);
	double mc_w3 = (double) face.getMassCenter("h3", subMatPt3);
	double mc_w4 = (double) face.getMassCenter("h4", subMatPt4);
	double mc_w5 = (double) face.getMassCenter("h5", subMatPt5);
	double mc_w6 = (double) face.getMassCenter("h6", subMatPt6);
	double mc_w7 = (double) face.getMassCenter("h7", subMatPt7);
	double mc_w8 = (double) face.getMassCenter("h8", subMatPt8);

	double mc[8] =
			{ mc_w1, mc_w2, mc_w3, mc_w4, mc_w5, mc_w6, mc_w7, mc_w8 };
	printf("The dataset is %g, %g, %g, %g, %g, %g, %g, %g\n", mc[0], mc[1],
			mc[2], mc[3], mc[4], mc[5], mc[6], mc[7]);

	double std = face.calculateStd(mc);
	printf("Std deviation = %f: \n", std);

	double si = 1 - face.sigmoid(std);

	printf("Sigmoid = %f: \n", face.sigmoid(std));

	printf("SI = %f: \n", si);

	// C. POSE NORMALIZATION ###########################################

	Mat crop = face.normalizePose(img, LPupil, RPupil, LEyebrowInner, CNoseTip, CNoseBase, CTipOfChin);

	imshow("crop da main", crop);

	// 4.(f) função sqi
//	Mat illumNorn;
//
//	IplImage copy = out2;
//
//	illumNorn = Mat(face.SQI(&copy));
//	cout << "globalCorr = " << face.globalCorrelation(out2, out2) << endl;

//	imshow("illumination norm with SQI", illumNorn);

	time = clock() - time;

	int ms = double(time) / CLOCKS_PER_SEC * 1000;

	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
