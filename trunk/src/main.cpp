/**
 * Author: Jorge Pereira
 * MESI-IPBEJA
 */

#include "Face.h"

int main() {

	clock_t time = clock();
	static const char* imgPath = "2013-12-24-163337.jpg";
	Face face = Face(imgPath);
	Mat img = face.loadMat();
	vector<Point> stasmPts = face.getStasmPts();

	Point LPupil = stasmPts.at(31);
	Point RPupil = stasmPts.at(36);
	Point CNoseTip = stasmPts.at(67);
	Point LEyebrowInner = stasmPts.at(24);
	Point CNoseBase = stasmPts.at(41);
	Point CTipOfChin = stasmPts.at(7);

	double sp = face.calcSp(LPupil, RPupil, LEyebrowInner, CNoseBase, CNoseBase, CTipOfChin);
	double si = face.calcSi(LPupil, RPupil, LEyebrowInner, CNoseBase, CNoseBase, CTipOfChin);

	// C. POSE NORMALIZATION ###########################################
	Mat crop = face.normalizePose(img, LPupil, RPupil, LEyebrowInner, CNoseTip, CNoseBase, CTipOfChin);

	// F. ILLUMINATION NORMALIZATION (SQI) ###########################################
	IplImage copy = crop;
	Mat illumNorn = Mat(face.SQI(&copy));

	cout << "globalCorr = " << face.globalCorrelation(crop, crop) << endl;
	imshow("illumination norm with SQI", illumNorn);

	time = clock() - time;
	int ms = double(time) / CLOCKS_PER_SEC * 1000;
	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
