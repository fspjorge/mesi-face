/**
 *   Created on: Dec 23, 2013
 *       Author: Jorge Silva Pereira
 * Organization: ESTIG-IPBEJA
 *
 * http://geosoft.no/development/cppstyle.html#Layout of the Recommendations
 * http://stackoverflow.com/questions/5945555/examples-of-how-to-write-beautiful-c-comments
 * http://vis-www.cs.umass.edu/lfw/sets_1.html // comparação de faces
 */

#include "Face.h"

int main() {

	clock_t time = clock();

	//FACE 1
	static const char* imgPath = "2013-11-23-030207.jpg";
	Face face = Face(imgPath);
	Mat img = face.loadMat();
	vector<Point> stasmPtsVector = face.getStasmPts();

	Point lPupil = stasmPtsVector.at(31);
	Point rPupil = stasmPtsVector.at(36);
	Point noseTip = stasmPtsVector.at(67);
	Point lEyebrowInner = stasmPtsVector.at(24);
	Point noseBase = stasmPtsVector.at(41);
	Point tipOfChin = stasmPtsVector.at(7);

	// Compute SP and SI
	double sp = face.computeSp(lPupil, rPupil, lEyebrowInner, noseTip, noseBase, tipOfChin);
	cout << "SP1 = " << sp << endl;
	double si = face.computeSi(lPupil, rPupil, lEyebrowInner, noseTip, noseBase, tipOfChin);
	cout << "SI1 = " << si << endl;

	// 4. A), B), C), D) and E) Pose Normalization
	img = face.normalizePose(img, lPupil, rPupil, lEyebrowInner, noseTip, noseBase, tipOfChin);
	imshow("before illumination norm with SQI", img);
	// 4.F) Illumination Normalization
	img = face.normalizeIllumination(img);

	imshow("illumination norm with SQI", img);


	// FACE 2
	static const char* imgPath2 = "2013-11-18-172954.jpg";
	Face face2 = Face(imgPath2);
	Mat img2 = face2.loadMat();
	vector<Point> stasmPtsVector2 = face2.getStasmPts();

	Point lPupil2 = stasmPtsVector2.at(31);
	Point rPupil2 = stasmPtsVector2.at(36);
	Point noseTip2 = stasmPtsVector2.at(67);
	Point lEyebrowInner2 = stasmPtsVector2.at(24);
	Point noseBase2 = stasmPtsVector2.at(41);
	Point tipOfChin2 = stasmPtsVector2.at(7);

	// Compute SP and SI
	double sp2 = face2.computeSp(lPupil2, rPupil2, lEyebrowInner2, noseTip2, noseBase2, tipOfChin2);
	cout << "SP2 = " << sp2 << endl;
	double si2 = face2.computeSi(lPupil2, rPupil2, lEyebrowInner2, noseTip2, noseBase2, tipOfChin2);
	cout << "SI2 = " << si2 << endl;

	// 4. A), B), C), D) and E) Pose Normalization
	img2 = face2.normalizePose(img2, lPupil2, rPupil2, lEyebrowInner2, noseTip2, noseBase2, tipOfChin2);

	// 4.F) Illumination Normalization
	img2 = face2.normalizeIllumination(img2);

	imshow("illumination norm with SQI2", img2);
	
	// COMPARAÇÃO DAS DUAS IMAGENS	cout << "localCorr = " << face.computelocalCorrelation(img, img2) << endl;
	cout << "globalCorr = " << face.computeGlobalCorrelation(img, img2) << endl;

	time = clock() - time;
	int ms = double(time) / CLOCKS_PER_SEC * 1000;
	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
