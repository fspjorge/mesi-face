/**
 * Author: Jorge Pereira
 * MESI-IPBEJA
 * http://geosoft.no/development/cppstyle.html#Layout of the Recommendations
 * http://stackoverflow.com/questions/5945555/examples-of-how-to-write-beautiful-c-comments
 */

#include "Face.h"

int main() {

	clock_t time = clock();
	static const char* imgPath = "scarlett_johansson_eyes_smile.jpg";
	Face face = Face(imgPath);
	Mat img = face.loadMat();
	vector<Point> stasmPtsVector = face.getStasmPts();

	Point lPupil = stasmPtsVector.at(31);
	Point rPupil = stasmPtsVector.at(36);
	Point noseTip = stasmPtsVector.at(67);
	Point lEyebrowInner = stasmPtsVector.at(24);
	Point noseBase = stasmPtsVector.at(41);
	Point tipOfChin = stasmPtsVector.at(7);

	double sp1 = face.computeSp(lPupil, rPupil, lEyebrowInner, noseTip, noseBase, tipOfChin);
	cout << "SP = " << sp1 << endl;
	double si1 = face.computeSi(lPupil, rPupil, lEyebrowInner, noseTip, noseBase, tipOfChin);
	cout << "SI = " << si1 << endl;

	// 4.C) POSE NORMALIZATION ###########################################
	Mat crop = face.normalizePose(img, lPupil, rPupil, lEyebrowInner, noseTip, noseBase, tipOfChin);

	// 4.F) ILLUMINATION NORMALIZATION (SQI) ###########################################
	IplImage copy = crop;
	Mat sqiImg = Mat(face.SQI(&copy));
	
	// COMPARAÇÃO DAS DUAS IMAGENS
	cout << "globalCorr = " << face.computeGlobalCorr(sqiImg, sqiImg) << endl;
	imshow("illumination norm with SQI", sqiImg);


	time = clock() - time;
	int ms = double(time) / CLOCKS_PER_SEC * 1000;
	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
