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
//adding -lboost_filesystem -lboost_system to compiler
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"

int main() {

	clock_t time = clock();

	//FACE 1
	static const char* imgPath = "img/2.jpg";
	Face face = Face(imgPath);
	Mat img = face.loadMat();
	vector<Point> stasmPtsVector = face.getStasmPts();

	Point lPupil = stasmPtsVector.at(31);
	Point rPupil = stasmPtsVector.at(36);
	Point noseTip = stasmPtsVector.at(67);
	Point lEyebrowInner = stasmPtsVector.at(24);
	Point noseBase = stasmPtsVector.at(41);
	Point tipOfChin = stasmPtsVector.at(7);

	cout << "Image 1 #############################" << endl;

	// Compute SP and SI
	double sp = face.computeSp(lPupil, rPupil, lEyebrowInner, noseTip, noseBase,
			tipOfChin);
	cout << "SP1 = " << sp << endl;
	double si = face.computeSi(lPupil, rPupil, lEyebrowInner, noseTip, noseBase,
			tipOfChin);
	cout << "SI1 = " << si << endl;

	// 4. A), B), C), D) and E) Pose Normalization
	img = face.normalizePose(img, lPupil, rPupil, lEyebrowInner, noseTip,
			noseBase, tipOfChin);

	// 4.F) Illumination Normalization
//	img = face.normalizeIllumination(img);

	imshow("illumination norm with SQI", img);

	waitKey(0);

	//percorrer todas as imagens de uma pasta

	namespace fs = boost::filesystem;
	fs::path someDir("/home/jorge/workspace/dissertacao/");
	fs::directory_iterator end_iter;

	typedef std::multimap<std::time_t, fs::path> result_set_t;
	result_set_t result_set;

	if (fs::exists(someDir) && fs::is_directory(someDir)) {
		for (fs::directory_iterator dir_iter(someDir); dir_iter != end_iter;
				++dir_iter) {
			if (fs::is_regular_file(dir_iter->status())) {
				cout << dir_iter->path().filename().extension() << endl;
				if (dir_iter->path().filename().extension() == ".jpg") {
					cout << "Image 2 #############################" << endl;
					cout << dir_iter->path().filename() << endl;
					// FACE 2
					static const char* imgPath2 =
							dir_iter->path().filename().string().c_str();
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
					double sp2 = face2.computeSp(lPupil2, rPupil2,
							lEyebrowInner2, noseTip2, noseBase2, tipOfChin2);
					cout << "SP2 = " << sp2 << endl;
					double si2 = face2.computeSi(lPupil2, rPupil2,
							lEyebrowInner2, noseTip2, noseBase2, tipOfChin2);
					cout << "SI2 = " << si2 << endl;

					// 4. A), B), C), D) and E) Pose Normalization
					img2 = face2.normalizePose(img2, lPupil2, rPupil2,
							lEyebrowInner2, noseTip2, noseBase2, tipOfChin2);

					// 4.F) Illumination Normalization
//				img2 = face2.normalizeIllumination(img2);

					imshow("illumination norm with SQI2", img2);

					double corr = face.computeLocalCorrelation(img, img2);

					// COMPARAÇÃO DAS DUAS IMAGENS	cout << "localCorr = " << face.computelocalCorrelation(img, img2) << endl;
					cout << "globalCorr = " << corr << endl;

					if (corr < 0.4) {
						cout << "Authentication rejected!" << endl;
					} else {
						cout << "Authentication accepted!" << endl;
					}
				}
			}
		}
	}

	time = clock() - time;
	int ms = double(time) / CLOCKS_PER_SEC * 1000;
	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
