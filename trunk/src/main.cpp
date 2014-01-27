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
	static const char* imgPath = "templates_jorge/4_5.pgm";
	Face face = Face(imgPath);
	Mat img1 = face.loadMat();
	vector<Point> stasmPtsVector = face.getStasmPts();
	vector<string> filenames;
	map<double, string> correlations_map;

	Point lPupil = stasmPtsVector.at(31);
	Point rPupil = stasmPtsVector.at(36);
	Point noseTip = stasmPtsVector.at(67);
	Point lEyebrowInner = stasmPtsVector.at(24);
	Point noseBase = stasmPtsVector.at(41);
	Point tipOfChin = stasmPtsVector.at(7);

	cout << "Candidate | ";

	// Compute SP and SI
	double sp = face.computeSp(lPupil, rPupil, lEyebrowInner, noseTip, noseBase,
			tipOfChin);
	cout << "SP = " << sp << " | ";
	double si = face.computeSi(lPupil, rPupil, lEyebrowInner, noseTip, noseBase,
			tipOfChin);
	cout << "SI = " << si << " | ";
	cout << imgPath << endl
			<< "-----------------------------------------------------------------------------------------------------------------"
			<< endl;

	// 4. A), B), C), D) and E) Pose Normalization
	img1 = face.normalizePose(img1, lPupil, rPupil, lEyebrowInner, noseTip,
			noseBase, tipOfChin);

//	imshow("img1_pose", img1);

// 4.F) Illumination Normalization
	img1 = face.normalizeIllumination(img1);

//	imshow("img1_ill", img1);

//percorrer todas as imagens de uma pasta

	namespace fs = boost::filesystem;
	fs::path someDir("/home/jorge/workspace/dissertacao/templates_jorge");
	fs::directory_iterator end_iter;

	typedef std::multimap<std::time_t, fs::path> result_set_t;
	result_set_t result_set;

	int count = 2;
	if (fs::exists(someDir) && fs::is_directory(someDir)) {
		for (fs::directory_iterator dir_iter(someDir); dir_iter != end_iter;
				++dir_iter) {
			if (fs::is_regular_file(dir_iter->status())) {
				if (dir_iter->path().filename().extension() == ".jpg"
						|| dir_iter->path().filename().extension() == ".pgm") {

					filenames.push_back(dir_iter->path().filename().string());
				}
				count++;
			}
		}
	}

	std::sort(filenames.begin(), filenames.end());

	for (unsigned f = 0; f < filenames.size(); f++) {

		// FACE 2
		string root = "/home/jorge/workspace/dissertacao/templates_jorge/";
		string imgPath2 = filenames.at(f).c_str();
		string absPath = root + imgPath2;
		face = Face(absPath.c_str());
		Mat img2 = face.loadMat();
		vector<Point> stasmPtsVector2 = face.getStasmPts();

		Point lPupil2 = stasmPtsVector2.at(31);
		Point rPupil2 = stasmPtsVector2.at(36);
		Point noseTip2 = stasmPtsVector2.at(67);
		Point lEyebrowInner2 = stasmPtsVector2.at(24);
		Point noseBase2 = stasmPtsVector2.at(41);
		Point tipOfChin2 = stasmPtsVector2.at(7);

		cout << filenames.at(f) << " | ";

		// Compute SP and SI
		double sp2 = face.computeSp(lPupil2, rPupil2, lEyebrowInner2, noseTip2,
				noseBase2, tipOfChin2);
		cout << "SP = " << sp2 << " | ";
		double si2 = face.computeSi(lPupil2, rPupil2, lEyebrowInner2, noseTip2,
				noseBase2, tipOfChin2);
		cout << "SI = " << si2 << " | ";

		// 4. A), B), C), D) and E) Pose Normalization
		img2 = face.normalizePose(img2, lPupil2, rPupil2, lEyebrowInner2,
				noseTip2, noseBase2, tipOfChin2);

//		imshow("img2_pose", img2);

// 		4.F) Illumination Normalization
		img2 = face.normalizeIllumination(img2);
//		imshow("img2_illim", img2);

//		double localCorrelation = face.computeLocalCorrelation(img1, img2);
		double globalCorrelation = face.computeGlobalCorrelation(img1, img2);

		correlations_map.insert(make_pair(globalCorrelation, filenames.at(f)));

		// COMPARAÇÃO DAS DUAS IMAGENS  cout << "localCorr = " << face.computelocalCorrelation(img, img2) << endl;
		cout << "globalCorrelation = " << globalCorrelation << endl;
//		cout << "localCorrelation = " << localCorrelation << endl;
	}

	/*
	 * The list of computed values is organized in decreasing order.
	 * Given the mean number n of templates per identity which are contained in the gallery,
	 * the identity Ij with the maximum number of images in the first
	 * n positions is returned. As a matter of fact, in an ideal situation,
	 * all of the templates for the correct identity should appear in the
	 * first positions of the ordered list.
	 */
	for (std::map<double, std::string>::reverse_iterator iter =
			correlations_map.rbegin(); iter != correlations_map.rend();
			++iter) {
		cout << iter->first << ": ";
		cout << iter->second << endl;
	}

	time = clock() - time;
	int ms = double(time) / CLOCKS_PER_SEC * 1000;
	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
