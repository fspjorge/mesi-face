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
	static const char* imgPath = "2014-01-29-222427.jpg";
	Face face = Face(imgPath);
	Mat img1 = face.loadMat();
	vector<Point> stasmPtsVector = face.getStasmPts();
	vector<string> filenames;
	map<double, string> correlationsMap;

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

	// 4.F) Illumination Normalization
//	img1 = face.normalizeIllumination(img1);

	/**
	 * Iterate trough all the gallery templates.
	 */
	namespace fs = boost::filesystem;
	fs::path someDir("/home/jorge/workspace/dissertacao/templates");
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
	double dmax = 0;

	// Gallery comparison
	for (unsigned f = 0; f < filenames.size(); f++) {

		string root = "/home/jorge/workspace/dissertacao/templates/";
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

		// 4.F) Illumination Normalization
//		img2 = face.normalizeIllumination(img2);

		double globalCorrelation = face.computeGlobalCorrelation(img1, img2);
		cout << "global correlation = " << globalCorrelation << endl;

		if((1-globalCorrelation) > dmax)
			dmax = globalCorrelation;

		correlationsMap.insert(make_pair(globalCorrelation, filenames.at(f)));
	}

	/*
	 * The list of computed values is organized in decreasing order.
	 * Given the mean number n of templates per identity which are contained in the gallery,
	 * the identity Ij with the maximum number of images in the first
	 * n positions is returned. As a matter of fact, in an ideal situation,
	 * all of the templates for the correct identity should appear in the
	 * first positions of the ordered list.
	 */
	double dg1 = 0;
	double dg2 = 0;
	double nb = 0;
	double dgG = 0;
	string sub_str;
	size_t index;
	string identityStr;

	for (map<double, string>::reverse_iterator iter =
			correlationsMap.rbegin(); iter != correlationsMap.rend();
			++iter) {
		cout << iter->first << ": ";
		cout << iter->second << endl;

		// compute the distance between p and gi1 (dg1).
		dg1 = (dg1 == 0) ? face.computeQls(iter->first, dmax) : dg1;

		// compute the distance between p and gi2 (dg2).
		string str = iter->second;
		index = str.find("_");
		sub_str = str.substr (0, index);

		if(identityStr.empty())
		{
			identityStr = sub_str;
		}
		else if(sub_str.compare(identityStr) != 0 && dg2 == 0)
		{
			dg2 = face.computeQls(iter->first, dmax);
		}
		else // compute the distance between p and gi|G| (dgG).
		{
			dgG = face.computeQls(iter->first, dmax);
		}

		/**
		 * The second function φ2 (p) (density ratio) is computed using
		 * the ratio between the number of subjects in the gallery, distinct
		 * from the returned identity, giving a distance lower than twice
		 * F(d(p, gi 1)), and the cardinality G of the gallery.
		 */
		if(face.computeQls(iter->first, dmax) < (2 * dg1))
		{
			nb++;// Nb = {gi k ∈ G|F (d(p, gi k )) < 2F (d(p, gi 1 ))}.
		}
	}

	cout << "dg1 = " << dg1 << endl;
	cout << "dg2 = " << dg2 << endl;
	cout << "dgG = " << dgG << endl;
	cout << "dmax = " << dmax << endl;

	double phi1 = (dg2 - dg1) / dgG;
	double phi2 = 1 - (nb / correlationsMap.size());

	/**
	 * Width of the subinterval from φj to the proper extreme of
	 * the overall [0, 1),interval of possible values, depending on the
	 * comparison between the current φj (p) and φj .
	 */
	double phij_ = 0.1;
	double s1 = (phi1 > phij_) ? 1 - phij_ : phij_;
	double s2 = (phi2 > phij_) ? 1 - phij_ : phij_;

	/*
	 * SRR I corresponds to the genuine element.
	 * SRR I is computed starting from the relative distance (computed starting
	 * from the distance between the scores of the first two retrieved
	 * distinct identities)
	 */
	double srr1 = abs(phi1 - phij_) / s1;

	/*
	 * SRR II corresponds to the element closest to the genuine one.
	 * SRR II uses the density ratio (relative amount of gallery templates which are “near” to the
	 * retrieved identity although belonging to different identities).
	 */
	double srr2 = abs(phi2 - phij_) / s2;

	cout << "dg1 = " << dg1 << endl;
	cout << "dg2 = " << dg2 << endl;
	cout << "dgG = " << dgG << endl;
	cout << "nb = " << nb << endl;
	cout << "phi1 = " << phi1 << endl;
	cout << "phi2 = " << phi2 << endl;
	cout << "srr1 = " << srr1 << endl;
	cout << "srr2 = " << srr2 << endl;


	time = clock() - time;
	int ms = double(time) / CLOCKS_PER_SEC * 1000;
	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
