/*
 * face.cpp
 *
 *  Created on: Dec 23, 2013
 *      Author: Jorge Silva Pereira
 */

#ifndef FACE_CPP_
#define FACE_CPP_

#include "Face.h"

Face::Face() {}

void Face::init(const char* imgPath) {

	if (!stasm_init("data", 0 /*trace*/))
		error("stasm_init failed: ", stasm_lasterr());

	cv::Mat_<unsigned char> img(cv::imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE));
	Rect rect(cv::Point(), img.size());

	if (!img.data)
		error("Cannot load", imgPath);

	if (!stasm_open_image((const char*) img.data, img.cols, img.rows, imgPath,
			1 /*multiface*/, 10 /*minwidth*/))
		error("stasm_open_image failed: ", stasm_lasterr());

	int foundface;
	float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

	if (!stasm_search_single(&foundface, landmarks, (char*) img.data, img.cols,
			img.rows, imgPath, "../data")) {
		printf("Error in stasm_search_single: %s\n", stasm_lasterr());
		exit(1);
	}

	if (!foundface) {
		printf("No face found in %s, bye...\n", imgPath);
		exit(1);
	} else {
		// draw the landmarks on the image as white dots
		stasm_force_points_into_image(landmarks, img.cols, img.rows);
		stasm_convert_shape(landmarks, 68);
		for (int i = 0; i < 68; i++) {
			if (rect.contains(
					Point(cvRound(landmarks[i * 2]),
							cvRound(landmarks[i * 2 + 1])))) {
				stasmPts.push_back(
						Point(cvRound(landmarks[i * 2]),
								cvRound(landmarks[i * 2 + 1])));
			} else {
				printf("Some point is out of the image, bye...\n");
				exit(1);
			}
		}
	}
	face = img;
}

/**
 * Illumination correction using the SQI algorithm.
 */
Mat Face::normalizeIllumination(Mat face) {
	IplImage copy = face;
	Mat sqiImg = Mat(SQI(&copy));

	return sqiImg;
}

/**
 * Get STASM points array from image.
 */
vector<Point> Face::getStasmPts() {
	return stasmPts;
}

/**
 * Pose correction:
 * 1) The center of the eyes is used to correct head rolling [Fig. 4(a)].
 * 2) The distances between the external corners of the left and right eyes and the tip of the nose, represented respectively by dl and dr [Fig. 4(b)].
 * 	  If this is the right half (dr ≥ dl), the image is left unchanged; otherwise, it is reflected with respect to the vertical axis (horizontal flip).
 * 3)
 */
Mat Face::normalizePose(Mat face, Point lPupil, Point rPupil,
		Point lEyebrowInner, Point noseTip, Point noseBase, Point tipOfChin) {

	double theta = computePointsAngle(lPupil, rPupil);
	face = rotateImage(face, theta);

	for (unsigned int i = 0; i < 68; i++) {
		stasmPts.at(i) = rotatePoint(getStasmPts().at(i), -theta);
	}

	lPupil = getStasmPts().at(31);
	rPupil = getStasmPts().at(36);
	noseTip = getStasmPts().at(67);
	lEyebrowInner = getStasmPts().at(24);
	noseBase = getStasmPts().at(41);
	tipOfChin = getStasmPts().at(7);

	double dl = cv::norm(lPupil - noseTip);
	double dr = cv::norm(rPupil - noseTip);

	int x1, y1, x2, y2;

	Mat crop;

	try {

		if (gsl_fcmp(dl, dr, DBL_EPSILON) > 0) { // x = y returns 0; if x < y returns -1; x > y returns +1;
			flip(face, face, 1);
			for (unsigned int i = 0; i < 68; i++) {
				stasmPts.at(i) = Point(face.cols - stasmPts.at(i).x,
						stasmPts.at(i).y);
			}

			x1 = stasmPts.at(13).x;
			x2 = stasmPts.at(1).x;
		} else {
			x1 = stasmPts.at(1).x;
			x2 = stasmPts.at(13).x;
		}

		y1 = stasmPts.at(23).y - 40;
		y2 = stasmPts.at(7).y + 5;

		int width = abs(x1 - x2);
		int height = abs(y2 - y1);

		crop = face(Rect(x1, y1, width, height));
	} catch (exception& e) {
		cout << "Unable to crop face - Exception: " << e.what() << endl;
	}

	noseTip = Point(stasmPts.at(67).x - x1, stasmPts.at(67).y - y1); //82,121
	Point noseTop = calcMidpoint(stasmPts.at(24).x - x1, stasmPts.at(24).y - y1,
			stasmPts.at(18).x - x1, stasmPts.at(18).y - y1); // 88,45
	Point topCenter = Point(noseTop.x, 0); //
	noseBase = Point(stasmPts.at(41).x - x1, stasmPts.at(41).y - y1); //OK
	Point lipTop = Point(stasmPts.at(51).x - x1, stasmPts.at(51).y - y1);
	Point lipBottom = Point(stasmPts.at(57).x - x1, stasmPts.at(57).y - y1);
	Point chinTip = Point(stasmPts.at(7).x - x1, stasmPts.at(7).y - y1);
	Point bottomCenter = Point(chinTip.x, crop.rows);

	cv::Mat stretch(crop.rows, crop.cols, CV_8U);
	cv::Mat mirror(crop.rows, crop.cols, CV_8U);

	Mat band1 = correctBandPerpective(crop, topCenter, noseTop,
			Point(crop.cols, noseTop.y));
	Mat band2 = correctBandPerpective(crop, noseTop, noseTip,
			Point(crop.cols - (abs(noseTop.x - noseTip.x)), noseTip.y));
	Mat band3 = correctBandPerpective(crop, noseTip, noseBase,
			Point(crop.cols - (abs(noseTip.x - noseBase.x)), noseBase.y));
	Mat band4 = correctBandPerpective(crop, noseBase, lipTop,
			Point(crop.cols - (abs(noseBase.x - lipTop.x)), lipTop.y));
	Mat band5 = correctBandPerpective(crop, lipTop, lipBottom,
			Point(crop.cols - (abs(lipTop.x - lipBottom.x)), lipBottom.y));
	Mat band6 = correctBandPerpective(crop, lipBottom, chinTip,
			Point(crop.cols - (abs(lipBottom.x - chinTip.x)), chinTip.y));
	Mat band7 = correctBandPerpective(crop, chinTip, bottomCenter,
			Point(crop.cols - (abs(chinTip.x - bottomCenter.x)),
					bottomCenter.y));

	for (int r = 0; r < crop.rows; r++) {
		if (r < noseTop.y) {
			band1.row(r).copyTo(stretch.row(r));
		} else if (r < noseTip.y) {
			band2.row(r).copyTo(stretch.row(r));
		} else if (r < noseBase.y) {
			band3.row(r).copyTo(stretch.row(r));
		} else if (r < lipTop.y) {
			band4.row(r).copyTo(stretch.row(r));
		} else if (r < lipBottom.y) {
			band5.row(r).copyTo(stretch.row(r));
		} else if (r < chinTip.y) {
			band6.row(r).copyTo(stretch.row(r));
		} else {
			band7.row(r).copyTo(stretch.row(r));
		}
	}

	for (int row = 0; row < crop.rows; row++) {
		for (int col = 0; col < crop.cols; col++) {
			if (col < crop.cols * 0.5)
				mirror.at<uchar>(row, col) = stretch.at<uchar>(row + 1, -col);
			else
				mirror.at<uchar>(row, col) = stretch.at<uchar>(row, col - 1);
		}
	}

	resize(mirror, mirror, Size(200, 240)); // image resizing for better performance

	return mirror;
}

/**
 * The SP index is defined as a weighted linear combination of values computed from them as follows:
 * SP = α · (1 − roll) + β · (1 − yaw) + γ · (1 − pitch) (4)
 */
double Face::computeSp(Point LPupil, Point RPupil, Point LEyebrowInner,
		Point CNoseTip, Point CNoseBase, Point CTipOfChin) {
	double theta = atan2((double) LPupil.y - RPupil.y, LPupil.x - RPupil.x); //deg = * 180 / CV_PI;
	double roll = min(abs((2 * theta) / CV_PI), 1.0); // rad

	double dl = cv::norm(LPupil - CNoseTip);
	double dr = cv::norm(RPupil - CNoseTip);
	double yaw = (max(dl, dr) - min(dl, dr)) / max(dl, dr);

	double eu = cv::norm(LEyebrowInner - CNoseTip);
	double ed = cv::norm(CNoseBase - CTipOfChin);
	double pitch = (max(eu, ed) - min(eu, ed)) / max(eu, ed);

	// being alpha = 0.1, beta = 0.6 and gamma = 0.3 as in article, page 153.
	double alpha = 0.1;
	double beta = 0.6;
	double gamma = 0.3;

	double sp = alpha * (1 - roll) + beta * (1 - yaw) + gamma * (1 - pitch);

	return sp;
}

/**
 * The sample illumination quality index SI is defined as a scalar in the interval [0,1] (the higher, the better), computed as:
 * SI = 1− F (std(mc)) (6)
 */
double Face::computeSi(Point LPupil, Point RPupil, Point LEyebrowInner,
		Point CNoseTip, Point CNoseBase, Point CTipOfChin) {

	Point p1 = calcMidpoint(stasmPts.at(0).x, stasmPts.at(0).y,
			stasmPts.at(39).x, stasmPts.at(39).y);
	Point p2 = calcMidpoint(stasmPts.at(3).x, stasmPts.at(3).y,
			stasmPts.at(39).x, stasmPts.at(39).y);
	Point p3 = LEyebrowInner;
	Point p4 = calcMidpoint((double) LEyebrowInner.x, (double) LEyebrowInner.y,
			(double) CNoseTip.x, (double) CNoseTip.y);
	Point p5 = CNoseTip;
	Point p6 = CTipOfChin;
	Point p7 = calcMidpoint(stasmPts.at(14).x, stasmPts.at(14).y,
			stasmPts.at(43).x, stasmPts.at(43).y);
	Point p8 = calcMidpoint(stasmPts.at(10).x, stasmPts.at(10).y,
			stasmPts.at(43).x, stasmPts.at(43).y);

	int length;
	if (face.rows > face.cols)
		length = face.cols;
	else
		length = face.rows;

	Mat subMatPt1 = face(
			cv::Rect(p1.x - (cvRound(length * 0.1) * 0.5),
					p1.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));
	Mat subMatPt2 = face(
			cv::Rect(p2.x - (cvRound(length * 0.1) * 0.5),
					p2.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));
	Mat subMatPt3 = face(
			cv::Rect(p3.x - (cvRound(length * 0.1) * 0.5),
					p3.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));
	Mat subMatPt4 = face(
			cv::Rect(p4.x - (cvRound(length * 0.1) * 0.5),
					p4.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));
	Mat subMatPt5 = face(
			cv::Rect(p5.x - (cvRound(length * 0.1) * 0.5),
					p5.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));
	Mat subMatPt6 = face(
			cv::Rect(p6.x - (cvRound(length * 0.1) * 0.5),
					p6.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));
	Mat subMatPt7 = face(
			cv::Rect(p7.x - (cvRound(length * 0.1) * 0.5),
					p7.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));
	Mat subMatPt8 = face(
			cv::Rect(p8.x - (cvRound(length * 0.1) * 0.5),
					p8.y - (cvRound(length * 0.1) * 0.5), cvRound(length * 0.1),
					cvRound(length * 0.1)));

	cv::imwrite("histograms/w1.png", subMatPt1); // save
	cv::imwrite("histograms/w2.png", subMatPt2); // save
	cv::imwrite("histograms/w3.png", subMatPt3); // save
	cv::imwrite("histograms/w4.png", subMatPt4); // save
	cv::imwrite("histograms/w5.png", subMatPt5); // save
	cv::imwrite("histograms/w6.png", subMatPt6); // save
	cv::imwrite("histograms/w7.png", subMatPt7); // save
	cv::imwrite("histograms/w8.png", subMatPt8); // save

	// histograms - http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
	double mc_w1 = (double) computeMassCenter("h1", subMatPt1);
	double mc_w2 = (double) computeMassCenter("h2", subMatPt2);
	double mc_w3 = (double) computeMassCenter("h3", subMatPt3);
	double mc_w4 = (double) computeMassCenter("h4", subMatPt4);
	double mc_w5 = (double) computeMassCenter("h5", subMatPt5);
	double mc_w6 = (double) computeMassCenter("h6", subMatPt6);
	double mc_w7 = (double) computeMassCenter("h7", subMatPt7);
	double mc_w8 = (double) computeMassCenter("h8", subMatPt8);

	double mc[8] = { mc_w1, mc_w2, mc_w3, mc_w4, mc_w5, mc_w6, mc_w7, mc_w8 };
//	printf("The dataset is %g, %g, %g, %g, %g, %g, %g, %g\n", mc[0], mc[1],
//			mc[2], mc[3], mc[4], mc[5], mc[6], mc[7]);

	double std = computeStdDev(mc);
//	printf("Std deviation = %f: \n", std);

	double si = 1 - computeSigmoid(std);

	return si;
}

/**
 * Computes the correlation between two cv::Mat.
 */
double Face::computeLocalCorrelation(Mat image1, Mat image2) {

	double img1PixMean = computePixelsMean(image1);
	double img2PixMean = computePixelsMean(image2);

	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;

	for (int i = 0; i < image1.cols; i++) {
		for (int j = 0; j < image1.rows; j++) {

			sum1 += (image1.at<uchar>(i, j) - img1PixMean)
					* (image2.at<uchar>(i, j) - img2PixMean);
			sum2 += pow(image1.at<uchar>(i, j) - img1PixMean, 2.0);
			sum3 += pow(image2.at<uchar>(i, j) - img2PixMean, 2.0);
		}
	}

	double correlation = sum1 / (sqrt(sum2 * sum3));

	return correlation;
}

/**
 *
 */
double Face::computeGlobalCorrelation(Mat regionA, Mat regionB) {
	double sum = 0.0;
	vector<double> localMax;
	int size = 40;
	int count = 0;

	for (int k = 0; k < regionA.cols; k += size) {
		for (int l = 0; l < regionA.rows; l += size) {

			Mat subRegionA = regionA(Rect(k, l, size, size));

			for (int u = -size; u <= size; u += size) {
				for (int v = -size; v <= size; v += size) {
					Rect rect(cv::Point(), regionB.size());
					Point p(k + u, l + v);

					if (rect.contains(p)) {

						Mat subRegionB = regionB(Rect(p.x, p.y, size, size));
						localMax.push_back(
								computeLocalCorrelation(subRegionA,
										subRegionB));
					}
				}
			}
			if (!localMax.empty()) {
				sum += *max_element(localMax.begin(), localMax.end());
				localMax.clear();
				count++;
			}
		}
	}

	return sum / count;
}

/**
 * Quasi-Linear Sigmoid (QLS) for data normalization.
 */
double Face::computeQls(double x, double xmax) {
	double distance = 1 - x;
	xmax = 1 - xmax;
	double a = 2 + sqrt(3);
	double b = 7 - (4 * sqrt(3));

	double qls = (1 - pow(b, distance / xmax))
			/ (a * pow(b, distance / xmax) + 1);

	return qls;
}

/**
 * Assume that all the images in the folder are models and are recorded in the system.
 * Training system to determine the threshold with current medelos.
 */
double Face::train(char* path) {

	vector<string> filenames;
	vector<double> bhk;

	/**
	 * Iterate trough all the gallery templates and search for .jpg and .pgm files, in order
	 * to obtain the list of elements of the database.
	 * File names are stored in filenames vector.
	 */
	namespace fs = boost::filesystem;
	fs::path someDir(path);
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

	// Sort filenames.
	std::sort(filenames.begin(), filenames.end());
	double dmax = 0;

	string root;
	string imgPath;
	string absPath;

	if (!filenames.empty()) {

		// Gallery comparison: Every file is compared to all the files in the gallery.
		for (unsigned i = 0; i < filenames.size(); i++) {

			root = path;
			imgPath = filenames.at(i).c_str();
			absPath = root + imgPath;
			init(absPath.c_str());
			Mat candidate = loadMat();
			vector<Point> stasmPtsVector = getStasmPts();
			map<double, string> correlationsMap;

			Point lPupil = stasmPtsVector.at(31);
			Point rPupil = stasmPtsVector.at(36);
			Point noseTip = stasmPtsVector.at(67);
			Point lEyebrowInner = stasmPtsVector.at(24);
			Point noseBase = stasmPtsVector.at(41);
			Point tipOfChin = stasmPtsVector.at(7);

			cout << "Candidate | ";

			// Compute SP and SI
			double sp = computeSp(lPupil, rPupil, lEyebrowInner, noseTip,
					noseBase, tipOfChin);
			cout << "SP = " << sp << " | ";
			double si = computeSi(lPupil, rPupil, lEyebrowInner, noseTip,
					noseBase, tipOfChin);
			cout << "SI = " << si << " | ";
			cout << path << endl
					<< "-----------------------------------------------------------------------------------------------------------------"
					<< endl;

			// 4. A), B), C), D) and E) Pose Normalization
			candidate = normalizePose(candidate, lPupil, rPupil, lEyebrowInner,
					noseTip, noseBase, tipOfChin);

			// 4.F) Illumination Normalization
			candidate = normalizeIllumination(candidate);

			for (unsigned f = 0; f < filenames.size(); f++) {

				root = path;
				imgPath = filenames.at(f).c_str();
				absPath = root + imgPath;
				init(absPath.c_str());
				Mat model = loadMat();
				vector<Point> stasmPtsVector2 = getStasmPts();

				Point lPupil2 = stasmPtsVector2.at(31);
				Point rPupil2 = stasmPtsVector2.at(36);
				Point noseTip2 = stasmPtsVector2.at(67);
				Point lEyebrowInner2 = stasmPtsVector2.at(24);
				Point noseBase2 = stasmPtsVector2.at(41);
				Point tipOfChin2 = stasmPtsVector2.at(7);

				cout << filenames.at(f) << " | ";

				// Compute SP and SI
				double sp2 = computeSp(lPupil2, rPupil2, lEyebrowInner2,
						noseTip2, noseBase2, tipOfChin2);
				cout << "SP = " << sp2 << " | ";
				double si2 = computeSi(lPupil2, rPupil2, lEyebrowInner2,
						noseTip2, noseBase2, tipOfChin2);
				cout << "SI = " << si2 << " | ";

				// 4. A), B), C), D) and E) Pose Normalization
				model = normalizePose(model, lPupil2, rPupil2, lEyebrowInner2,
						noseTip2, noseBase2, tipOfChin2);

				// 4.F) Illumination Normalization
				model = normalizeIllumination(model);

				double globalCorrelation = computeGlobalCorrelation(candidate,
						model);
				cout << "global correlation = " << globalCorrelation << endl;

				if ((1 - globalCorrelation) > dmax)
					dmax = globalCorrelation;

				correlationsMap.insert(
						make_pair(1 - globalCorrelation, filenames.at(f)));
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
				dg1 = (dg1 == 0) ? computeQls(iter->first, dmax) : dg1;

				// compute the distance between p and gi2 (dg2).
				string str = iter->second;
				index = str.find("_");
				sub_str = str.substr(0, index);

				if (identityStr.empty()) {
					identityStr = sub_str;
				} else if (sub_str.compare(identityStr) != 0 && dg2 == 0) {
					dg2 = computeQls(iter->first, dmax);
				} else // compute the distance between p and gi|G| (dgG).
				{
					dgG = computeQls(iter->first, dmax);
				}

				/**
				 * The second function φ2 (p) (density ratio) is computed using
				 * the ratio between the number of subjects in the gallery, distinct
				 * from the returned identity, giving a distance lower than twice
				 * F(d(p, gi 1)), and the cardinality G of the gallery.
				 */
				if (computeQls(iter->first, dmax) < (2 * dg1)) {
					nb++; // Nb = {gi k ∈ G|F (d(p, gi k )) < 2F (d(p, gi 1 ))}.
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
			double phij_ = 0.1; // COMO NO EXEMPLO DO ARTIGO, SERÁ NECESSÁRIO ENCONTRAR UM VALOR?
			double s1 = (phi1 > phij_) ? 1 - phij_ : phij_;
			double s2 = (phi2 > phij_) ? 1 - phij_ : phij_;

			/*
			 * SRR I corresponds to the genuine element.
			 * SRR I is computed starting from the relative distance (computed starting
			 * from the distance between the scores of the first two retrieved
			 * distinct identities)
			 *
			 * Primeiro medimos a distância absoluta entre φj(p) e o ponto “crítico”.
			 * Esta distância obtém valores mais altos para φj(p) e
			 * muito mais mais altos/baixos que φj (genuíno/impostor respetivamente).
			 */
			double dist1 = abs(phi1 - phij_);
			double srr1 = dist1 / s1;

			/*
			 * SRR II corresponds to the element closest to the genuine one.
			 * SRR II uses the density ratio (relative amount of gallery templates which are “near” to the
			 * retrieved identity although belonging to different identities).
			 *
			 * Primeiro medimos a distância absoluta entre φj(p) e o ponto “crítico”.
			 * Esta distância obtém valores mais altos para φj(p) e
			 * muito mais mais altos/baixos que φj (genuíno/impostor respetivamente).
			 */
			double dist2 = abs(phi2 - phij_);
			double srr2 = dist2 / s2;

			bhk.push_back(srr2);
		}
	}

	// calcular thk e devolver o valor do linear!
	// ver http://www.softwareandfinance.com/CPP/Covariance_Correlation.html

	if (!bhk.empty() && !bhk.empty())
	{
		double srrMean = computeMean(bhk.data());
		double srrStdDev = computeStdDev(bhk.data());

		// thk = média^2 - desvio padrão / média
		double threshold = abs(pow(srrMean, 2.0) - srrStdDev) / srrMean;
	}

	return 0.0;
}

/**
 * Print error if STASM library is not found.
 */
void Face::error(const char* s1, const char* s2) {
	printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
	exit(1);
}

/**
 * Rotate image depending on the angle.
 * From http://stackoverflow.com/questions/2289690/opencv-how-to-rotate-iplimage
 */
Mat Face::rotateImage(const Mat& source, double angle) {
	Point2f src_center(source.cols / 2.0F, source.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	Mat dst;
	warpAffine(source, dst, rot_mat, source.size());

	return dst;
}

/**
 * Affine transformations to correct the band perspective.
 * Code based on http://stackoverflow.com/questions/7838487/executing-cvwarpperspective-for-a-fake-deskewing-on-a-set-of-cvpoint
 * 1------3
 * |    -
 * |  -
 * 2-
 */
Mat Face::correctBandPerpective(Mat src, Point pt1, Point pt2, Point pt3) {
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
double Face::computeSigmoid(double x) {

	double s = 0.0;
	s = 1.0 / (1.0 + exp(-x / 160));

	return s;
}

/**
 * Calculate mean. http://www.softwareandfinance.com/CPP/MeanVarianceStdDevi.html
 */
double Face::computeMean(double value[]) {
	double max = 8;

	double sum = 0;
	for (int i = 0; i < max; i++) {
		sum += value[i];
	}

	return (sum / max);
}

/**
 * Calculate Variance. From http://www.softwareandfinance.com/CPP/MeanVarianceStdDevi.html
 */
double Face::computeStdDev(double value[]) {
	int max = 8;
	double mean;
	mean = computeMean(value);

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
double Face::computeMassCenter(std::string const& name, Mat1b const& image) {
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
//	printf("sum1=%f / sum2=%f | mc = %f\n", sum1, sum2, sum1 / sum2);
// mc formula
	double mc = sum1 / sum2;

// show and save histograms
	circle(hist_image, Point(mc, 255), 5, cvScalar(255, 0, 0), -1, 8);
	cv::imwrite(std::string("histograms/") + name + ".png", hist_image); // save
//    cv::imshow(name, hist_image);

	return mc;
}

/**
 * Returns the midpoint coordinates.
 */
Point Face::calcMidpoint(double x1, double y1, double x2, double y2) {
	return Point((x1 + x2) / 2, (y1 + y2) / 2);
}

/**
 * Calculates the average value of the pixels of a matrix.
 */
double Face::computePixelsMean(Mat img) {
	vector<Mat> channels;
	split(img, channels);
	Scalar m = mean(channels[0]);

	return m[0];
}

/**
 * Calculates the angle between two points in degrees.
 */
double Face::computePointsAngle(Point pt1, Point pt2) {
	double deltaY = (pt2.y - pt1.y);
	double deltaX = (pt2.x - pt1.x);

	double angleInDegrees = atan2(deltaY, deltaX) * 180 / CV_PI;

	return angleInDegrees;
}

/**
 * Rotate STASM coordinates to match rotated image.
 */
Point Face::rotatePoint(Point pt, double angle) {

	angle = angle / (180 / CV_PI); // convert angle to rad

	Point center = Point(face.cols * 0.5, face.rows * 0.5);
	Point midpoint = Point(pt.x - center.x, pt.y - center.y);

	pt.x = (midpoint.x * cos(angle)) - (midpoint.y * sin(angle));
	pt.y = (midpoint.x * sin(angle)) + (midpoint.y * cos(angle));

	pt.x = (int) ceil(pt.x) + center.x;
	pt.y = (int) ceil(pt.y) + center.y;

	return pt;
}

/**
 * Function to convert color image to a gray scale image
 * Input: IplImage* - input color imageData Output: IplImage* - the gray scale image
 */
IplImage * Face::Rgb2Gray(IplImage * src) {
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

/**
 * This function is used to convert IplImage datatype to a CvMat datatype.
 * Input: IplImage* - Input image(grey scale) Output: CvMat*- The converted matrix.
 */
CvMat * Face::IplImage2Mat(IplImage * inp_img) {

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

/**
 * This function is used to convert a matrix into a image.
 * Input: CvMat* - the input matrix that needs to be converted. int - This is used to specify the type of output image(eg:IPL_DEPTH_32F or IPL_DEPTH_8U) Output: IplImage* - The converted image.
 */
IplImage* Face::Mat2IplImage(CvMat *inp_mat, int type) {

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

/**
 * This function is used to scale a input matrix Input: CvMat* - The input matrix,
 * that needs to be scaled(the same matrix is scaled) double - The maximum range(eg:255)
 * Return : int - 1 is returned when everything works fine.
 */
int Face::Scale_Mat(CvMat *input, double scale) {
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

/**
 * This function is used to compute weighted gaussian kernel matrix
 * Input: CvMat* - A small portion of the image where weighted gaussian needs to be applied. * The size of this portion should be same as the Gaussian filter. CvMat* - The gaussian filter.
 * Output: CvMat* - The weighted gaussian kernel filter
 */
CvMat * Face::Weighted_Gaussian(CvMat * inp, CvMat * gaussian) {
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
			tmp2 = tmp1 / (scl_factor * 1);
			cvmSet(w_gauss, i, j, tmp2);
		}
	}

	return w_gauss;
}

/**
 * This function is used to extract a small region within an image.
 * Input: CvPoint - Specifying the starting point. int- width, int - height, IplImage* - The input image. Output: CvMat* - The extracted portion.
 */
CvMat* Face::Get_Mat(CvPoint a, int width, int height, IplImage *image) {

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
 * This function is used for applying wighted gaussian filter over an image.
 * This Input: IplImage* - The input image. The image should be a grey scale image. CvMat* - The gaussian kernel matrix.
 * Output: CvMat* -The result after applying the kernel on the input image. Size of this is same as that of the input image.
 */
CvMat * Face::Conv_Weighted_Gaussian(IplImage * inp_img, CvMat * kernel) {
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

/**
 * 	This function returns a gaussian kernel matrix of prescribed size.
 * 	Input: int -- The size of the gaussian kernel Output: CvMat* -- The Gaussian kernel matrix.
 */
CvMat* Face::Gaussian(int size) {
	CvMat *res;
	int i, j;
	int x, y;
	double tmp;
	double sigma;
	int halfsize;

	sigma = (double) size / 10;
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

/**
 * "This function computes SQI(self quotient image) of an input image. Input: IplImage* - input image Output: IplImage* - the SQI of the input image".
 * From http://libface.sourceforge.net/libface-0.1-html/d8/d92/libface_8h.html.
 */
IplImage * Face::SQI(IplImage * inp) {
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

#endif /* FACE_CPP_ */
