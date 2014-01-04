/*
 * face.cpp
 *
 *  Created on: Dec 23, 2013
 *      Author: jorge
 */

#ifndef FACE_CPP_
#define FACE_CPP_

#include "Face.h"

Face::Face(const char* imgPath) {

	if (!stasm_init("data", 0 /*trace*/))
		error("stasm_init failed: ", stasm_lasterr());

	cv::Mat_<unsigned char> img(cv::imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE));

	if (!img.data)
		error("Cannot load", imgPath);

	if (!stasm_open_image((const char*) img.data, img.cols, img.rows, imgPath,
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
		stasm_convert_shape(landmarks, 68);

		// draw the landmarks on the image as white dots
		int i = 0;
		stasm_force_points_into_image(landmarks, img.cols, img.rows);
		for (i = 0; i < stasm_NLANDMARKS; i++) {
			stasmPts.push_back(
					Point(cvRound(landmarks[i * 2]),
							cvRound(landmarks[i * 2 + 1])));
		}

	}
	this->face = img;
	nfaces++;
}

void Face::error(const char* s1, const char* s2) {
	printf("Stasm version %s: %s %s\n", stasm_VERSION, s1, s2);
	exit(1);
}

/**
 * Calculates the angle between two points in degrees.
 */
double Face::angleBetweenTwoPoints(Point pt1, Point pt2) {
	double deltaY = (pt2.y - pt1.y);
	double deltaX = (pt2.x - pt1.x);

	double angleInDegrees = atan2(deltaY, deltaX) * 180 / CV_PI;

	return angleInDegrees;
}

/**
 * Get STASM coordinates after rotation.
 * http://stackoverflow.com/questions/7953316/rotate-a-point-around-a-point-with-opencv
 */
Point Face::rotatePoint(Point pt, double angle) {

	angle = angle / (180 / CV_PI);

	Point center = Point(face.cols * 0.5, face.rows * 0.5);
	Point midpoint = Point(pt.x - center.x, pt.y - center.y);

	pt.x = (midpoint.x * cos(angle)) - (midpoint.y * sin(angle));
	pt.y = (midpoint.x * sin(angle)) + (midpoint.y * cos(angle));

	pt.x = (int) ceil(pt.x) + center.x;
	pt.y = (int) ceil(pt.y) + center.y;

	return pt;
}

/**
 * Pose correction.
 */
Mat Face::normalizePose(Mat face, Point LPupil, Point RPupil,
		Point LEyebrowInner, Point CNoseTip, Point CNoseBase,
		Point CTipOfChin) {

	double theta = angleBetweenTwoPoints(LPupil, RPupil);
	face = rotateImage(face, theta);

	for (unsigned int i = 0; i < 68; i++) {
		stasmPts.at(i) = rotatePoint(getStasmPts().at(i), -theta);
	}

	LPupil = getStasmPts().at(31);
	RPupil = getStasmPts().at(36);
	CNoseTip = getStasmPts().at(67);
	LEyebrowInner = getStasmPts().at(24);
	CNoseBase = getStasmPts().at(41);
	CTipOfChin = getStasmPts().at(7);
	double dl = cv::norm(LPupil - CNoseTip);
	double dr = cv::norm(RPupil - CNoseTip);

	if (gsl_fcmp(dl, dr, DBL_EPSILON) > 0) { // x = y returns 0; if x < y returns -1; x > y returns +1;
		flip(face, face, 1);
		for (unsigned int i = 0; i < 68; i++) {
			stasmPts.at(i) = Point(face.cols - stasmPts.at(i).x,
					stasmPts.at(i).y);
		}
	}

	int x2 = stasmPts.at(1).x + 7;
	int y2 = stasmPts.at(7).y + 7;
	int x1 = stasmPts.at(13).x - 7;
	int y1 = stasmPts.at(23).y - 20;

	int width = abs(x1 - x2);
	int height = abs(y2 - y1);

	cout << "x1 = " << x1 << endl;
	cout << "y1 = " << y1 << endl;
	cout << "x2 = " << x2 << endl;
	cout << "y2 = " << y2 << endl;

	cout << "width = " << width << endl;
	cout << "height = " << height << endl;

//	imshow("face", face);

//	sleep(30);

	 // OK até aqui

	Mat crop = face(Rect(x1, y1, width, height)); // problema no crop

	imshow("crop", crop);

	Point noseTip = Point(stasmPts.at(67).x - x1 - 2, stasmPts.at(67).y - y1); //82,121
	Point noseTop = calcMidpoint(stasmPts.at(24).x - x1 - 2,
			stasmPts.at(24).y - y1, stasmPts.at(18).x - x1 - 2,
			stasmPts.at(18).y - y1); // 88,45
	Point topCenter = Point(noseTop.x, 0); //
	Point noseBase = Point(stasmPts.at(41).x - x1 - 2, stasmPts.at(41).y - y1); //OK
	Point lipTop = Point(stasmPts.at(51).x - x1 - 2, stasmPts.at(51).y - y1);
	Point lipBottom = Point(stasmPts.at(57).x - x1 - 2, stasmPts.at(57).y - y1);
	Point chinTip = Point(stasmPts.at(7).x - x1 - 2, stasmPts.at(7).y - y1);
	Point bottomCenter = Point(chinTip.x, crop.rows);

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

	for (int row = 0; row < crop.rows; row++) {
		for (int col = 0; col < crop.cols; col++) {
			if (col < crop.cols * 0.5)
				out2.at<uchar>(row, col) = out.at<uchar>(row + 1, -col);
			else
				out2.at<uchar>(row, col) = out.at<uchar>(row, col - 1);
		}
	}

	imshow("out2", out2);

	return out2;
}

double Face::calcSp(Point LPupil, Point RPupil, Point LEyebrowInner,
		Point CNoseTip, Point CNoseBase, Point CTipOfChin) {
	double theta = atan2((double) LPupil.y - RPupil.y, LPupil.x - RPupil.x); //deg = * 180 / CV_PI;
	double roll = min(abs((2 * theta) / CV_PI), 1.0); // rad

	double dl = cv::norm(LPupil - CNoseTip);
	double dr = cv::norm(RPupil - CNoseTip);
	double yaw = (max(dl, dr) - min(dl, dr)) / max(dl, dr);

	double eu = cv::norm(LEyebrowInner - CNoseTip);
	double ed = cv::norm(CNoseBase - CTipOfChin);
	double pitch = (max(eu, ed) - min(eu, ed)) / max(eu, ed);

	// being alpha = 0.1, beta = 0.6 and gamma = 0.3 | article page 153
	double alpha = 0.1;
	double beta = 0.6;
	double gamma = 0.3;

	double sp = alpha * (1 - roll) + beta * (1 - yaw) + gamma * (1 - pitch);

	return sp;
}

double Face::calcSi(Point LPupil, Point RPupil, Point LEyebrowInner,
		Point CNoseTip, Point CNoseBase, Point CTipOfChin) {
// Finding the 8 points
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

	double mc[8] = { mc_w1, mc_w2, mc_w3, mc_w4, mc_w5, mc_w6, mc_w7, mc_w8 };
	printf("The dataset is %g, %g, %g, %g, %g, %g, %g, %g\n", mc[0], mc[1],
			mc[2], mc[3], mc[4], mc[5], mc[6], mc[7]);

	double std = calculateStd(mc);
	printf("Std deviation = %f: \n", std);

	double si = 1 - sigmoid(std);

	return si;
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
 * 1-----3
 * |  -
 * 2-
 */
Mat Face::correctPerpective(Mat src, Point pt1, Point pt2, Point pt3) {
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
double Face::sigmoid(double x) {

	double s = 0.0;
	s = 1.0 / (1.0 + exp(-x / 160));

	return s;
}

/**
 * Calculate mean. http://www.softwareandfinance.com/CPP/MeanVarianceStdDevi.html
 */
double Face::calculateMean(double value[]) {
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
double Face::calculateStd(double value[]) {
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
double Face::getMassCenter(std::string const& name, Mat1b const& image) {
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

/**
 * Returns the midpoint coordinates.
 */
Point Face::calcMidpoint(double x1, double y1, double x2, double y2) {
	return Point((x1 + x2) / 2, (y1 + y2) / 2);
}

/**
 * Get STASM points array from image.
 */
vector<Point> Face::getStasmPts() {

	return this->stasmPts;
}

/**
 * Calculates the average value of the pixels of a matrix.
 */
double Face::pixelsMean(Mat img) {
	vector<Mat> channels;
	split(img, channels);
	Scalar m = mean(channels[0]);

	return m[0];
}

/**
 * Divides a region in 8x8 square sub-regions and puts them in a vector.
 * Returns a vector of sub-regions (cv::Mat).
 */
vector<Mat> Face::divideIntoSubRegions(Mat region) {

	vector<Mat> subRegions;
	int roiSize = 8;

	for (int i = 0; i < region.cols / roiSize; ++i) {
		for (int j = 0; j < region.rows / roiSize; ++j) {

			try {
				if (region.at<uchar>(i, j) > 0) {
					Mat subRegion = region(Rect(i, j, roiSize, roiSize));
					subRegions.push_back(subRegion);
//					cout << "i = " << i << " j = " << j << endl;
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
double Face::localCorrelation(Mat rA, Mat rB) {

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
//	cout << "corrLocal = " << corr << endl;

	return corr;
}

/**
 * Soma dos máximos das subregiões (WORK IN PROGRESS).
 */
double Face::globalCorrelation(Mat A, Mat B) {
	double SumAB = 0.0;
	vector<double> localMax;

	cv::resize(A, B, A.size(), 0, 0, cv::INTER_CUBIC);

	vector<Mat> subRegionsOfA = divideIntoSubRegions(A);
	vector<Mat> subRegionsOfB = divideIntoSubRegions(B);

	unsigned int regionsPerLine = div(A.cols, 8).quot;

//	cout << "regionsPerLine = " << regionsPerLine << endl;

	for (unsigned int i = 0; i < subRegionsOfA.size(); i++) {
		localMax.clear();

		if (i == 0) {
			/**
			 * ____
			 * |__|
			 *
			 */
//			cout << "i = 0 " << endl;
//			cout << "A comparar a região i(" << i << ") com a região i(" << i << ")"<< endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i)));
		} else if (i > 0 && i < regionsPerLine) {

			/**
			 * ______
			 * |__|__|
			 *
			 */
//			cout << "i > 0 && i < regionsPerLine " << endl;
//			cout << "A comparar a região i(" << i << ") com a região i(" << i << ")"<< endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i)));
//			cout << "A comparar a região i(" << i << ") com a região i(" << (i-1) << ")"<< endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i),
							subRegionsOfB.at(i - 1)));
		} else if (i % regionsPerLine == 0) {
			/**
			 * ____
			 * |__|
			 * |__|
			 *
			 */
//			cout << "i > 0 && i < regionsPerLine " << endl;
//			cout << "A comparar a região i(" << i << ") com a região i-regionsPerLine(" << (i-regionsPerLine) << ")"<< endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i),
							subRegionsOfB.at(i - regionsPerLine)));
//			cout << "A comparar a região i(" << i << ") com a região i(" << i << ")"<< endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i)));
		} else {
			/**
			 * ______
			 * |__|__|
			 * |__|__|
			 *
			 */

//			cout << "i > 0 && i < regionsPerLine " << endl;
//
//			cout << "A comparar a região " << i << " com a região i(" << i << ")" << endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i), subRegionsOfB.at(i)));
//			cout << "A comparar a região " << i << " com a região i-1(" << (i - 1) << ")" << endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i),
							subRegionsOfB.at(i - 1)));
//			cout << "A comparar a região " << i << " com a região i - regionsPerLine(" << i - regionsPerLine << ")" << endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i),
							subRegionsOfB.at(i - regionsPerLine)));
//			cout << "A comparar a região " << i << " com a região i - regionsPerLine + 1 (" << (i - regionsPerLine + 1) << ")" << endl;
			localMax.push_back(
					localCorrelation(subRegionsOfA.at(i),
							subRegionsOfB.at(i - regionsPerLine + 1)));
		}
		//somar máximos locais
		SumAB += *max_element(localMax.begin(), localMax.end());
	}

	return SumAB;
}

IplImage* Face::Rgb2Gray(IplImage *src) {
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

CvMat* Face::IplImage2Mat(IplImage *inp_img) {

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

CvMat* Face::Weighted_Gaussian(CvMat *inp, CvMat *gaussian) {
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
 *
 */
CvMat* Face::Conv_Weighted_Gaussian(IplImage *inp_img, CvMat *kernel) {
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

CvMat* Face::Gaussian(int size) {
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

IplImage* Face::SQI(IplImage* inp) {
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
