/**
 * Author: Jorge Pereira
 * MESI
 */

#include "Face.h"

#define C_TEXT( text ) ((char*)std::string( text ).c_str())

using namespace cv;
using namespace std;

int main() {

	clock_t time = clock();

	Face face;

	if (!stasm_init("data", 0 /*trace*/))
		face.error("stasm_init failed: ", stasm_lasterr());

	static const char* path = "2013-11-18-173422.jpg";

	cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));

	if (!img.data)
		face.error("Cannot load", path);

	if (!stasm_open_image((const char*) img.data, img.cols, img.rows, path,
			1 /*multiface*/, 10 /*minwidth*/))
		face.error("stasm_open_image failed: ", stasm_lasterr());

	int foundface;
	float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

	int nfaces = 0;
	while (1) {
		if (!stasm_search_auto(&foundface, landmarks))
			face.error("stasm_search_auto failed: ", stasm_lasterr());

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
			p1 = face.midpoint((double) cvRound(landmarks[0 * 2]),
					(double) cvRound(landmarks[0 * 2 + 1]),
					(double) cvRound(landmarks[58 * 2]),
					(double) cvRound(landmarks[58 * 2 + 1]));
			p2 = face.midpoint((double) cvRound(landmarks[3 * 2]),
					(double) cvRound(landmarks[3 * 2 + 1]),
					(double) cvRound(landmarks[58 * 2]),
					(double) cvRound(landmarks[58 * 2 + 1]));
			p3 = LEyebrowInner;
			p4 = face.midpoint((double) LEyebrowInner.x, (double) LEyebrowInner.y,
					(double) CNoseTip.x, (double) CNoseTip.y);
			p5 = CNoseTip;
			p6 = CTipOfChin;
			p7 = face.midpoint((double) cvRound(landmarks[54 * 2]),
					(double) cvRound(landmarks[54 * 2 + 1]),
					(double) cvRound(landmarks[12 * 2]),
					(double) cvRound(landmarks[12 * 2 + 1]));
			p8 = face.midpoint((double) cvRound(landmarks[54 * 2]),
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

		imshow("Original img", img);

		// C. POSE NORMALIZATION ###########################################

		// 4.(a) rotation
		double theta_deg = theta * 180 / CV_PI;
		img = face.rotateImage(img, -180 + theta_deg);
//		cv::imshow("rotatedImg", img);
//		cv::waitKey(0);

		// 4.(b) horizontal flip if dr smaller than dl
		if (gsl_fcmp(dl, dr, DBL_EPSILON) > 0) { // x = y returns 0; if x < y returns -1; x > y returns +1;
			flip(img, img, 1);
		}

		std::vector<Point> roi_vector = face.getNewStasmPts(img, 68);

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
		std::vector<Point> stasm_vector = face.getNewStasmPts(crop, 68);

		Point noseTop = face.midpoint(stasm_vector.at(24).x, stasm_vector.at(24).y,
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

		Mat band1 = face.correctPerpective(crop, topCenter, noseTop,
				Point(crop.cols, noseTop.y));
		Mat band2 = face.correctPerpective(crop, noseTop, noseTip,
				Point(crop.cols - (abs(noseTop.x - noseTip.x)), noseTip.y));
		Mat band3 = face.correctPerpective(crop, noseTip, noseBase,
				Point(crop.cols - (abs(noseTip.x - noseBase.x)), noseBase.y));
		Mat band4 = face.correctPerpective(crop, noseBase, lipTop,
				Point(crop.cols - (abs(noseBase.x - lipTop.x) + 2), lipTop.y));
		Mat band5 = face.correctPerpective(crop, lipTop, lipBottom,
				Point(crop.cols - (abs(lipTop.x - lipBottom.x)), lipBottom.y));
		Mat band6 = face.correctPerpective(crop, lipBottom, chinTip,
				Point(crop.cols - (abs(lipBottom.x - chinTip.x)), chinTip.y));
		Mat band7 = face.correctPerpective(crop, chinTip, bottomCenter,
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

		std::vector<Point> stasm_vector2 = face.getNewStasmPts(out2, 68);

		imshow("Mirror right to left", out2);

		// 4.(f) função sqi
		Mat illumNorn;

		IplImage copy = out2;

		illumNorn = Mat(face.SQI(&copy));

		face.localCorrelation(out2, out2);

		face.divideIntoSubRegions(out2);

		cout << "globalCorr = " << face.globalCorrelation(out2, out2) << endl;

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
