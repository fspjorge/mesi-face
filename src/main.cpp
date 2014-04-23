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

	Face face = Face();
	char path[] = "/home/jorge/workspace/dissertacao/templates/";
	face.train(path);

	time = clock() - time;
	int ms = double(time) / CLOCKS_PER_SEC * 1000;
	cout << "Elapsed time is " << ms << " milliseconds" << endl;

	cv::waitKey(0);

	return 0;
}
