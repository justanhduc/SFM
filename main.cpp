#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include "utility.h"
#include "sfm.h"

#include <iostream>
#include <vector>
#include <assert.h>

#define nnRatio		0.8
#define RANSAC		1				//whether use RANSAC for fundamental matrix or not
#define THRESHOLD	1.				//RANSAC threshold for outliers
#define EPSILON		0.55			//proportion of outliers in data

using namespace std;
using namespace cv;

void main() {
	Mat colorImgL = imread("bicycle/im0.png", CV_LOAD_IMAGE_COLOR);
	Mat colorImgR = imread("bicycle/im0.png", CV_LOAD_IMAGE_COLOR);

	Mat imageL = imread("bicycle/im0.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imageR = imread("bicycle/im1.png", CV_LOAD_IMAGE_GRAYSCALE);

	assert(imageL.data && imageR.data && colorImgL.data && colorImgR.data);

	int minHessian = 500;
	SurfFeatureDetector detector(minHessian);
	vector<KeyPoint> keypointsL, keypointsR;
	detector.detect(imageL, keypointsL);
	detector.detect(imageR, keypointsR);

	SurfDescriptorExtractor extractor;
	Mat descriptorsL, descriptorsR;

	extractor.compute(imageL, keypointsL, descriptorsL);
	extractor.compute(imageR, keypointsR, descriptorsR);

	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;

	if (keypointsL.size() >= keypointsR.size())
		matcher.match(descriptorsR, descriptorsL, matches);
	else
		matcher.match(descriptorsL, descriptorsR, matches);

	int nPoints = matches.size();
	Mat kpntsL = Mat::zeros(3, nPoints, CV_64FC1);
	Mat kpntsR = Mat::zeros(3, nPoints, CV_64FC1);

	int i = 0;
	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); i++, ++it) {
		if (keypointsL.size() >= keypointsR.size()) {
			kpntsL.at<double>(0, i) = keypointsL[it->trainIdx].pt.x;
			kpntsL.at<double>(1, i) = keypointsL[it->trainIdx].pt.y;
			kpntsL.at<double>(2, i) = 1;

			kpntsR.at<double>(0, i) = keypointsR[it->queryIdx].pt.x;
			kpntsR.at<double>(1, i) = keypointsR[it->queryIdx].pt.y;
			kpntsR.at<double>(2, i) = 1;
		}
		else {
			kpntsL.at<double>(0, i) = keypointsL[it->queryIdx].pt.x;
			kpntsL.at<double>(1, i) = keypointsL[it->queryIdx].pt.y;
			kpntsL.at<double>(2, i) = 1;

			kpntsR.at<double>(0, i) = keypointsR[it->trainIdx].pt.x;
			kpntsR.at<double>(1, i) = keypointsR[it->trainIdx].pt.y;
			kpntsR.at<double>(2, i) = 1;
		}
	}

	Mat F, E;

#if RANSAC
	F = getFundamentalMatrixRansac(kpntsL, kpntsR, nPoints, THRESHOLD);
#else
	Mat TnormL = getNormalizationMatrix(kpntsL, nPoints);
	Mat TnormR = getNormalizationMatrix(kpntsR, nPoints);
	Mat pntsNormL = TnormL*kpntsL;
	Mat pntsNormR = TnormR*kpntsR;

	F = TnormR.t()*getFundamentalMatrix(pntsNormL, pntsNormR, 8)*TnormL;
#endif

	Vec2DPoint points2DL = matPointToVecPoint(kpntsL);
	Vec2DPoint points2DR = matPointToVecPoint(kpntsR);

	//Bicycle
	Mat leftK = (Mat_<double>(3, 3) << 5299.313, 0, 1263.818, 0, 5299.313, 977.763, 0, 0, 1);
	Mat rightK = (Mat_<double>(3, 3) << 5299.313, 0, 1438.004, 0, 5299.313, 977.763, 0, 0, 1);

	//Piano
	//Mat leftK = (Mat_<double>(3, 3) << 2826.171, 0, 1292.2, 0, 2826.171, 965.806, 0, 0, 1);
	//Mat rightK = (Mat_<double>(3, 3) << 2826.171, 0, 1415.97, 0, 2826.171, 965.806, 0, 0, 1);

	//temple
	//Mat leftK = (Mat_<double>(3, 3) << 1520.400000, 0, 302.32, 0, 1525.900000, 246.870000, 0, 0, 1);
	//Mat rightK = (Mat_<double>(3, 3) << 1520.400000, 0, 302.32, 0, 1525.900000, 246.870000, 0, 0, 1);

	//Flowers
	//Mat leftK = (Mat_<double>(3, 3) << 4396.869, 0, 1353.072, 0, 4396.869, 989.702, 0, 0, 1);
	//Mat rightK = (Mat_<double>(3, 3) << 4396.869, 0, 1538.86, 0, 4396.869, 989.702, 0, 0, 1);

	//backpack
	//Mat leftK = (Mat_<double>(3, 3) << 7190.247, 0, 1035.513, 0, 7190.247, 945.196, 0, 0, 1);
	//Mat rightK = (Mat_<double>(3, 3) << 7190.247, 0, 1378.036, 0, 7190.247, 945.196, 0, 0, 1);

	E = getEssentialMatrix(F, leftK, rightK);

	VecMat cddP = getCandidateP(E);
	VecMat pairP = getProjectionMatrix(cddP, points2DL, points2DR, leftK, rightK);
	//VecMat pairP = getTempProjectionMatrix(F);
	Vec3DPoint worldPoints = triangulatePoints(points2DL, points2DR, points2DL.size(), pairP[0], pairP[1]);
	
	if (keypointsL.size() >= keypointsR.size()) {
		exportPointCloud(worldPoints, colorImgL, matches, keypointsL);
	}
	else {
		exportPointCloud(worldPoints, colorImgR, matches, keypointsR);
	}

	exit(0);
}