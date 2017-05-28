/* utility.h */

#ifndef UTILITY_H_
#define UTILITY_H_

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"

#include <iostream>

#include "sfm.h"

using namespace std;
using namespace cv;

Mat solveLSE(Mat A);

void cvMatFromMat(Mat inMat, CvMat* outMat);

Mat normalizeCvMatRow(Mat A);

double vectorNorm(Mat A);

template <typename T> int sign(T val);

template <typename T> int getMaxIdx(T* arr, int size);

Mat reshape(Mat inMat, int newNumRows, int newNumCols);

Mat getNormalizationMatrix(Mat point, int nPoints);

void exportPointCloud(Vec3DPoint worldPoints, Mat colorRefImg, 
	vector<DMatch> matches, vector<KeyPoint> refKeypoints);

VecMat getRowsFromMat(Mat inMat);

Mat getRowsFromMat(Mat inMat, int from, int to);

VecMat getColsFromMat(Mat inMat);

Mat getColsFromMat(Mat inMat, int from, int to);

Vec2DPoint matPointToVecPoint(Mat matPoint);

Mat vecPointToMatPoint(Vec3DPoint vecpoint);

/************************************************************************************/

#include "utility.tpp"

#endif