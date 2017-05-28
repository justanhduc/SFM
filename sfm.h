/* sfm.h */

#ifndef SFM_H_
#define SFM_H_

#include "../../../AlgLib/stdafx.h"
#include <iostream>
#include <vector>

#include <opencv\cv.h>
#include <opencv\cxcore.h>
#include <opencv\highgui.h>
#include "opencv2/opencv.hpp" 
#include "../../../AlgLib/optimization.h"

using namespace std;
using namespace cv;

#define VecMat			vector<Mat>
#define Vec3DPoint		vector<Point3d>
#define Vec2DPoint		vector<Point2d>

/************************************************************************/
/* two-view structure from motion                                       */
/************************************************************************/

Mat getFundamentalMatrix(Mat keypointsL, Mat keypointsR, int nPoints); //8-point estimation

Mat getFundamentalMatrixRansac(Mat keypointsL, Mat keypointsR, int nPoints,
	double thres);

Mat getEssentialMatrix(Mat F, Mat leftK, Mat rightK);

VecMat getCandidateP(Mat E);

VecMat getProjectionMatrix(VecMat cddP, Vec2DPoint pboxL, Vec2DPoint pboxR, Mat leftK, Mat rightK);

double getDepth(Mat P, double* point4D);

Vec3DPoint triangulatePoints(Vec2DPoint matchedPointsL, Vec2DPoint matchedPointsR, int nPoints, Mat leftP, Mat rightP);

void viewPointCloud(Vec3DPoint worldPoints);

VecMat getTempProjectionMatrix(Mat F);

/************************************************************************/
#endif