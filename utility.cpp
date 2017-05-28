/* utility.cpp */
#include "utility.h"

#include <fstream>
#include <math.h>
#include <assert.h>

Mat solveLSE(Mat A) {
	Mat U, D, Vt;
	SVD::compute(A, D, U, Vt, SVD::FULL_UV);
	return Vt.row(Vt.rows - 1);
}

double vectorNorm(Mat A) {
	double norm = 0;
	if (A.rows > A.cols) {
		for (int i = 0; i < A.rows; ++i)
			norm += pow(A.at<double>(i,0), 2);
	}
	else {
		for (int j = 0; j < A.cols; ++j)
			norm += pow(A.at<double>(0, j), 2);
	}
	return sqrt(norm);
}

Mat normalizeCvMatRow(Mat A) {
	int m = A.rows;
	int n = A.cols;

	for (int i = 0; i < m; ++i) {
		double norm = 0;
		for (int j = 0; j < n; ++j) {
			norm += pow(A.at<double>(i, j), 2);
			if (j == n - 1) {
				for (int k = 0; k < n; k++)
					A.at<double>(i, k) = A.at<double>(i, k) / sqrt(norm);
			}
		}
	}
	return A;
}

void exportPointCloud(Vec3DPoint worldPoints, Mat colorRefImg, vector<DMatch> matches, vector<KeyPoint> refKeypoints) {
	cout << "Exporting 3D Points..." << endl;
	Vector<Vec3b> pointColor;
	ofstream outfile("3Dpoints.csv", ios::trunc | ios::out);

	for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); ++it) {
		int j = (int)refKeypoints[it->trainIdx].pt.x;
		int i = (int)refKeypoints[it->trainIdx].pt.y;
		pointColor.push_back(colorRefImg.at<Vec3b>(i, j));
	}

	if (outfile.is_open()) {
		for (Vec3DPoint::iterator it = worldPoints.begin(); it != worldPoints.end(); ++it) {
			outfile << it->x << "," << it->y << "," << it->z << "," <<
				(int)pointColor[it - worldPoints.begin()][0] << "," << (int)pointColor[it - worldPoints.begin()][1] << "," << (int)pointColor[it - worldPoints.begin()][2] << endl;
		}
		outfile.close();
	}
	else cout << "Unable to open file";
}

VecMat getRowsFromMat(Mat inMat) {
	VecMat outVec;
	for (int i = 0; i < inMat.rows; ++i)
		outVec.push_back(inMat.row(i));
	return outVec;
}

Mat getRowsFromMat(Mat inMat, int from, int to) {
	Mat outMat = Mat::zeros(to - from + 1, inMat.cols, CV_64FC1);
	int k = 0;
	for (int i = from - 1; i < to; ++i, ++k)
		inMat.row(i).copyTo(outMat.row(k));
	return outMat;
}

VecMat getColsFromMat(Mat inMat) {
	VecMat outVec;
	for (int j = 0; j < inMat.cols; ++j)
		outVec.push_back(inMat.col(j));
	return outVec;
}

Mat getColsFromMat(Mat inMat, int from, int to) {
	Mat outMat = Mat::zeros(inMat.rows, to - from + 1, CV_64FC1);
	int k = 0;
	for (int j = from - 1; j < to; ++j, ++k)
		inMat.col(j).copyTo(outMat.col(k));
	return outMat;
}

Vec2DPoint matPointToVecPoint(Mat matPoint) {
	Vec2DPoint point2DVec;
	Point2d point;
	for (int j = 0; j < matPoint.cols; point2DVec.push_back(point), ++j) {
		point.x = matPoint.at<double>(0, j);
		point.y = matPoint.at<double>(1, j);
	}
	return point2DVec;
}

Mat vecPointToMatPoint(Vec3DPoint vecpoint) {
	Mat matPoint = Mat::zeros(3, vecpoint.size(), CV_64FC1);
	for (int i = 0; i < vecpoint.size(); i++) {
		matPoint.at<double>(0, i) = vecpoint[i].x;
		matPoint.at<double>(1, i) = vecpoint[i].y;
		matPoint.at<double>(2, i) = vecpoint[i].z;
	}
	return matPoint;
}

Mat reshape(Mat inMat, int newNumRows, int newNumCols) {
	assert(inMat.rows == 1 || inMat.cols == 1);
	Mat outMat = Mat::zeros(newNumRows, newNumCols, CV_64FC1);
	if (inMat.rows > inMat.cols) {
		int elemNum = 0;
		for (int i = 0; i < newNumRows; ++i) {
			for (int j = 0; j < newNumCols; ++j) {
				outMat.at<double>(i, j) = inMat.at<double>(elemNum, 0);
				++elemNum;
			}
		}
	}
	else
	{
		int elemNum = 0;
		for (int i = 0; i < newNumRows; ++i) {
			for (int j = 0; j < newNumCols; ++j) {
				outMat.at<double>(i, j) = inMat.at<double>(0, elemNum);
				++elemNum;
			}
		}
	}
	return outMat;
}

Mat getNormalizationMatrix(Mat point,int nPoints) {
	assert(point.rows == 3);
	double sumX = 0, sumY = 0;

	for (int i = 0; i < nPoints; ++i) {
		sumX += point.at<double>(0, i);
		sumY += point.at<double>(1, i);
	}
	double meanX = sumX / (double)nPoints;
	double meanY = sumY / (double)nPoints;

	double distSum = 0;
	for (int i = 0; i<nPoints; i++) {
		double newX = point.at<double>(0, i) - meanX;
		double newY = point.at<double>(1, i) - meanY;
		distSum += sqrt(newX*newX + newY*newY);
	}
	double meanDist = distSum / nPoints;
	double scale = sqrt(2.0) / meanDist;

	return (Mat_<double>(3, 3) << scale, 0, -scale*meanX, 0, scale, -scale*meanY, 0, 0, 1);
}