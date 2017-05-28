/* sfm.cpp */
#include "sfm.h"
#include "utility.h"

#include <math.h>

inline double distPointLine(Mat point, Mat line) {
	Mat tmp = point.t()*line;
	double prod = tmp.at<double>(0, 0);
	double d = abs(prod) / sqrt(pow(line.at<double>(0, 0), 2) + pow(line.at<double>(1, 0), 2));
	return d;
}

Mat getFundamentalMatrix(Mat keypointsL, Mat keypointsR, int nPoints) { // 8-point
	Mat TnormL = getNormalizationMatrix(keypointsL, nPoints);
	Mat TnormR = getNormalizationMatrix(keypointsR, nPoints);
	Mat pntsNormL = TnormL*keypointsL;
	Mat pntsNormR = TnormR*keypointsR;
	Mat A = Mat::zeros(nPoints, 9, CV_64FC1);
	Mat U, D, Vt;

	for (int i = 0; i < nPoints; ++i) {
		A.at<double>(i, 0) = pntsNormR.at<double>(0, i)*pntsNormL.at<double>(0, i);
		A.at<double>(i, 1) = pntsNormR.at<double>(0, i)*pntsNormL.at<double>(1, i);
		A.at<double>(i, 2) = pntsNormR.at<double>(0, i);
		A.at<double>(i, 3) = pntsNormR.at<double>(1, i)*pntsNormL.at<double>(0, i);
		A.at<double>(i, 4) = pntsNormR.at<double>(1, i)*pntsNormL.at<double>(1, i);
		A.at<double>(i, 5) = pntsNormR.at<double>(1, i);
		A.at<double>(i, 6) = pntsNormL.at<double>(0, i);
		A.at<double>(i, 7) = pntsNormL.at<double>(1, i);
		A.at<double>(i, 8) = 1;
	}

	SVD::compute(A, D, U, Vt, SVD::FULL_UV);

	Mat Fnorm = Mat::zeros(3, 3, CV_64FC1);
	Fnorm = reshape(solveLSE(A), 3, 3);

	SVD::compute(Fnorm, D, U, Vt);
	D.at<double>(2, 0) = 0;
	Mat Fprime = U*Mat::diag(D)*Vt;
	Mat F = TnormR.t()*Fprime*TnormL;

	return F;
}

Mat getEssentialMatrix(Mat F, Mat leftK, Mat rightK) {
	Mat E;
	return E = rightK.t()*F*leftK;
}

VecMat getCandidateP(Mat E) {
	Mat U, S, Vt;
	Mat W = (Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
	SVD::compute(E, S, U, Vt, SVD::FULL_UV);
	VecMat cddP;

	cddP.push_back(U*W*Vt);
	hconcat(cddP[0], U.col(2), cddP[0]);

	cddP.push_back(U*W.t()*Vt);
	hconcat(cddP[1], U.col(2), cddP[1]);

	cddP.push_back(U*W*Vt);
	hconcat(cddP[2], -U.col(2), cddP[2]);

	cddP.push_back(U*W.t()*Vt);
	hconcat(cddP[3], -U.col(2), cddP[3]);

	return cddP;
}

VecMat getProjectionMatrix(VecMat cddP, Vec2DPoint pboxL, Vec2DPoint pboxR, Mat leftK, Mat rightK) {
	VecMat pairP;
	Mat rightP = Mat::zeros(3, 4, CV_64FC1);
	Mat leftP = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);

	int k = 0;
	
	for (VecMat::iterator it = cddP.begin(); it != cddP.end(); ++it) {
		double pts4D[4]; //[X; Y; Z; T] 
		Mat rightPtmp = *it;
		Mat S, U, Vt;
		Mat A = Mat::zeros(4, 4, CV_64FC1);
		Mat tmp = Mat::zeros(1, 4, CV_64FC1);			
		int idx = rand() % pboxL.size(); //take a test point correspondence

		tmp = leftP.row(2)*pboxL[idx].x - leftP.row(0);
		tmp.copyTo(A.row(0));
		tmp = leftP.row(2)*pboxL[idx].y - leftP.row(1);
		tmp.copyTo(A.row(1));
		tmp = rightPtmp.row(2)*pboxR[idx].x - rightPtmp.row(0);
		tmp.copyTo(A.row(2));
		tmp = rightPtmp.row(2)*pboxR[idx].y - rightPtmp.row(1);
		tmp.copyTo(A.row(3));
		A = normalizeCvMatRow(A);

		SVD::compute(A, S, U, Vt);

		for (int j = 0; j < 4; j++)
			pts4D[j] = Vt.at<double>(3, j);
		
		double depthR = getDepth(rightPtmp, pts4D);
		double depthL = getDepth(leftP, pts4D);

		cout << depthL << " " << depthR << endl;
		if (depthR > 0 && depthL > 0) {
			++k;
			rightPtmp.copyTo(rightP);
		}
	}

	if (k > 1) {
		cout << "More than 1 right Camera Matrix satisfy. Something went wrong!!!" << endl;
		exit(1);
	}
	//cout << rightP << endl;
	Mat Pleft = leftK*leftP;
	pairP.push_back(Pleft);

	Mat Pright = rightK*rightP;
	pairP.push_back(Pright);

	return pairP;
}

double getDepth(Mat P, double* point4D) {
	Mat p4D = Mat::zeros(4, 1, CV_64FC1);
	Mat M = getColsFromMat(P, 1, 3);
	Mat p3 = getRowsFromMat(P, 3, 3);
	Mat	m3 = getRowsFromMat(M, 3, 3);
	Mat res = Mat::zeros(1, 1, CV_64FC1);
	double w;
	double d;

	for (int i = 0; i < 4; ++i)
		p4D.at<double>(i, 0) = point4D[i];

	res = p3*p4D;
	w = res.at<double>(0, 0);

	d = sign(determinant(M))*w / (point4D[3] * vectorNorm(m3));

	return d;
}

Vec3DPoint triangulatePoints(Vec2DPoint matchedPointsL, Vec2DPoint matchedPointsR, int nPoints, Mat leftP, Mat rightP) { // Linear triangulation
	Vec3DPoint points3D;
	for (int k = 0; k < nPoints; k++) { // npoints : # of matching
		double pts4D[4]; //[X; Y; Z; T] 
		double xL, yL, xR, yR;
		Mat A = Mat::zeros(4, 4, CV_64FC1);
		Mat tmp = Mat::zeros(1, 4, CV_64FC1);			//take a test point correspondence
		Mat S, U, Vt;
		Point3d point;

		xL = matchedPointsL[k].x;
		yL = matchedPointsL[k].y;
		xR = matchedPointsR[k].x;
		yR = matchedPointsR[k].y;

		tmp = leftP.row(2)*xL - leftP.row(0);
		tmp.copyTo(A.row(0));
		tmp = leftP.row(2)*yL - leftP.row(1);
		tmp.copyTo(A.row(1));
		tmp = rightP.row(2)*xR - rightP.row(0);
		tmp.copyTo(A.row(2));
		tmp = rightP.row(2)*yR - rightP.row(1);
		tmp.copyTo(A.row(3));
		A = normalizeCvMatRow(A);

		SVD::compute(A, S, U, Vt);

		for (int j = 0; j < 4; j++)
			pts4D[j] = Vt.at<double>(3, j);

		point.x = pts4D[0] / pts4D[3];
		point.y = pts4D[1] / pts4D[3];
		point.z = pts4D[2] / pts4D[3];

		points3D.push_back(point);
	}

	return points3D;
}

Mat getFundamentalMatrixRansac(Mat keypointsL, Mat keypointsR, int nPoints, double thres) {
	int nSamples = 8;
	double p = 0.99;
	int N = 1e10; //N = infinity
	int sample_count = 0;
	int maxInliers = 0;
	double probInliers, probNoOutliers;
	Mat F;

	Mat TnormL = getNormalizationMatrix(keypointsL, nPoints);
	Mat TnormR = getNormalizationMatrix(keypointsR, nPoints);
	Mat pntsNormL = TnormL*keypointsL;
	Mat pntsNormR = TnormR*keypointsR;
	int* curInliersIdx = new int[nPoints];
	int* bestInliersIdx = new int[nPoints];
	int inliersCnt;
	int bestInliersCnt = 0;

	while (N > sample_count) {
		Mat pntsNormLtmp = Mat::zeros(3, nSamples, CV_64FC1);
		Mat pntsNormRtmp = Mat::zeros(3, nSamples, CV_64FC1);

		for (int j = 0; j < nSamples; ++j) {
			int idx = rand() % nPoints;
			for (int i = 0; i < 3; ++i) {
				pntsNormLtmp.at<double>(i, j) = pntsNormL.at<double>(i, idx);
				pntsNormRtmp.at<double>(i, j) = pntsNormR.at<double>(i, idx);
			}
		}

		Mat Ftmp = getFundamentalMatrix(pntsNormLtmp, pntsNormRtmp, nSamples);

		inliersCnt = 0;
		for (int i = 0; i < nPoints; ++i) {
			Mat lPrime = Ftmp*pntsNormL.col(i);
			double d = distPointLine(pntsNormR.col(i), lPrime);
			if (d < thres){
				curInliersIdx[inliersCnt] = i;
				inliersCnt++;
			}
		}
	
		if (inliersCnt > bestInliersCnt) {
			bestInliersCnt = inliersCnt;
			Ftmp.copyTo(F);

			for (int k = 0; k<inliersCnt; k++)
				bestInliersIdx[k] = curInliersIdx[k];

			probInliers = (double)inliersCnt / (double)nPoints;
			probNoOutliers = 1 - pow(probInliers, nSamples);
			N = (int)(log(1.0 - p) / log(probNoOutliers));
		}

		sample_count++;
	}

	delete[] curInliersIdx;
	delete[] bestInliersIdx;

	return TnormR.t()*F*TnormL;
}

VecMat getTempProjectionMatrix(Mat F) {
	Mat P = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
	Mat e = solveLSE(F.t());
	Mat ex = (Mat_<double>(3, 3) << 0, -e.at<double>(0, 2), e.at<double>(0, 1),
		e.at<double>(0, 2), 0, -e.at<double>(0, 0),
		-e.at<double>(0, 1), e.at<double>(0, 0), 0);
	Mat S = ex * F;
	Mat Pprime;
	hconcat(S, e.t(), Pprime);
	VecMat pairP;
	pairP.push_back(P);
	pairP.push_back(Pprime);
	return pairP;
}

//void getFundamentalMatrixRansacLM(Mat keypointsL, Mat keypointsR, int nPoints, double thres) {
//	Mat Ftmp = getFundamentalMatrixRansac(keypointsL, keypointsR, nPoints, thres);
//	VecMat pairP = getTempProjectionMatrix();
//	Mat TnormL = getNormalizationMatrix(keypointsL, nPoints);
//	Mat TnormR = getNormalizationMatrix(keypointsR, nPoints);
//	Mat pntsNormL = TnormL*keypointsL;
//	Mat pntsNormR = TnormR*keypointsR;
//	int* curInliersIdx = new int[nPoints];
//	int inliersCnt = 0;
//
//	for (int i = 0; i < nPoints; ++i) {
//		Mat lPrime = Ftmp*pntsNormL.col(i);
//		double d = distPointLine(pntsNormR.col(i), lPrime);
//		if (d < thres){
//			curInliersIdx[inliersCnt] = i;
//			inliersCnt++;
//		}
//	}
//
//	Mat inlierRef = Mat::zeros(3, inliersCnt, CV_64FC1);
//	Mat inlierPrime = Mat::zeros(3, inliersCnt, CV_64FC1);
//	for (int i = 0; i < inliersCnt; i++) {
//		keypointsL.col(curInliersIdx[i]).copyTo(inlierRef.col(i));
//		keypointsR.col(curInliersIdx[i]).copyTo(inlierPrime.col(i));
//	}
//	Vec2DPoint matchedInlierPointsRef = matPointToVecPoint(inlierRef);
//	Vec2DPoint matchedInlierPointsPrime = matPointToVecPoint(inlierPrime);
//	Vec3DPoint worldPointsRawVec = triangulatePoints(matchedInlierPointsRef, matchedInlierPointsPrime, nPoints, pairP[0], pairP[1]);
//	Mat worldPointsRaw = vecPointToMatPoint(worldPointsRawVec);
//	Mat xHat = pairP[0] * worldPointsRaw;
//	Mat xHatPrime = pairP[1] * worldPointsRaw;
//	for (int j = 0; j < inliersCnt; j++) {
//		for (int i = 0; i < 3; i++) {
//			xHat.at<double>(i, j) = xHat.at<double>(i, j) / xHat.at<double>(2, j);
//			xHatPrime.at<double>(i, j) = xHatPrime.at<double>(i, j) / xHatPrime.at<double>(2, j);
//		}
//	}
//
//	//double c = cost(inlierRef, xHat, inlierPrime, xHatPrime);
//	Mat theta = getRowsFromMat(worldPointsRaw, 1, 3).reshape(0, worldPointsRaw.cols * 3);
//	theta.push_back(pairP[1].reshape(0, 12));
//	alglib::real_1d_array x;
//	for (int i = 0; i < theta.rows; i++)
//		x(i) = theta.at<double>(i, 0);
//	double epsg = 0.0000000001;
//	double epsf = 0;
//	double epsx = 0;
//	alglib::ae_int_t maxits = 0;
//	alglib::minlbfgsstate state;
//	alglib::minlbfgsreport rep;
//
//	alglib::minlbfgscreate(1, x, state);
//	alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);
//	//alglib::minlbfgsoptimize(state, function_grad);
//
//	/*Mat grad = numericalGrad(theta, inlierRef, inlierPrime, cost);*/
//}
//
//Mat numericalGrad(Mat theta, Mat x, Mat xPrime, double(*func) (Mat, Mat, Mat, Mat)) {
//	double eps = 10e-5;
//	Mat P = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
//	Mat numgrad = Mat::zeros(theta.rows, 1, CV_64FC1);
//	for (int i = 0; i < theta.rows; i++) {
//		Mat thetaPlus = theta;
//		Mat thetaMinus = theta;
//		thetaPlus.at<double>(i, 0) = thetaPlus.at<double>(i, 0) + eps;
//		thetaMinus.at<double>(i, 0) = thetaMinus.at<double>(i, 0) - eps;
//
//		Mat PprimePlus = getRowsFromMat(thetaPlus, theta.rows - 11, theta.rows).reshape(0, 3);
//		Mat wPointsPlus = getRowsFromMat(thetaPlus, 1, theta.rows - 12).reshape(0, 3);
//		Mat unit = Mat::ones(1, theta.rows / 3 - 4, CV_64FC1);
//		wPointsPlus.push_back(unit);
//
//		Mat PprimeMinus = getRowsFromMat(thetaMinus, theta.rows - 11, theta.rows).reshape(0, 3);
//		Mat wPointsMinus = getRowsFromMat(thetaMinus, 1, theta.rows - 12).reshape(0, 3);
//		wPointsMinus.push_back(unit);
//
//		//double cPlus = cost(x, P*wPointsPlus, xPrime, PprimePlus*wPointsPlus);
//		//double cMinus = cost(x, P*wPointsMinus, xPrime, PprimeMinus*wPointsMinus);
//
//		//numgrad.at<double>(i, 0) = (cPlus - cMinus) / (2 * eps);
//	}
//
//	return numgrad;
//}