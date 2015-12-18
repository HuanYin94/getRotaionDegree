#include <opencv2/calib3d/calib3d.hpp>
#include "matcher.h"

using namespace std;

double CalcDepth(const cv::Mat &X, const cv::Mat &P)
{
    // back project
    cv::Mat X2 = P*X;

    double det = cv::determinant(P(cv::Range(0,3), cv::Range(0,3)));
    double w = X2.at<double>(2,0);
    double W = X.at<double>(3,0);

    double a = P.at<double>(0,2);
    double b = P.at<double>(1,2);
    double c = P.at<double>(2,2);

    double m3 = sqrt(a*a + b*b + c*c);  // 3rd column of M

    double sign;

    if(det > 0) {
        sign = 1;
    }
    else {
        sign = -1;
    }

    return (w/W)*(sign/m3);
}

cv::Mat TriangulatePoint(const cv::Point2d &pt1, const cv::Point2d &pt2, const cv::Mat &P1, const cv::Mat &P2)
{
    cv::Mat A(4,4,CV_64F);

    A.at<double>(0,0) = pt1.x*P1.at<double>(2,0) - P1.at<double>(0,0);
    A.at<double>(0,1) = pt1.x*P1.at<double>(2,1) - P1.at<double>(0,1);
    A.at<double>(0,2) = pt1.x*P1.at<double>(2,2) - P1.at<double>(0,2);
    A.at<double>(0,3) = pt1.x*P1.at<double>(2,3) - P1.at<double>(0,3);

    A.at<double>(1,0) = pt1.y*P1.at<double>(2,0) - P1.at<double>(1,0);
    A.at<double>(1,1) = pt1.y*P1.at<double>(2,1) - P1.at<double>(1,1);
    A.at<double>(1,2) = pt1.y*P1.at<double>(2,2) - P1.at<double>(1,2);
    A.at<double>(1,3) = pt1.y*P1.at<double>(2,3) - P1.at<double>(1,3);

    A.at<double>(2,0) = pt2.x*P2.at<double>(2,0) - P2.at<double>(0,0);
    A.at<double>(2,1) = pt2.x*P2.at<double>(2,1) - P2.at<double>(0,1);
    A.at<double>(2,2) = pt2.x*P2.at<double>(2,2) - P2.at<double>(0,2);
    A.at<double>(2,3) = pt2.x*P2.at<double>(2,3) - P2.at<double>(0,3);

    A.at<double>(3,0) = pt2.y*P2.at<double>(2,0) - P2.at<double>(1,0);
    A.at<double>(3,1) = pt2.y*P2.at<double>(2,1) - P2.at<double>(1,1);
    A.at<double>(3,2) = pt2.y*P2.at<double>(2,2) - P2.at<double>(1,2);
    A.at<double>(3,3) = pt2.y*P2.at<double>(2,3) - P2.at<double>(1,3);

    cv::SVD svd(A);

    cv::Mat X(4,1,CV_64F);

    X.at<double>(0,0) = svd.vt.at<double>(3,0);
    X.at<double>(1,0) = svd.vt.at<double>(3,1);
    X.at<double>(2,0) = svd.vt.at<double>(3,2);
    X.at<double>(3,0) = svd.vt.at<double>(3,3);

    return X;
}

int main(int argc,char *argv[])
{	
	//Define intrinsic matrix
	cv::Mat intrinsic = (cv::Mat_<double>(3,3) << 522.4825, 0, 300.9989, 
												0, 522.5723, 258.1389, 
												0, 0, 1);	
	//cv::Mat intrinsic_inverse = intrinsic.inv();
	//cout<<intrinsic<<endl;

	// Read input images
	string jpg1 = argv[1];
	jpg1.append(".jpg");
	string jpg2 = argv[2];
	jpg2.append(".jpg");
	cv::Mat image1 = cv::imread(jpg1,0);
	cv::Mat image2 = cv::imread(jpg2,0);
	if (!image1.data || !image2.data)
		return 0;

 //    // Display the images
	// cv::namedWindow("Image 1");
	// cv::imshow("Image 1",image1);
	// cv::namedWindow("Image 2");
	// cv::imshow("Image 2",image2);

	// Prepare the matcher
	RobustMatcher rmatcher;
	rmatcher.setConfidenceLevel(0.98);
	rmatcher.setMinDistanceToEpipolar(1.0);
	rmatcher.setRatio(0.65f);
	cv::Ptr<cv::FeatureDetector> pfd= new cv::SurfFeatureDetector(3000); 
	rmatcher.setFeatureDetector(pfd);

	// Match the two images
	vector<cv::DMatch> matches;
	vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat fundamental = rmatcher.match(image1,image2,matches, keypoints1, keypoints2);
	//cout<<"F:"<<endl<<fundamental<<endl;

	// //draw the matches
	// cv::Mat imageMatches;
	// cv::drawMatches(image1,keypoints1,  // 1st image and its keypoints
	// 	            image2,keypoints2,  // 2nd image and its keypoints
	// 				matches,			// the matches
	// 				imageMatches,		// the image produced
	// 				cv::Scalar(255,255,255)); // color of the lines
	// cv::namedWindow("Matches");
	// cv::imshow("Matches",imageMatches);

	// Calculate the essential matrix
	cv::Mat essential = intrinsic.t()*fundamental*intrinsic;
	//cout<<"E:"<<endl<<essential<<endl;

	// Singular value decomposition, U, W, Vt, Z
	cv::SVD decomp = cv::SVD(essential);
	cv::Mat U = decomp.u;
	if(cv::determinant(decomp.u) < 0)
		U = -U;
	cv::Mat Vt = decomp.vt;
	if(cv::determinant(decomp.vt) < 0)
		Vt = -Vt;
	cv::Mat W = (cv::Mat_<double>(3,3) << 0,-1, 0, 
										  1, 0, 0, 
										  0, 0, 1);

	// // Calculate the rotation matrix
	// cv::Mat R1 = U*W.t()*Vt;
	// cv::Mat R2 = U*W*Vt;

	// //Calculate Euler angle theta
	// double theta1 = -1 * asin(R1.at<double>(2,0))/CV_PI*180;
	// double theta2 = -1 * asin(R2.at<double>(2,0))/CV_PI*180;


	// //Calculate Euler angle Psi
	// double psi1 = atan2(R1.at<double>(2,1)/cos(theta1),R1.at<double>(2,2)/cos(theta1));
	// double psi2 = atan2(R2.at<double>(2,1)/cos(theta2),R2.at<double>(2,2)/cos(theta2));

	// //Calculate Euler angle Phi
	// double phi1 = atan2(R1.at<double>(1,0)/cos(theta1),R1.at<double>(0,0)/cos(theta1));
	// double phi2 = atan2(R2.at<double>(1,0)/cos(theta2),R2.at<double>(0,0)/cos(theta2));


	// cout<<"R1:  "<<theta1<<endl;	
	// cout<<"R2:  "<<theta2<<endl;

	// Initialize P1, P2, P3, P4
    cv::Mat P1, P2, P3, P4;
    P1 = cv::Mat::eye(3,4,CV_64F);
    P2 = cv::Mat::eye(3,4,CV_64F);
    P3 = cv::Mat::eye(3,4,CV_64F);
    P4 = cv::Mat::eye(3,4,CV_64F);

    // Rotation
    P1(cv::Range(0,3), cv::Range(0,3)) = U*W*Vt;
    P2(cv::Range(0,3), cv::Range(0,3)) = U*W*Vt;
    P3(cv::Range(0,3), cv::Range(0,3)) = U*W.t()*Vt;
    P4(cv::Range(0,3), cv::Range(0,3)) = U*W.t()*Vt;

    // Translation
    P1(cv::Range::all(), cv::Range(3,4)) = U(cv::Range::all(), cv::Range(2,3))*1;
    P2(cv::Range::all(), cv::Range(3,4)) = -U(cv::Range::all(), cv::Range(2,3));
    P3(cv::Range::all(), cv::Range(3,4)) = U(cv::Range::all(), cv::Range(2,3))*1;
    P4(cv::Range::all(), cv::Range(3,4)) = -U(cv::Range::all(), cv::Range(2,3));

//    cv::Mat P1 = (cv::Mat_<double>(3,4) << -0.97349008, -0.04968834,  0.22326697,  0.15707857,
//            -0.02015803, -0.95368323, -0.30013655, -0.15161232,
//            0.22783925, -0.29668058,  0.92739954,  0.9758791 );
//    cv::Mat P2 = (cv::Mat_<double>(3,4) << -0.97349008, -0.04968834,  0.22326697, -0.15707857,
//            -0.02015803, -0.95368323, -0.30013655,  0.15161232,
//            0.22783925, -0.29668058,  0.92739954, -0.9758791 );
//    cv::Mat P3 = (cv::Mat_<double>(3,4) << 0.99626186,  0.00170422,  0.08636778,  0.15707857,
//            -0.00182131,  0.99999753,  0.00127692, -0.15161232,
//            -0.08636539, -0.00142945,  0.9962625 ,  0.9758791 );
//    cv::Mat P4 = (cv::Mat_<double>(3,4) << 0.99626186,  0.00170422,  0.08636778, -0.15707857,
//            -0.00182131,  0.99999753,  0.00127692,  0.15161232,
//            -0.08636539, -0.00142945,  0.9962625 , -0.9758791);

    cv::Mat P[4] = {P1, P2, P3, P4};
    // Test to see if this E matrix is the correct one
    cv::Mat P_ref = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat best_E, best_P;
    int num_pts = matches.size();
    int best_inliers = 0;
    bool found = false;
    cv::Mat pt3d;

    for(int j=0; j < 4; j++) {
        pt3d = TriangulatePoint(rmatcher.points1[0], rmatcher.points2[0], P_ref, P[j]);
        double depth1 = CalcDepth(pt3d, P_ref);
        double depth2 = CalcDepth(pt3d, P[j]);

        if(depth1 > 0 && depth2 > 0){
            int inliers = 1; // number of points in front of the camera

            for(int k=1; k < num_pts; k++) {
                pt3d = TriangulatePoint(rmatcher.points1[k], rmatcher.points2[k], P_ref, P[j]);
                depth1 = CalcDepth(pt3d, P_ref);
                depth2 = CalcDepth(pt3d, P[j]);

                if(depth1 > 0 && depth2 > 0) {
                    inliers++;
                }
            }
            if(inliers > best_inliers && inliers >= 5) {
                best_inliers = inliers;
                essential.copyTo(best_E);
                P[j].copyTo(best_P);
                found = true;
            }
            // Special case, with 5 points you can get a perfect solution
            if(num_pts == 5 && inliers == 5) {
                break;
            }
        }
    }

    double theta = atan2(best_P.at<double>(0,2), best_P.at<double>(2,2)) / CV_PI * 180;


    cout << "Rotation angle: " << theta << endl;

	//cv::waitKey();
	return 0;
}
