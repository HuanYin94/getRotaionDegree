#if !defined MATCHER
#define MATCHER

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;

class RobustMatcher 
{
	private:
		// pointer to the feature point detector object
		cv::Ptr<cv::FeatureDetector> detector;
		// pointer to the feature descriptor extractor object
		cv::Ptr<cv::DescriptorExtractor> extractor;
		float ratio; // max ratio between 1st and 2nd NN
		bool refineF; // if true will refine the F matrix
		double distance; // min distance to epipolar
		double confidence; // confidence level (probability)

	public:
		vector<cv::Point2f> points1, points2;	
		RobustMatcher() : ratio(0.65f), refineF(true), confidence(0.99), distance(3.0) 
		{	  
			// SURF is the default feature
			detector= new cv::SurfFeatureDetector();
			extractor= new cv::SurfDescriptorExtractor();
		}

		// Set the feature detector
		void setFeatureDetector(cv::Ptr<cv::FeatureDetector>& detect)
		{
		  	detector= detect;
		}

		// Set descriptor extractor
		void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>& desc) 
		{
		  	extractor= desc;
		}

		// Set the minimum distance to epipolar in RANSAC
		void setMinDistanceToEpipolar(double d) 
		{
		  	distance= d;
		}

		// Set confidence level in RANSAC
		void setConfidenceLevel(double c) 
		{
		  	confidence= c;
		}

		// Set the NN ratio
		void setRatio(float r) 
		{
		  	ratio= r;
		}

		// if you want the F matrix to be recalculated
		void refineFundamental(bool flag) 
		{
		  	refineF= flag;
		}

		// Clear matches for which NN ratio is > than threshold
		// return the number of removed points 
		// (corresponding entries being cleared, i.e. size will be 0)
		int ratioTest(vector<vector<cv::DMatch> >& matches) 
		{
			int removed=0;
			// for all matches
			for (vector<vector<cv::DMatch> >::iterator matchIterator= matches.begin();
			 	matchIterator!= matches.end(); ++matchIterator) 
			{
				// if 2 NN has been identified
				if (matchIterator->size() > 1) 
				{
					// check distance ratio
					if ((*matchIterator)[0].distance/(*matchIterator)[1].distance > ratio) 
					{
						matchIterator->clear(); // remove match
						removed++;
					}
				} 
				else 
				{ 
					// does not have 2 neighbours
					matchIterator->clear(); // remove match
					removed++;
				}
			}
			return removed;
		}

		// Insert symmetrical matches in symMatches vector
		void symmetryTest(const vector<vector<cv::DMatch> >& matches1,
		                const vector<vector<cv::DMatch> >& matches2,
					    vector<cv::DMatch>& symMatches) 
		{
			// for all matches image 1 -> image 2
			for (vector<vector<cv::DMatch> >::const_iterator matchIterator1= matches1.begin();
				 matchIterator1!= matches1.end(); ++matchIterator1) 
			{

				if (matchIterator1->size() < 2) // ignore deleted matches 
					continue;

				// for all matches image 2 -> image 1
				for (vector<vector<cv::DMatch> >::const_iterator matchIterator2= matches2.begin();
					matchIterator2!= matches2.end(); ++matchIterator2) 
				{

					if (matchIterator2->size() < 2) // ignore deleted matches 
						continue;

					// Match symmetry test
					if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx  && 
						(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) 
					{

						// add symmetrical match
						symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
										  				(*matchIterator1)[0].trainIdx,
														(*matchIterator1)[0].distance));
						break; // next match in image 1 -> image 2
					}
				}
			}
		}

		// Identify good matches using RANSAC
		// Return fundemental matrix
		cv::Mat ransacTest(const vector<cv::DMatch>& matches,
						const vector<cv::KeyPoint>& keypoints1, 
						const vector<cv::KeyPoint>& keypoints2,
						vector<cv::DMatch>& outMatches) 
		{
			// Convert keypoints into Point2f	
			for (vector<cv::DMatch>::const_iterator it= matches.begin();
				 it!= matches.end(); ++it) 
			{
				 // Get the position of left keypoints
				 float x= keypoints1[it->queryIdx].pt.x;
				 float y= keypoints1[it->queryIdx].pt.y;
				 points1.push_back(cv::Point2f(x,y));
				 // Get the position of right keypoints
				 x= keypoints2[it->trainIdx].pt.x;
				 y= keypoints2[it->trainIdx].pt.y;
				 points2.push_back(cv::Point2f(x,y));
			}

			// Compute F matrix using RANSAC
			vector<uchar> inliers(points1.size(),0);
			cv::Mat fundemental= cv::findFundamentalMat(
				cv::Mat(points1),cv::Mat(points2), // matching points
			    inliers,      // match status (inlier ou outlier)  
			    CV_FM_RANSAC, // RANSAC method
			    distance,     // distance to epipolar line
			    confidence);  // confidence probability

			// extract the surviving (inliers) matches
			vector<uchar>::const_iterator itIn= inliers.begin();
			vector<cv::DMatch>::const_iterator itM= matches.begin();
			// for all matches
			for ( ;itIn!= inliers.end(); ++itIn, ++itM) 
			{
				if (*itIn) 
				{ 
					// it is a valid match
					outMatches.push_back(*itM);
				}
			}

			cout << "Number of matched points (after RANSAC): " << outMatches.size() << endl;
			//cout<<"F1:"<<fundemental<<endl;

			if (refineF) 
			{
			// The F matrix will be recomputed with all accepted matches
				// Convert keypoints into Point2f for final F computation	
				points1.clear();
				points2.clear();
				for (vector<cv::DMatch>::const_iterator it= outMatches.begin();
					 it!= outMatches.end(); ++it) 
				{
					 // Get the position of left keypoints
					 float x= keypoints1[it->queryIdx].pt.x;
					 float y= keypoints1[it->queryIdx].pt.y;
					 points1.push_back(cv::Point2f(x,y));
					 // Get the position of right keypoints
					 x= keypoints2[it->trainIdx].pt.x;
					 y= keypoints2[it->trainIdx].pt.y;
					 points2.push_back(cv::Point2f(x,y));
					 //cout<<"sssssssssssssssssssssss"<<endl;
				}
				// Compute 8-point F from all accepted matches
				fundemental= cv::findFundamentalMat(
					cv::Mat(points1),cv::Mat(points2), // matching points
					CV_FM_8POINT); // 8-point method
			}
			//cout<<"F2:"<<fundemental<<endl;
			return fundemental;
		}

		// Match feature points using symmetry test and RANSAC
		// returns fundemental matrix
		cv::Mat match(cv::Mat& image1, cv::Mat& image2, // input images 
		  			vector<cv::DMatch>& matches, // output matches and keypoints
		  			vector<cv::KeyPoint>& keypoints1, vector<cv::KeyPoint>& keypoints2) 
		{
			time_t start[6],end[6];

			start[0] = clock();
			// 1a. Detection of the SURF features
			detector->detect(image1,keypoints1);
			detector->detect(image2,keypoints2);
			end[0] = clock();
			cout<<"time0: "<<(double)(end[0]-start[0])/CLOCKS_PER_SEC<<endl;

			//cout << "Number of SURF points (1): " << keypoints1.size() << endl;
			//cout << "Number of SURF points (2): " << keypoints2.size() << endl;
			
			start[1] = clock();	
			// 1b. Extraction of the SURF descriptors
			cv::Mat descriptors1, descriptors2;
			extractor->compute(image1,keypoints1,descriptors1);
			extractor->compute(image2,keypoints2,descriptors2);
			end[1] = clock();
			cout<<"time1: "<<(double)(end[1]-start[1])/CLOCKS_PER_SEC<<endl;

			// cout << "descriptors1 matrix size: " << descriptors1.rows << " by " << descriptors1.cols << endl;
			// cout << "descriptors2 matrix size: " << descriptors2.rows << " by " << descriptors2.cols << endl;

			// 2. Match the two image descriptors
			start[2] = clock();
			// Construction of the matcher 
			cv::BruteForceMatcher<cv::L2<float> > matcher;

			// from image 1 to image 2
			// based on k nearest neighbours (with k=2)
			vector<vector<cv::DMatch> > matches1;
			matcher.knnMatch(descriptors1,descriptors2, 
				matches1, // vector of matches (up to 2 per entry) 
				2);		  // return 2 nearest neighbours

			// from image 2 to image 1
			// based on k nearest neighbours (with k=2)
			vector<vector<cv::DMatch> > matches2;
			matcher.knnMatch(descriptors2,descriptors1, 
				matches2, // vector of matches (up to 2 per entry) 
				2);		  // return 2 nearest neighbours
			end[2] = clock();
			cout<<"time1: "<<(double)(end[2]-start[2])/CLOCKS_PER_SEC<<endl;
			// cout << "Number of matched points 1->2: " << matches1.size() << endl;
			// cout << "Number of matched points 2->1: " << matches2.size() << endl;

			// 3. Remove matches for which NN ratio is > than threshold
			start[3] = clock();
			// clean image 1 -> image 2 matches
			int removed= ratioTest(matches1);
			// cout << "Number of matched points 1->2 (ratio test) : " << matches1.size()-removed << endl;
			// clean image 2 -> image 1 matches
			removed= ratioTest(matches2);
			// cout << "Number of matched points 2->1 (ratio test) : " << matches2.size()-removed << endl;
			end[3] = clock();
			cout<<"time3: "<<(double)(end[3]-start[3])/CLOCKS_PER_SEC<<endl;

			// 4. Remove non-symmetrical matches
			start[4] = clock();
			vector<cv::DMatch> symMatches;
			symmetryTest(matches1,matches2,symMatches);
			end[4] = clock();
			cout<<"time4: "<<(double)(end[4]-start[4])/CLOCKS_PER_SEC<<endl;
			// cout << "Number of matched points (symmetry test): " << symMatches.size() << endl;

			start[5] = clock();
			// 5. Validate matches using RANSAC
			cv::Mat fundemental= ransacTest(symMatches, keypoints1, keypoints2, matches);
			end[5] = clock();
			cout<<"time5: "<<(double)(end[5]-start[5])/CLOCKS_PER_SEC<<endl;
			// return the found fundemental matrix
			return fundemental;
		}
};

#endif
