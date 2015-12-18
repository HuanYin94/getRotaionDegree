import os
import sys
import cv2
import numpy as np
import math

def PfromE(E):
    
    U, D, VT = np.linalg.svd(E)

    Z = [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]
    W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    if np.linalg.det(U) < 0:
        U = -U 
    if np.linalg.det(VT) < 0:
        VT = -VT

    P1 = U*W*VT
    P1 = np.hstack((P1, U[:, 2]))
    P2 = U*W*VT 
    P2 = np.hstack((P2, -U[:, 2]))
  
    WT = map(list, zip(*W))
    P3 = U*WT*VT
    P3 = np.hstack((P3, U[:, 2]))
    P4 = U*WT*VT
    P4 = np.hstack((P4, -U[:, 2]))

    return(P1, P2, P3, P4)

def visualOdometry(img1, img2):
    # sift = cv2.SURF()

    detector = cv2.FeatureDetector_create("FAST")    # SURF, FAST, SIFT
    descriptor = cv2.DescriptorExtractor_create("SURF") # SURF, SIFT

    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)

    kp1 = detector.detect(img1)
    kp2 = detector.detect(img2) 

    k1, des1 = descriptor.compute(img1,kp1)
    k2, des2 = descriptor.compute(img2,kp2)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    good = []

    # Apply ratio test
    for m,n in matches:
        if m.distance < 0.7*n.distance:
                good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    src_pts = src_pts[:, 0, :]        # adjust the dimensions of two feature lists
    dst_pts = dst_pts[:, 0, :]        

    v = np.ones((len(src_pts), 1))

    src_pts = np.c_[src_pts, v]    # and express them in homogeneous coordinates
    dst_pts = np.c_[dst_pts, v] 

    E = K.T*F*K 
    P = PfromE(E)

    q = src_pts[0,:]
    qp = dst_pts[0,:]

    E = np.array(E)
 
    foundP = False
    foundI = None
    for i in range(4):
        # Pick the correct P according to Nister's five-ponit algorithm
        Pa = P[:][:][i]

        a = np.dot(E.T, qp)
        b = np.cross(q, np.dot(np.diag([1,1,0]), a))
        c = np.cross(qp, np.dot(np.dot(np.diag([1,1,0]), E), q))
        d = np.cross(a, b)
        C = np.array(np.dot(Pa.T, c).T)
        C = C[:,0]
        Q = np.append(d.T*C[3], -(d[0]*C[0]+d[1]*C[1]+d[2]*C[2]))

        # Test the solution
 
        c1 = Q[2]*Q[3]
        t = np.array(np.dot(Pa, Q))
        t = t[0,:]
        c2 = t[2]*Q[3]
        if c1 > 0 and c2 > 0:
            foundP = True
            foundI = i

        if foundP:
            Pa = P[:][:][foundI]
            th = math.atan2(Pa[0, 2], Pa[2, 2]) # Rotation angle
        else:             
            th = None

    return th, Pa

# Main Function
if __name__ == '__main__':
    K = np.matrix([[522.4825, 0,        300.9989], 
                   [0,        522.5723, 258.1389], 
                   [0.0,      0.0,      1.0]])

img_1 = cv2.imread(sys.argv[1] + ".jpg")
img_2 = cv2.imread(sys.argv[2] + ".jpg")
rot, R = visualOdometry(img_1, img_2)
print "Rotation angle: ", rot * 180 / cv2.cv.CV_PI
print "Pa:\n", R