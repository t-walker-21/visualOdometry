"""

Class to implement feature detection for image
"""

import cv2
import numpy
import argparse


class FeatureExtractor(object):
    def __init__(self):
        pass

    def detect_features(self, image, feature='surf'):
        """

        Function to compute features given an input image
        """


        # Get a gray image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_image = cv2.resize(gray_image, (500,500))

        image = cv2.resize(image, (500, 500))

        if feature == "orb":
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray_image, None)

        elif feature == "sift":
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        
        elif feature == "surf":
            surf = cv2.xfeatures2d.SURF_create()
            keypoints, descriptors = surf.detectAndCompute(gray_image, None)


        kps_image = cv2.drawKeypoints(image, keypoints, None)

        cv2.imshow("keypoints", kps_image)
        cv2.waitKey(0)

        return keypoints, descriptors    

    
    def feature_matcher(self, image_one, image_two, features="surf", crossCheck=True):
        """

        Match features in two images
        """


        # Get features and keypoints

        keypoints_1, descriptors_1 = self.detect_features(image_one, feature=features)

        keypoints_2, descriptors_2 = self.detect_features(image_two, feature=features)

        norm = cv2.NORM_L2

        if features == "orb":
            norm = cv2.NORM_HAMMING
        
        
        print norm

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(descriptors_1 ,descriptors_2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in xrange(len(matches))]

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.5*n.distance:
                matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = 0)

        matched_image = cv2.drawMatchesKnn(image_one,keypoints_1,image_two,keypoints_2,matches,None,**draw_params)

        matched_image = cv2.resize(matched_image, (700, 700))

        cv2.imshow("matches", matched_image)
        cv2.waitKey(0)