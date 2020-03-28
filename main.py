import numpy as np
import cv2 as cv
# from matplotlib import pyplot as plt


def featureDetection(img):
    # Initiate FAST object with default values
    fast_threshold = 25
    fast = cv.FastFeatureDetector_create(fast_threshold)
    # find and draw the keypoints
    kp = fast.detect(img, None)

    # img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    # Print all default params
    # print("Threshold: {}".format(fast.getThreshold()))
    # print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    # print("neighborhood: {}".format(fast.getType()))
    # print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    # cv.imwrite('fast_true.png', img2)
    # Disable nonmaxSuppression
    # fast.setNonmaxSuppression(0)
    # kp = fast.detect(img, None)
    # print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
    # img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    # cv.imwrite('fast_false.png', img3)
    return kp


def featureTracking():
    cv.CalcOpticalFlowPyrLK(prev, curr, prevPts, currPts, prevFeatures,
                            winSize, level, criteria, flags, guesses=None)


vid = cv.VideoCapture('input/video.mp4')
# Get a frame from the video
ret1, img1 = vid.read()
# Change it to gray scale
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# Get a secod frame from the video
ret2, img2 = vid.read()
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

cv.imshow('Frame', gray)
# cv.waitKey()

# frame_size.height = cvGetCaptureProperty(vid, CV_CAP_PROP_FRAME_HEIGHT)

# kp1 = featureDetection(gray1)
# kp2 = featureDetection(gray2)

# Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 400
detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
# Step 2: Matching descriptor vectors with a brute force matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE)
matches = matcher.match(descriptors1, descriptors2)
# Draw matches
img_matches = np.empty((max(img1.shape[0], img2.shape[0]),
                        img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches)
# Show detected matches
cv.imshow('Matches', img_matches)
cv.waitKey()

# cv.imshow('Frame', gray2)
# cv.waitKey()
