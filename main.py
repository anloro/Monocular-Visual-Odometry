import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def featureDetection(img):
    # Initiate FAST object with default values
    fast_threshold = 25
    fast = cv.FastFeatureDetector_create(fast_threshold)
    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    # Print all default params
    print("Threshold: {}".format(fast.getThreshold()))
    print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
    print("neighborhood: {}".format(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))
    cv.imwrite('fast_true.png', img2)
    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)
    print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))
    img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
    cv.imwrite('fast_false.png', img3)


def featureTracking():
    cv.CalcOpticalFlowPyrLK(prev, curr, prevPts, currPts, prevFeatures,
                            winSize, level, criteria, flags, guesses=None)


vid = cv.VideoCapture('input/video.mp4')
ret, img = vid.read()
cv.imshow('Frame', img)
# frame_size.height = cvGetCaptureProperty(vid, CV_CAP_PROP_FRAME_HEIGHT)


# img = cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)
featureDetection(img)
