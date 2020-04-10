import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import entropy

cap = cv2.VideoCapture('input/video2.mp4')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters of the ORB detector
orb_params = dict(nfeatures=1500, scaleFactor=1.2, nlevels=4,
                  edgeThreshold=31, firstLevel=0, WTA_K=2,
                  scoreType=cv2.ORB_HARRIS_SCORE,
                  patchSize=31, fastThreshold=20)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

# Calibration matrix obtained with calibration.py
cal_mtx = np.array([[1000.5, 0.0, 634.5711],
                    [0.0, 998.3832, 489.9649],
                    [0.0, 0.0, 1.0]])

# Distortion obtained during the calibration
dist = np.array([0.0254, 0.0075, 0.0, 0.0])


# Initial position of the camera
current_pos = np.zeros((3, 1))  # [0,0,0]
current_rot = np.eye(3)
camera_traj = np.zeros((3, 1))  # Initialize camera postions history

# Create figure for 3D plot
position_figure = plt.figure()
ax = plt.axes(projection='3d')

# Take the first frame and rectify it
ret, old_frame = cap.read()
h,  w = old_frame.shape[:2]
new_mtx, roi = cv2.getOptimalNewCameraMatrix(cal_mtx, dist, (w, h), 1, (w, h))
x, y, w, h = roi  # region of interest after the rectification
old_frame = cv2.undistort(old_frame, cal_mtx, dist, None, new_mtx)
old_frame = old_frame[y:y+h, x:x+w]
cv2.imshow('img', old_frame)
cv2.waitKey()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Compute entropy (for keyframes) and features to track
old_ent = entropy(old_gray, base=2)
# surf = cv2.xfeatures2d.SURF_create()
# p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
orb = cv2.ORB_create(**orb_params)
p0, desc = orb.detectAndCompute(old_gray, None)
p0 = cv2.KeyPoint_convert(p0)
p0 = np.array(p0)
p0 = p0.reshape(-1, 1, 2)
# print(p0.shape)

# Create a mask image for drawing purposes
doodles = np.zeros_like(old_frame)
# Create some random colors
color = np.random.randint(0, 255, (1500, 3))
thres = 10

while(1):
    ret, frame = cap.read()
    if ret is False:
        print('Fin del video')
        break
    # frame = cv2.undistort(frame, cal_mtx, dist)
    frame = cv2.undistort(frame, cal_mtx, dist, None, new_mtx)
    frame = frame[y:y+h, x:x+w]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ent = entropy(frame_gray, base=2)
    dif = np.sum(np.absolute(ent-old_ent))

    if dif > thres:
        print('Dif is: ', dif)
        old_ent = ent  # Reassign old entropy to new keyframe

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                               **lk_params)
        print(p1.shape)
        # Select good points (the ones that are detected from the old image in
        # the new one)
        good_old = p0[st == 1]
        good_new = p1[st == 1]

        # Find Essential or Fundamental matrix
        # EssMatrix, mask = cv2.findEssentialMat(good_old, good_new, cal_mtx,
        #                                        cv2.RANSAC, 0.999, 1.0, None)
        EssMatrix, mask = cv2.findFundamentalMat(good_old, good_new,
                                                 cv2.FM_RANSAC, 0.999, 1.0)

        # Compute relativa camera rotatio and translation from camera matrix
        points, R, t, mask = cv2.recoverPose(EssMatrix, good_old, good_new,
                                             cal_mtx)

        # Update current camera position
        scale = 1.0
        current_pos += current_rot.dot(t) * scale
        current_rot = R.dot(current_rot)

        # Draw the detected points in the 2D image (for debbuging)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # doodles = cv2.line(doodles, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        # img = cv2.add(frame, doodles)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30)

        # Add camera pose to the saved trajectory (for representation)
        camera_traj = np.concatenate((camera_traj, current_pos), axis=1)

        # Update previous points
        p0 = good_new.reshape(-1, 1, 2)

    # Update the previous frame
    old_gray = frame_gray.copy()


# Save the camera trajectory
np.save('traj.npy', camera_traj)

cv2.destroyAllWindows()
cap.release()
