import numpy as np
import cv2
# import matplotlib.pyplot as plt
from scipy.stats import entropy

cap = cv2.VideoCapture('input/video.mp4')

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

count = 0
# Initial position of the camera
current_pos = np.zeros((3, 1))  # [0,0,0]
current_rot = np.eye(3)
camera_traj = np.zeros((3, 1))  # Initialize camera postions history

# Projection matrix
projMatr0 = np.concatenate((np.dot(cal_mtx, current_rot),
                            np.dot(cal_mtx, current_pos)), axis=1)

# Create figure for 3D plot
# position_figure = plt.figure()
# ax = plt.axes(projection='3d')

# Initialize environment
env = np.zeros((3, 1))

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
p0_k, desc0 = orb.detectAndCompute(old_gray, None)
p0 = cv2.KeyPoint_convert(p0_k)
p0 = np.array(p0)
p0 = p0.reshape(-1, 1, 2)
# print(p0.shape)

# Create a mask image for drawing purposes
doodles = np.zeros_like(old_frame)
# Create some random colors
color = np.random.randint(0, 255, (1500, 3))
thres = 10

while(1):
    # Get new frame
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

    # if dif > thres:
    # print('Dif is: ', dif)
    old_ent = ent  # Reassign old entropy to new keyframe

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                           **lk_params)

#     # -----------------------------------------------------------------
#     # Try with ORB detector
#     p1_k, desc1 = orb.detectAndCompute(frame_gray, None)
#     p1_Or = cv2.KeyPoint_convert(p1_k)
#     p1_Or = np.array(p1_Or)
#     p1_Or = p1.reshape(-1, 1, 2)
#
#     # -- Step 2: Matching descriptor vectors with a FLANN based matcher
#     # Since SURF is a floating-point descriptor NORM_L2 is used
#     FLANN_INDEX_LSH = 6
#     index_params = dict(algorithm=FLANN_INDEX_LSH,
#                         table_number=6,  # 12
#                         key_size=12,     # 20
#                         multi_probe_level=1)  # 2
#     search_params = dict(checks=100)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
# # matchr = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
#     matches = flann.knnMatch(desc0, desc1, k=2)
#     # print(matches)
#     # Need to draw only good matches, so create a mask
#     matchesMask = [[0, 0] for i in range(len(matches))]
#
#     # ratio test as per Lowe's paper
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.7*n.distance:
#             matchesMask[i] = [1, 0]
#
#     draw_params = dict(matchColor=(0, 255, 0),
#                        singlePointColor=(255, 0, 0),
#                        matchesMask=matchesMask,
#                        flags=cv2.DrawMatchesFlags_DEFAULT)
#
#     # img3 = cv2.drawMatchesKnn(old_frame, p0_k, frame, p1_k, matches,
#     #                           None, **draw_params)
#     # plt.imshow(img3,), plt.show()
#     # cv2.waitKey()
#
#     matchesMask = np.array(matchesMask)
#     # print(matchesMask.shape)
#     # print(p0[matchesMask == 1])
#
# #     # # -- Filter matches using the Lowe's ratio test
# #     # ratio_thresh = 0.7
# #     # good_matches = []
# #     # for m, n in knn_matches:
# #     #     if m.distance < ratio_thresh * n.distance:
# #     #         good_matches.append(m)
#
#     # Initialize lists
#     list_kp0 = []
#     list_kp1 = []
#
#     # For each match...
#     for mat in range(len(matches)):
#         # Get the matching keypoints for each of the images
#         # print(mat)
#         # print(matches[mat])
#         # print(matches.shape)
#         img1_idx = matches[mat][0].queryIdx
#         img2_idx = matches[mat][1].trainIdx
#         # x - columns
#         # y - rows
#         # Get the coordinates
#         (x1, y1) = p0_k[img1_idx].pt
#         (x2, y2) = p1_k[img2_idx].pt
#         # Append to each list
#         list_kp0.append((x1, y1))
#         list_kp1.append((x2, y2))
#
#     # print(np.array(list_kp0).shape)
#     good_old = np.array(list_kp0, dtype='Float32')
#     good_new = np.array(list_kp1, dtype='Float32')
#     # print(p1.shape)
#     # print(st.shape)
#     # -----------------------------------------------------------------

    # Select inlayer points
    # print('p0 shape: ', p0.shape)
    good_old = p0[st == 1]
    good_new = p1[st == 1]
    good_new = good_new.reshape(-1, 1, 2)
    good_old = good_old.reshape(-1, 1, 2)

    # Debugging
    # print('good_old shape: ', good_old.shape)
    # print('type of good_old: ', type(good_old))
    # print('good_old: ', good_old)
    # print('good_old shape: ', good_new.shape)
    # print('type of good_new: ', type(good_new))
    # print('good_old: ', good_new)

    # Find Essential or Fundamental matrix
    # EssMatrix, mask = cv2.findEssentialMat(good_old, good_new, cal_mtx,
    #                                        cv2.RANSAC, 0.999, 1.0, None)
    EssMatrix, mask = cv2.findFundamentalMat(good_old, good_new,
                                             cv2.FM_RANSAC, 0.999, 1.0)

    # Only select inlayers
    good_old = good_old[mask == 1]
    good_new = good_new[mask == 1]
    # print('good_old shape: ', good_old.shape)

    # # Refine points
    r, c = good_new.shape
    # good_new = good_new.reshape(1, r, 2)
    # good_old = good_old.reshape(1, r, 2)
    # # print(good_new.shape)
    # good_old, good_new = cv2.correctMatches(EssMatrix, good_old, good_new)
    good_new = good_new.reshape(r, 1, 2)
    good_old = good_old.reshape(r, 1, 2)
    # print(good_new.shape)

    # Compute relative camera rotation and translation
    points, R, t, mask = cv2.recoverPose(EssMatrix, good_old, good_new,
                                         cal_mtx)
    # Debugging
    # print('R is: ', R)
    # print('t is: ', t)

    # Euler angles from rotation matrix
    thetax = np.arctan2(R[2][1], R[2][2])
    thetay = np.arctan2(-R[2][0], np.sqrt(R[2][1]*R[2][1]+R[2][2]*R[2][2]))
    thetaz = np.arctan2(R[1][0], R[0][0])

    # print('thetax: ', thetax)
    # print('thetay: ', thetay)
    # print('thetaz: ', thetaz)

    # if thetax > 0.5:  # and thetax > 1 and thetax > 1:
    # Update current camera position
    scale = 1.0
    current_pos += current_rot.dot(t) * scale
    current_rot = R.dot(current_rot)

    # print(good_new)
    # print(good_old)
    # print(good_new.shape)
    good_new = good_new[~np.isnan(good_new).any(axis=2)]
    good_old = good_old[~np.isnan(good_old).any(axis=2)]
    # print(good_new.shape)
    # print(good_old)

    # Draw the detected points in the 2D image (for debbuging)
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # doodles = cv2.line(doodles, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    # img = cv2.add(frame, doodles)
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    if cv2.waitKey(30) & 0xFF == ord('s'):
        cv2.imwrite('output/featureDetect{}.png'.format(count), frame)
        count += 1

    # For creating the video
    # cv2.imwrite('output/videoImFeat/featureDetect{}.png'.format(count), frame)
    # count += 1

    # Add camera pose to the saved trajectory (for representation)
    camera_traj = np.concatenate((camera_traj, current_pos), axis=1)

    # Update previous points
    p0 = good_new.reshape(-1, 1, 2)

    # Compute projection matrix
    projMatr1 = np.concatenate((np.dot(cal_mtx, R),
                                np.dot(cal_mtx, t)), axis=1)

    points3d = cv2.triangulatePoints(projMatr0, projMatr1, p0, p0)
    points3d = np.array(points3d)
    points3d = cv2.convertPointsFromHomogeneous(points3d.transpose())
    # points3d = points3d.reshape(-1, 3)
    points3d = points3d.reshape(3, -1)
    print(points3d.shape)
    env = np.concatenate((env, points3d), axis=1)

    # Update the previous frame
    old_gray = frame_gray.copy()


# Save the camera trajectory
np.save('output/traj.npy', camera_traj)
np.save('output/environm.npy', env)

cv2.destroyAllWindows()
cap.release()
