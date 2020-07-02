import numpy as np
import cv2
import copy
from os import path

class camera:
    """Calibrated camera
    
    Attributes:
        cal_mtx: Calibration matrix
        dist: Distortion parameters
    """

    def __init__(self):
        self.cal_mtx = np.array([[1000.5, 0.0, 634.5711],
                                [0.0, 998.3832, 489.9649],
                                [0.0, 0.0, 1.0]])
        self.dist = np.array([0.0254, 0.0075, 0.0, 0.0])


class controlVideo:
    """Controls for the video
    
    Attributes:
        video: Video reader
        frame_old: Previous image from the video
        frame_new: New image from the video
    """

    def __init__(self, video):
        self.video = video
        ret, self.frame_old = video.read()
        ret, self.frame_new = video.read()

    def updateFrame(self):
        """Gets new frame from video"""
        self.frame_old = self.frame_new
        ret, self.frame_new = self.video.read()
        return ret


class keyframe:
    """ A keyframe.
    
    Attributes:
        frame: Single image
        keypoints: Tuples that indicate feature's positions
    """

    def __init__(self, frame):
        self.frame = frame
        self.keypoints = []
    
    def setFrame(self, frame):
        """Sets a new image"""
        self.frame = self.frame

    def normalizeFrame(self, camera):
        """Normalizes frame using the camera parameters"""
        frame = self.frame
        h,  w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(camera.cal_mtx, camera.dist, 
                                                    (w, h), 1, (w, h))
        x, y, w, h = roi  # region of interest after the rectification
        frame = cv2.undistort(frame, camera.cal_mtx, camera.dist, None, new_mtx)
        frame = frame[y:y+h, x:x+w]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame = frame
    
    def getNumKeypoints(self, features):
        """Transforms keypoint object into tuples"""
        kp = features.keypoints
        kp = cv2.KeyPoint_convert(kp)
        kp = np.array(kp)
        kp = kp.reshape(-1, 1, 2)
        self.keypoints = kp

class orb_features:
    """ORB detector
    
    Attributes:
        param: Parameters of the ORB detector
        detector: Detector object
    """

    def __init__(self):
        self.param = dict(nfeatures=1500, scaleFactor=1.2, nlevels=4,
                        edgeThreshold=31, firstLevel=0, WTA_K=2,
                        scoreType=cv2.ORB_HARRIS_SCORE,
                        patchSize=31, fastThreshold=20)        
        self.detector = cv2.ORB_create(**self.param)
    
    def detect(self, frame):
        """Detects ORB features in a given frame object"""
        frame = frame.frame
        keypoints, descriptors = self.detector.detectAndCompute(frame, None)
        self.keypoints = keypoints
        self.descriptors = descriptors


class lkfilter:
    """Lucas-Kanade optical filter
    
    Attributes:
        param: Parameters of the lk algorithm
    """

    def __init__(self):
        # Parameters for lucas kanade optical flow
        self.param = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            10, 0.03))
    
    def computeFlow(self, old_frame, new_frame):
        """ Compute optical flow"""    
        old_kp = old_frame.keypoints
        new_kp, mask, err = cv2.calcOpticalFlowPyrLK(old_frame.frame, new_frame.frame, 
                                            old_kp, None, **self.param)
        old_kp = self.getInlayers(old_kp, mask)
        new_kp = self.getInlayers(new_kp, mask)
        old_frame.keypoints = old_kp
        new_frame.keypoints = new_kp

        return old_frame, new_frame
        
                                        
    def getInlayers(self, keypoints, mask):
        """Gets matched keypoints"""
        keypoints = keypoints[mask == 1]
        keypoints = keypoints.reshape(-1, 1, 2)
        return keypoints


class trajectory:
    """Trajectory of the camera
    
    Attributes:
        current_pos: Spatial position of the camera in Euclidean coordinates
        current_rot: Current rotation of the camera with respect to reference
        camera_traj: Record of all the positons of the camera
    """

    def __init__(self):
        self.current_pos = np.zeros((3, 1))  # [0,0,0]
        self.current_rot = np.eye(3)
        self.camera_traj = np.zeros((3, 1))  # Initialize camera postions history

    def computeEssential(self, old_kp, new_kp, cal_mtx):
        """Find Essential Matrix"""
        EssMatrix, mask = cv2.findEssentialMat(old_kp, new_kp, cal_mtx,
                                            cv2.RANSAC, 0.999, 1.0, None)
        old_kp = old_kp[mask == 1]
        new_kp = new_kp[mask == 1]
        r, c = new_kp.shape
        new_kp = new_kp.reshape(r, 1, 2)
        old_kp = old_kp.reshape(r, 1, 2)

        return old_kp, new_kp, EssMatrix
    
    def computeTransform(self, old_kp, new_kp, cal_mtx, EssMatrix):
        """Compute relative camera rotation and translation"""
        points, R, t, mask = cv2.recoverPose(EssMatrix, old_kp, new_kp,
                                            cal_mtx)
        old_kp = old_kp[~np.isnan(old_kp).any(axis=2)]
        new_kp = new_kp[~np.isnan(new_kp).any(axis=2)]

        return old_kp, new_kp, R, t

    def updatePose(self, R, t):
        """Update the current camera position and orientation""" 
        scale = 1.0
        self.current_pos += self.current_rot.dot(t) * scale
        self.current_rot = R.dot(self.current_rot)

    def computePose(self, old_frame, new_frame, camera):
        """Compute new camera positon in world coordinates"""
        old_kp = old_frame.keypoints
        new_kp = new_frame.keypoints
        cal_mtx = camera.cal_mtx
        old_kp, new_kp, EssMatrix = self.computeEssential(old_kp, new_kp, cal_mtx)
        old_kp, new_kp, R, t = self.computeTransform(old_kp, new_kp, cal_mtx, EssMatrix)
        self.updatePose(R, t)
        self.updateTrajectory()
    
    def updateTrajectory(self):
        """Add new position to the record"""
        self.camera_traj = np.concatenate((self.camera_traj, self.current_pos), axis=1)
    
    def saveTrajectory(self):
        """Save the trajectory of the camera into a .npy file"""
        filepath = 'output/'
        if path.exists(filepath):
            filename = 'traj.npy'
            filepath += filename 
            np.save(filepath, self.camera_traj)
            print("Trajectory saved to: " + filepath)
        else:
            print("Writing path does not exist!")

def drawPoints(frame):
        """Draw the detected points in the 2D image (for debbuging)"""
        kp = frame.keypoints
        frame = frame.frame
        for i, (p) in enumerate(zip(kp)):
            a, b = np.ravel(p)
            frame = cv2.circle(frame, (a, b), 5, [0, 0, 255], -1)
        cv2.imshow('frame', frame)
        cv2.waitKey(30)


def main():
    # Initialize camera and video objects
    cam = camera()
    video_cap = cv2.VideoCapture('input/video4.mp4')
    video = controlVideo(video_cap)
    video_trajectory = trajectory()
    ret = True

    # Initialize frames
    video.updateFrame()
    old_frame = keyframe(video.frame_old)
    old_frame.normalizeFrame(cam)
    new_frame = keyframe(video.frame_new) 
    new_frame.normalizeFrame(cam)
    # cv2.imshow("", video.frame_new)
    # cv2.waitKey()

    # Initialize detector
    orb = orb_features()

    while(1):
        # Compute features
        orb.detect(old_frame)
        old_frame.getNumKeypoints(orb)

        # Match features
        lk = lkfilter()
        old_frame, new_frame = lk.computeFlow(old_frame, new_frame)

        # Get the transformation between the 2 frames
        video_trajectory.computePose(old_frame, new_frame, cam)

        # drawPoints(new_frame)

        # Update keyframes
        old_frame = copy.copy(new_frame)
        ret = video.updateFrame()
        if ret is False:
            print('End of the video')
            break
        new_frame.frame = video.frame_new 
        new_frame.normalizeFrame(cam)

    video_trajectory.saveTrajectory()


if __name__ == '__main__':
    print("Obtaining camera trajectory")
    main()