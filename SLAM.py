import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

prevX = 0
prevY = 0
prevZ = 0
prev_Pk = 0.05


class Slam:
    def __init__(self):
        self.kp1 = []
        self.des1 = []
        self.camPos = [0, 0, 0]
        self.cam_xyz = []
        self.lm_xyz = []
        self.scale = 6
        self.M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    def runSlam(self, currImg):
        points1, points2 = self.dataAssociation(currImg)
        if points1 and points2:
            self.SfM(points1, points2)
        self.loopClosure()

    def featureExtraction(self, frame):
        # resize the image
        frame = cv2.resize(frame, (800, 600))

        # Initiate ORB
        orb = cv2.ORB_create()

        # find the keypoints and descriptors
        kp, des = orb.detectAndCompute(frame, None)

        # draw keypoints
        img2 = cv2.drawKeypoints(frame, kp, None, color=(255, 0, 0), flags=0)

        # Display results on screen
        cv2.imshow('vSLAM', img2)

    def dataAssociation(self, currImg):
        cv2.resize(currImg, (800, 600))

        # Initiate STAR detector
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        kp2, des2 = orb.detectAndCompute(currImg, None)

        points1 = []
        points2 = []

        if self.kp1:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # horizontal(x) and vertical(y) length of image from center
            x = 800 / 2
            y = 600 / 2

            for mat in matches[:50]:
                # Get the matching keypoints for each of the images
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx

                # x - columns
                # y - rows
                # Get the coordinates
                (x1, y1) = self.kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

                # Append to each list
                points1.append([x1, y1])
                points2.append([x2, y2])

        self.kp1 = kp2
        self.des1 = des2

        return points1, points2

    def SfM(self, points1, points2):
        x = 800 / 2
        y = 600 / 2

        fov = 60 * (math.pi / 180)
        f_x = x / math.tan(fov / 2)
        f_y = y / math.tan(fov / 2)

        # camera matrix
        K = np.array([[f_x, 0, x],
                      [0, f_y, y],
                      [0, 0, 1]])

        E, mask = cv2.findFundamentalMat(np.float32(points2), np.float32(points1), cv2.FM_8POINT)
        points, R, t, mask = cv2.recoverPose(E, np.float32(points2), np.float32(points1), K, 500)
        self.cam_xyz.append([self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]])

        C = np.hstack((R, t))

        for i in range(len(points2)):
            pts2d = np.asmatrix([points2[i][0], points2[i][1], 1]).T
            P = np.asmatrix(K) * np.asmatrix(C)
            pts3d = np.asmatrix(P).I * pts2d
            self.lm_xyz.append([pts3d[0][0] * self.scale + self.camPos[0],
                                pts3d[1][0] * self.scale + self.camPos[1],
                                pts3d[2][0] * self.scale + self.camPos[2]])

        self.camPos = [self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]]

    def buildMap(self):
        self.lm_xyz = np.array(self.lm_xyz)
        self.cam_xyz = np.array(self.cam_xyz)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.lm_xyz[:, [0]], self.lm_xyz[:, [1]], self.lm_xyz[:, [2]])
        ax.scatter(self.cam_xyz[:, [0]], self.cam_xyz[:, [1]], self.cam_xyz[:, [2]], c='r')
        plt.show()


    def loopClosure(self):
        return


