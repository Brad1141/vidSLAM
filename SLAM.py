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
        self.scale = 5


    #need to add bundle adjustment, loop closure, and p3p
    def runSlam(self, currImg):
        points1, points2 = self.dataAssociation(currImg)
        if points1 and points2:
            self.SfM(points1, points2)

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

        # focal lengths (assumes that the field of view is 60)
        #convert to radians
        fov = 60 * (math.pi / 180)
        f_x = x / math.tan(fov / 2)
        f_y = y / math.tan(fov / 2)

        # intrinsic matrix
        K = np.array([[f_x, 0, x],
                      [0, f_y, y],
                      [0, 0, 1]])

        #find the fundamental matrix
        F, mask = cv2.findFundamentalMat(np.float32(points2), np.float32(points1), cv2.FM_8POINT)

        #calculate the Rotation matrix and Translation vector
        points, R, t, mask = cv2.recoverPose(F, np.float32(points2), np.float32(points1), K, 500)
        R = np.asmatrix(R).I

        #find the new camera position
        self.cam_xyz.append([self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]])

        #calcuate the camera matrix
        C = np.hstack((R, t))
        P = np.asmatrix(K) * np.asmatrix(C)

        #turn the 2d landmarks into 3d points
        for i in range(len(points2)):
            #calculate the 3x1 matrix of the point
            x_i = np.asmatrix([points2[i][0], points2[i][1], 1]).T

            #calculate the 3d equivalent
            X_i = np.asmatrix(P).I * x_i
            self.lm_xyz.append([X_i[0][0] * self.scale + self.camPos[0],
                                X_i[1][0] * self.scale + self.camPos[1],
                                X_i[2][0] * self.scale + self.camPos[2]])

        #update the camera position
        self.camPos = [self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]]

    def buildMap(self):

        self.lm_xyz = np.array(self.lm_xyz)
        self.cam_xyz = np.array(self.cam_xyz)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.lm_xyz[:, [0]], self.lm_xyz[:, [1]], self.lm_xyz[:, [2]])
        ax.scatter(self.cam_xyz[:, [0]], self.cam_xyz[:, [1]], self.cam_xyz[:, [2]], c='r')
        plt.show()


