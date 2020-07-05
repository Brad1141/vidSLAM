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
    def __init__(self, fov):
        self.kp1 = []
        self.des1 = []
        self.camPos = [0, 0, 0]
        self.cam_xyz = []
        self.lm_xyz = []
        self.scale = 5
        self.fov = fov


    #need to add bundle adjustment, loop closure, and p3p
    def runSlam(self, currImg):
        points1, points2 = self.dataAssociation(currImg)
        if points1 and points2:
            self.reconstructCoords(points1, points2)

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

    def reconstructCoords(self, points1, points2):
        x = 800 / 2
        y = 600 / 2
        currLM = []

        # focal lengths (assumes that the field of view is 60)
        fov = self.fov * (math.pi / 180)
        f_x = x / math.tan(fov / 2)
        f_y = y / math.tan(fov / 2)

        # camera matrix
        K = np.array([[f_x, 0, x],
                      [0, f_y, y],
                      [0, 0, 1]])

        # ret, K, dist, R, t = cv2.calibrateCamera()

        # E, mask = cv2.findEssentialMat(np.float32(points1), np.float32(points2), K)
        E, mask = cv2.findFundamentalMat(np.float32(points2), np.float32(points1), cv2.FM_8POINT)
        points, R, t, mask = cv2.recoverPose(E, np.float32(points2), np.float32(points1), K, 500)
        R = np.asmatrix(R).I
        # scale = np.sqrt((t[0] - self.camPos[0]) * (t[0] - self.camPos[0]) +
        #                 (t[1] - self.camPos[1]) * (t[1] - self.camPos[1]) +
        #                 (t[2] - self.camPos[2]) * (t[2] - self.camPos[2]))

        self.cam_xyz.append([self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]])

        C = np.hstack((R, t))

        for i in range(len(points2)):
            # pts2d = np.asmatrix([points2[i][0], points2[i][1], 1]).T
            # pts2d1 = np.asmatrix([points1[i][0], points1[i][1], 1]).T
            # print(pts2d.T * E * pts2d1)
            # pts2d = P * pts3d
            pts2d = np.asmatrix([points2[i][0], points2[i][1], 1]).T
            P = np.asmatrix(K) * np.asmatrix(C)
            pts3d = np.asmatrix(P).I * pts2d
            # prevPts2d = np.asmatrix([points1[i][0], points1[i][1], 1]).T
            # prevPts3d = np.asmatrix(P).I * prevPts2d
            # currLM.append([pts3d[0][0] * scale + pt[0], pts3d[1][0] * scale + pt[1], pts3d[2][0] * scale + pt[2]])
            # prevLM.append([prevPts3d[0][0] * scale + pt[0], prevPts3d[1][0] * scale + pt[1], prevPts3d[2][0] * scale + pt[2]])
            self.lm_xyz.append([pts3d[0][0] * self.scale + self.camPos[0],
                                pts3d[1][0] * self.scale + self.camPos[1],
                                pts3d[2][0] * self.scale + self.camPos[2]])

        self.camPos = [self.camPos[0] + t[0], self.camPos[1] + t[1], self.camPos[2] + t[2]]

        # self.camPos = pt
        # x_arr1, y_arr1, z_arr1 = SLAM.EKF(currLM, prevLM, points1, points2, self.camPos)
        # for i in range(len(x_arr1)):
        #     x_arr.append(x_arr1)
        #     y_arr.append(y_arr1)
        #     z_arr.append(z_arr1)

    def buildMap(self):
        self.lm_xyz = np.array(self.lm_xyz)
        self.cam_xyz = np.array(self.cam_xyz)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(self.lm_xyz[:, [0]], self.lm_xyz[:, [1]], self.lm_xyz[:, [2]])
        ax.scatter(self.cam_xyz[:, [0]], self.cam_xyz[:, [1]], self.cam_xyz[:, [2]], c='r')
        plt.show()


# extended kalman filter (still in development)
def EKF(currLM, prevLM, points1, points2, camPos):
    global prevX, prevY, prevZ, prev_Pk
    # models
    # xk is a model of the previous state
    xk = np.array([])
    xk = np.append(xk, prevX)
    xk = np.append(xk, prevY)
    xk = np.append(xk, prevZ)

    for i in range(len(prevLM)):
        xk = np.append(xk, prevLM[i][0])
        xk = np.append(xk, prevLM[i][1])
        xk = np.append(xk, prevLM[i][2])

    xk = np.asmatrix(xk)

    # zk is a model of the current state
    zk = camPos
    for i in range(len(currLM)):
        zk = np.append(zk, currLM[i][0])
        zk = np.append(zk, currLM[i][1])
        zk = np.append(zk, currLM[i][2])

    # predict
    # because f(x) = x, Fk is the identity matrix
    Fk = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

    Fk = np.asmatrix(Fk)

    # The process noise
    W = np.array([abs(zk[0] - prevX), abs(zk[1] - prevY), zk[2] - prevZ])
    C = 0.05

    W = np.asmatrix(W)

    Q = W * C * W.T

    Q = np.asmatrix(Q)

    Hk = []
    for i in range(len(currLM)):
        x_change = (currLM[i][0] - prevLM[i][0]) / (zk[0] - prevX)
        y_change = (currLM[i][1] - prevLM[i][1]) / (zk[1] - prevY)
        z_change = (currLM[i][2] - prevLM[i][2]) / (zk[2] - prevZ)

        arr = [x_change, y_change, z_change]
        Hk = np.append(Hk, arr)

    Hk = np.resize(Hk, (50, 3))
    Hk = np.asmatrix(Hk)

    R = np.array([[1, 0],
                  [0, 1]])
    R = np.asmatrix(R)

    Pk = Fk * prev_Pk * Fk.T + Q
    Pk = np.asmatrix(Pk)

    # update
    # caculate kalman gain
    try:
        Gk = Pk * Hk.T * (Hk * Pk * Hk.T).I
    except:
        Gk = np.ones([3, 50])

    xk = np.array(xk)
    zk = np.array(zk)
    Gk = np.array(Gk)

    xk_x = [xk[0][i] + Gk[0][i] * (zk[i] - xk[0][i]) for i in range(len(Gk[0]))]
    xk_y = [xk[0][i] + Gk[1][i] * (zk[i] - xk[0][i]) for i in range(len(Gk[0]))]
    xk_z = [xk[0][i] + Gk[2][i] * (zk[i] - xk[0][i]) for i in range(len(Gk[0]))]

    # 3x3 indentity matrix
    I = np.asmatrix(np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]))

    prev_Pk = (I - Gk * Hk) * Pk
    prevX = xk[0][0]
    prevY = xk[0][1]
    prevZ = xk[0][2]

    return xk_x, xk_y, xk_z