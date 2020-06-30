from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np
import math
import SLAM

kp1 = []
des1 = []
camX_arr = []
camY_arr = []
camZ_arr = []
x_arr = []
y_arr = []
z_arr = []
prevLM = []
prevC = []
pt = [0, 0, 0]


def featureExtraction(frame):
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp, des = orb.detectAndCompute(frame, None)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(frame, kp, None, color=(0, 0, 255), flags=0)

    cv2.imshow('drive', img2)


def dataAssociation(currImg):
    global kp1, des1

    cv2.resize(currImg, (800, 600))

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp2, des2 = orb.detectAndCompute(currImg, None)

    if kp1 != []:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        points1 = []
        points2 = []

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
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Append to each list
            points1.append([x1, y1])
            points2.append([x2, y2])

        getWorldCoords(points1, points2, kp2)

    kp1 = kp2
    des1 = des2


def orbCompare():
    img = cv2.imread("trees.jpg")
    img = cv2.resize(img, (600, 600))
    (h, w) = img.shape[:2]

    # calculate the center of the image
    center = (w / 2, h / 2)

    angle90 = 90
    scale = 1.0

    M = cv2.getRotationMatrix2D(center, 27, scale)
    r90 = cv2.warpAffine(img, M, (h, w))

    orb = cv2.ORB_create()

    kp, des = orb.detectAndCompute(img, None)
    kp2, des2 = orb.detectAndCompute(r90, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img, kp, r90, kp2, matches[:10], r90, flags=2)
    cv2.imshow('corners', img3)
    cv2.waitKey()


def getWorldCoords(points1, points2, kp2):
    global kp1, camX_arr, camY_arr, camZ_arr, x_arr, y_arr, z_arr, prevLM, prevC, pt

    x = 800 / 2
    y = 600 / 2
    currLM = []

    # focal lengths (assumes that the field of view is 60)
    f_x = x / math.tan(60 / 2) * -1
    f_y = y / math.tan(60 / 2) * -1

    # camera matrix
    K = np.array([[525, 0, x],
                  [0, 525, y],
                  [0, 0, 1]])

    #ret, K, dist, R, t = cv2.calibrateCamera()

    # E, mask = cv2.findEssentialMat(np.float32(points1), np.float32(points2), K)
    E, mask = cv2.findFundamentalMat(np.float32(points1), np.float32(points2), cv2.FM_8POINT)
    points, R, t, mask = cv2.recoverPose(E, np.float32(points1), np.float32(points2), K, 500)

    R = np.asmatrix(R).I
    scale = np.sqrt((t[0] - pt[0])*(t[0] - pt[0]) + (t[1] - pt[1])*(t[1] - pt[1]) + (t[2] - pt[2])*(t[2] - pt[2]))

    camX_arr.append(pt[0] + t[0])
    camY_arr.append(pt[1] + t[1])
    camZ_arr.append(pt[2] + t[2])

    # R = np.asmatrix(R).I
    C = np.hstack((R, t))
    # points1, points2 = np.asmatrix(points1).T, np.asmatrix(points2).T
    #ret, K = cv2.calibrateCamera(np.float32(points1), np.float32(points2), (800, 600), K)


    if prevC != []:
        # cords4d = cv2.triangulatePoints(prevC, C, np.float32(points1), np.float32(points2))
        # pts3d = cv2.convertPointsFromHomogeneous(cords4d.transpose())
        #
        # for point in pts3d:
        #     x_arr.append(point[0][0])
        #     y_arr.append(point[0][1])
        #     z_arr.append(point[0][2])

        # for i in range(50):
        #     x_arr.append(coords4d[0][i])
        #     y_arr.append(coords4d[1][i])
        #     z_arr.append(coords4d[2][i])
        # points2 = np.array(points2)
        for i in range(len(points2)):
            # pts2d = np.asmatrix([points2[i][0], points2[i][1], 1]).T
            # pts2d1 = np.asmatrix([points1[i][0], points1[i][1], 1]).T
            # print(pts2d.T * E * pts2d1)
            # pts2d = P * pts3d
            pts2d = np.asmatrix([points2[i][0], points2[i][1], 1]).T
            P = np.asmatrix(K) * np.asmatrix(C)
            pts3d = np.asmatrix(P).I * pts2d
            x_arr.append(pts3d[0][0] * scale)
            y_arr.append(pts3d[1][0] * scale)
            z_arr.append(pts3d[2][0] * scale)

    pt = [pt[0] + t[0], pt[1] + t[1], pt[2] + t[2]]


    # U, S, vh = np.linalg.svd(E)
    #
    # w_i = np.array([[0, 1, 0],
    #                 [-1, 0, 0],
    #                 [0, 0, 1]])
    # z = np.array([[0, 1, 0],
    #               [-1, 0, 0],
    #               [0, 0, 0]])
    #
    # t = np.asmatrix(U) * np.asmatrix(z) * np.asmatrix(U).T
    #
    # R = np.asmatrix(U) * np.asmatrix(w_i) * np.asmatrix(vh).T

    # for i in range(len(points1)):
    #     #compute the 3d coordinate (x1, x2, x3) for each point
    #     x3 = ((R[0] - (kp2[i].pt[0] * R[2]) * t) / ((R[0] - kp2[i].pt[0] * R[2]) * y))
    #     x1 = x3 * kp1[i].pt[0]
    #     x2 = x3 * kp1[i].pt[1]
    #
    #     x_arr.append(x1)
    #     y_arr.append(x2)
    #     z_arr.append(x3)
    #     # #
    #     # currZ = ((R[0] - (points2[i][0] * R[2]) * t) / ((R[0] - points2[i][0] * R[2]) * y))
    #     # prevZ = ((R[0] - (points1[i][0] * R[2]) * t) / ((R[0] - points1[i][0] * R[2]) * y))
    #     # currZ = np.array(currZ)
    #     # prevZ = np.array(prevZ)
    #     # # # currZ = R.dot(t)
    #     # currLM.append([points2[i][0] * currZ[0][1], points2[i][1] * currZ[0][1], currZ[0][2]])
    #     # prevLM.append([points1[i][0] * prevZ[0][1], points1[i][1] * prevZ[0][1], prevZ[0][2]])
    #     #currLM.append([points2[i][0] * t[0], points2[i][1] * t[1], t[2]])
    #     # p = R.dot(t)
    #     # # currLM.append([points2[i][0] * p[0], points2[i][1] * p[1], p[2]])
    #     # # prevLM.append([points1[i][0] * p[0], points1[i][1] * p[1], p[2]])
    #     # x_arr.append(points2[i][0] * p[0])
    #     # y_arr.append(points2[i][1] * p[1])
    #     # z_arr.append(p[2])

    # if prevLM:
    #     x_arr1, y_arr1, z_arr1 = SLAM.predictState(currLM, prevLM, points1, points2)
    #     for i in range(len(x_arr1)):
    #         x_arr.append(x_arr1)
    #         y_arr.append(y_arr1)
    #         z_arr.append(z_arr1)
    #     camX, camY, camZ = x_arr[0], y_arr[0], z_arr[0]
    #     camX_arr.append(camX)
    #     camY_arr.append(camY)
    #     camZ_arr.append(camZ)

    prevC = C

    return currLM, prevLM


def buildMap():
    global x_arr, y_arr, z_arr, camX_arr, camY_arr, camZ_arr
    #x_arr, y_arr, z_arr = removeOutliers(x_arr, y_arr, z_arr)
    #camX_arr, camY_arr, camZ_arr = removeOutliers(camX_arr, camY_arr, camZ_arr)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_arr, y_arr, z_arr)
    ax.scatter(camX_arr, camY_arr, camZ_arr, c='r')
    plt.show()


def removeOutliers(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    upper_quartile = np.percentile(z, 75)
    lower_quartile = np.percentile(z, 25)
    IQR = (upper_quartile - lower_quartile)
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for i in range(len(z[0])):
        if z[0][i] <= quartileSet[0] or z[0][i] >= quartileSet[1]:
            del x[0][i]
            del y[0][i]
            del z[0][i]
    return x, y, z
