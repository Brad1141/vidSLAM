from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np
import math

kp1 = []
des1 = []
x_arr = []
y_arr = []
z_arr = []

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
            points1.append((x1, y1))
            points2.append((x2, y2))

        getWorldCoords(points1, points2, kp2)

    kp1 = kp2
    des1 = des2

def getWorldCoords(points1, points2, kp2):
    global kp1, x_arr, y_arr, z_arr

    x = 800 / 2
    y = 600 / 2

    #focal lengths (assumes that the field of view is 60)
    f_x = x / math.tan(60 / 2)
    f_y = y / math.tan(60 / 2)

    #camera matrix
    K = np.array([[f_x, 0, x],
                  [0, f_y, y],
                  [0, 0, 1]])

    E, mask = cv2.findEssentialMat(np.float32(points1), np.float32(points2), K)

    U, S, vh = np.linalg.svd(E)

    w_i = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])
    z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                 [0, 0, 0]])

    t = np.asmatrix(U) * np.asmatrix(z) * np.asmatrix(U).T

    R = np.asmatrix(U) * np.asmatrix(w_i) * np.asmatrix(vh).T

    for i in range(len(kp1)):
        #compute the 3d coordinate (x1, x2, x3) for each point
        x3 = ((R[0] - (kp2[i].pt[0] * R[2]) * t) / ((R[0] - kp2[i].pt[0] * R[2]) * y))
        x1 = x3 * kp1[i].pt[0]
        x2 = x3 * kp1[i].pt[1]

        x_arr.append(x1)
        y_arr.append(x2)
        z_arr.append(x3)


def buildMap():
    global x_arr, y_arr, z_arr
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_arr, y_arr, z_arr)

    plt.show()



