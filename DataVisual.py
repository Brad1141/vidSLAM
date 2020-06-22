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

    #create the fundamental matrix
    E, mask = cv2.findFundamentalMat(np.float32(points1), np.float32(points2), cv2.FM_RANSAC)

    #preform single value decomposition on the matrix
    U, S, vt = np.linalg.svd(E)

    w_i = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])
    z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                 [0, 0, 0]])

    #calcluate the Rotation matrix and translation vector
    t = np.asmatrix(U) * np.asmatrix(z) * np.asmatrix(U).T

    R = np.asmatrix(U) * np.asmatrix(w_i) * np.asmatrix(vt).T

    #for each landmark, calculate the 3D coordinates
    for i in range(len(kp2)):
        #compute the 3d coordinate (x1, x2, x3) for each point
        x3 = ((R[0] - (kp2[i].pt[0] * R[2]) * t) / ((R[0] - kp2[i].pt[0] * R[2]) * y))
        x1 = x3 * kp1[i].pt[0]
        x2 = x3 * kp1[i].pt[1]

        x_arr.append(x1)
        y_arr.append(x2)
        z_arr.append(x3)


#build the point cloud
def buildMap():
    global x_arr, y_arr, z_arr
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_arr, y_arr, z_arr)

    plt.show()



