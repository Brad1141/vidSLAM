from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

import cv2
import numpy as np
import math

prevKP = []
prevDes = []
prevImg = []

def featureExtraction(frame):

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp, des = orb.detectAndCompute(frame, None)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(frame, kp, None, color=(0, 0, 255), flags=0)

    cv2.imshow('drive', img2)
    dataAssociation(frame)

def dataAssociation(currImg):
    global prevKP, prevImg, prevDes

    cv2.resize(currImg, (600, 600))

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    currKP, currDes = orb.detectAndCompute(currImg, None)

    if prevImg != []:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prevDes, currDes)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(prevImg, prevKP, currImg, currKP, matches[:10], currImg, flags=2)
        cv2.imshow('corners', img3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()



    prevKP = currKP
    prevDes = currDes
    prevImg = currImg

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

def buildMap(matches, kp1, kp2):

    points1 = []
    points2 = []

    #horizontal(x) and vertical(y) length of image from center
    x = 600 / 2
    y = 600 / 2


    for mat in matches:
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

    #focal lengths (assumes that the field of view is 60)
    f_x = x / math.tan(60 / 2)
    f_y = y / math.tan(60 / 2)

    #camera matrix
    K = np.array([[f_x, 0, x],
                  [0, f_y, y],
                  [0, 0, 1]])

    E = cv2.findEssentialMat(points1, points2, K)

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
        x3 = ((R[0] - kp2[i]))


    return

class coordinates:
    #x_ij: the 3D position of the point with respect to the camera
    #t_iw: the pose of the camera

    prevCamPos = [0, 0, 0]
    def __init__(self, x_ij, t_iw):
        self.x_ij = x_ij
        self.t_iw = t_iw

    def calculateCoord(self, xy, camPos):
        #calulate the rotation matrix
        v = np.cross(self.prevCamPos, camPos)
        #magnitude of v
        s = np.linalg.norm(v)
        c = np.asmatrix(self.prevCamPos) * np.asmatrix(camPos)

        v_x = np.array([0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0])

        R_iw = np.identity(3) + v_x + (v_x * v_x * ((1 - c)/(s * s)))

        robotPos = R_iw * x_wj + t_iw



