import DataVisual
import numpy as np
import cv2
import math


# def SLAM(frame):
#     DataVisual.featureExtraction(frame)

def getPos(points1, points2):
    x = 800 / 2
    y = 600 / 2

    # focal lengths (assumes that the field of view is 60)
    f_x = x / math.tan(60 / 2)
    f_y = y / math.tan(60 / 2)

    # camera matrix
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

    pts, rr, tt, mask2 = cv2.recoverPose(E, np.float32(points1), np.float32(points2), K, R, t, mask)

    # compute the 3d coordinate of the camera
    x3 = ((rr[0] - (pts * rr[2]) * tt) / ((rr[0] - pts * rr[2]) * y))
    x1 = x3 * pts
    x2 = x3 * pts

    return [x1, x2, x3]


prevX = 0
prevY = 0
prevZ = 0
prev_Pk = 0.05

def predictState(currLM, prevLM, points1, points2):
    print(currLM)
    global prevX, prevY, prevZ, prev_Pk
    # models
    # xk is a model of the previous state
    xk = np.array([])
    np.append(xk, prevX)
    np.append(xk, prevY)
    np.append(xk, prevZ)

    for i in range(len(prevLM)):
        np.append(xk, prevLM[i][0])
        np.append(xk, prevLM[i][1])
        np.append(xk, prevLM[i][2])

    xk = np.asmatrix(xk)

    # zk is a model of the current state
    camX, camY, camZ = getPos(points1, points2)
    zk = np.array([])
    np.append(zk, camX)
    np.append(zk, camY)
    np.append(zk, camZ)

    for i in range(len(currLM)):
        np.append(zk, currLM[i][0])
        np.append(zk, currLM[i][1])
        np.append(zk, currLM[i][2])

    zk = np.asmatrix(zk)

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
    C = np.asmatrix(C)

    Q = W * C * W.T

    Q = np.asmatrix(Q)

    Hk = []
    for i in range(currLM):
        x_change = (currLM[i][0] - prevLM[i][0]) / (zk[0] - prevX)
        y_change = (currLM[i][1] - prevLM[i][1]) / (zk[1] - prevY)
        z_change = (currLM[i][2] - prevLM[i][2]) / (zk[2] - prevZ)
        Hk.append([x_change, y_change, z_change])

    Hk = np.asmatrix(Hk)

    R = np.array([[1, 0],
                  [0, 1]])
    R = np.asmatrix(R)

    Pk = Fk * prev_Pk * Fk.T + Q
    Pk = np.asmatrix(Pk)

    # update
    # caculate kalman gain
    Gk = Pk * Hk.T * (Hk * Pk * Hk.T + R).I
    Gk = np.asmatrix(Gk)
    xk = xk + Gk * (zk - xk)

    # 3x3 indentity matrix
    I = np.asmatrix(np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]))

    prev_Pk = (I - Gk * Hk) * Pk
    prevX = xk[0]
    prevY = xk[1]
    prevZ = xk[2]

    return xk
