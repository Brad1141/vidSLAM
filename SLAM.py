import DataVisual
import numpy as np

def SLAM(frame):
    DataVisual.featureExtraction(frame)

def getPos(LM):
    return 0

def EKF(currLM):
    #models
    #xk is a model of the previous state
    xk = np.array([prevX, prevY, prevZ])
    for i in range(len(prevLM)):
        np.append(xk, prevLM[i].x)
        np.append(xk, prevLM[i].y)
        np.append(xk, prevLM[i].z)

    #zk is a model of the current state
    zk = getPos(currLM)

    #predict
    Pk = Fk  * prev_Pk * Fk.T + prev_Q


    #update
    #caculate kalman gain
    Gk = Pk * Hk.T * (Hk * Pk * Hk.T + R).I
    xk = xk + Gk * (zk - xk)

    prev_Pk = (I - Gk * Hk)*Pk
    prevX = xk[0]
    prevY = xk[1]
    prevZ = xk[2]
    prevLM = currLM
