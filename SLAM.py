import DataVisual
import numpy as np

def SLAM(frame):
    DataVisual.featureExtraction(frame)

def getPos(LM):
    return 0

class EKF():
    def __int__(self):
        self.prevX = 0
        self.prevY = 0
        self.prevZ = 0

        self.prev_PK = []


    def predictState(self, currLM, prevLM):
        if prevLM:
            #models
            #xk is a model of the previous state
            xk = np.array([self.prevX, self.prevY, self.prevZ])
            for i in range(len(prevLM)):
                np.append(xk, prevLM[i][0])
                np.append(xk, prevLM[i][1])
                np.append(xk, prevLM[i][2])

            #zk is a model of the current state
            zk = getPos(currLM)

            #predict
            Fk = []
            prev_Q = []
            Hk = []
            R = []
            Pk = Fk  * self.prev_Pk * Fk.T + prev_Q


            #update
            #caculate kalman gain
            Gk = Pk * Hk.T * (Hk * Pk * Hk.T + R).I
            xk = xk + Gk * (zk - xk)

        self.prev_Pk = (I - Gk * Hk)*Pk
        self.prevX = xk[0]
        self.prevY = xk[1]
        self.prevZ = xk[2]
        prevLM = currLM

        return xk
