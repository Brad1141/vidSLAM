import DataVisual
import cv2

frame = cv2.imread('trees.png')

#get orb features in an image
#DataVisual.featureExtraction(frame)

#data association for landmarks
DataVisual.dataAssociation()




