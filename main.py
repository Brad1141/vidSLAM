import DataVisual
import cv2
import numpy as np


cap = cv2.VideoCapture('drive.mp4')
count = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    count = count + 1
    # Display the resulting frame
    frame = cv2.resize(frame, (600, 600))
    cv2.imshow('drive', frame)

    if count >= 60:
      DataVisual.dataAssociation(frame)
      count = 0


    # Display the resulting frame
    frame = cv2.resize(frame, (600, 600))
    cv2.imshow('drive', frame)

    DataVisual.dataAssociation(frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

DataVisual.buildMap()

# Closes all the frames
cv2.destroyAllWindows()




