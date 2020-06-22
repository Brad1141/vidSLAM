import DataVisual
import cv2

cap = cv2.VideoCapture('drive.mp4')
count = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    count = count + 1
    # Display the resulting frame
    frame = cv2.resize(frame, (800, 600))

    DataVisual.featureExtraction(frame)

    if count >= 10:
      DataVisual.dataAssociation(frame)
      count = 0

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




