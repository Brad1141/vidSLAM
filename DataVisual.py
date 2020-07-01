import cv2

prevKP = []
prevDes = []

def featureExtraction(frame):
    #resize the image
    frame = cv2.resize(frame, (800, 600))

    # Initiate ORB
    orb = cv2.ORB_create()

    # find the keypoints and descriptors
    kp, des = orb.detectAndCompute(frame, None)

    # draw keypoints
    img2 = cv2.drawKeypoints(frame, kp, None, color=(255, 0, 0), flags=0)

    #Display results on screen
    cv2.imshow('vidSLAM', img2)

def dataAssociation():

    #get images
    img1 = cv2.imread('frame1.png')
    img2 = cv2.imread('frame2.png')

    img1 = cv2.resize(img1, (600, 800))
    img2 = cv2.resize(img2, (600, 800))

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    #creates the brute force matcher
    #pass in NORM_HAMMING because ORB uses binary string descriptors
    #set crossCheck equal to True for better accuracy
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    #best matches got to the front of the matches array
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img1, flags=2)
    cv2.imshow('Data association', img3)
    cv2.waitKey()
