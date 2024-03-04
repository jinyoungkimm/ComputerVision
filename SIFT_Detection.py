import cv2 as cv


img = cv.imread("soccer.png")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


sift = cv.SIFT_create()
kp,des = sift.detectAndCompute(gray,None)


gray = cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT",gray)


cv.waitKey()
cv.destroyAllWindows()
