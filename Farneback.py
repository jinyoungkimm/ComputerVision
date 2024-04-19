import numpy as np
import cv2 as cv
import sys

def draw_OpticalFlow(img,flow,step=16):
    
    for y in range(step//2,frame.shape[0],step):

        for x in range(step//2,frame.shape[1],step):

            dx,dy = flow[y,x].astype(np.int)

            if (dx*dx + dy*dy) > 1: # (dx,dy)의 원점에서부터의 길이
                cv.line(img,(x,y),(x+dx, y+dy),(0,0,255),2) # 큰 모션이 있는 곳은 빨간색

            else:
                cv.line(img,(x,y),(x+dx, y+dy),(0,255,0),2) # 작은 모션이 있는 곳은 초록색



cap = cv.VideoCapture(0,cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit("카메라 연결 실패")

prev = None
while(True):

    ret, frame = cap.read()
    if not ret : sys("Failed to get a frame")

    if prev is None:
        prev = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        continue


    curr = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0)

    draw_OpticalFlow(frame,flow)
    cv.imshow("Optical flow",frame)

    prev = curr

    key = cv.waitKey(1)
    if key == ord("q"):
        break


cap.release()
cv.destroyAllWindows()
