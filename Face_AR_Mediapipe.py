import cv2 as cv
import mediapipe as mp

dice = cv.imread("dice2.png",cv.IMREAD_UNCHANGED)
print(dice.shape)
dice = cv.resize(dice,dsize=(0,0),fx=0.1,fy=0.1)
w,h = dice.shape[1], dice.shape[0]
print(dice.shape)        
  


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)

cap = cv.VideoCapture(0,cv.CAP_DSHOW)

while True:

    ret,frame = cap.read()
    if not ret:
        break

    res = face_detection.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))

    if res.detections:
        for det in res.detections:

            p = mp_face_detection.get_key_point(det,mp_face_detection.FaceKeyPoint.RIGHT_EYE)
            x1,x2 = int(p.x * frame.shape[1] - w//2), int(p.x*frame.shape[1] + w//2)
            y1,y2 = int(p.y * frame.shape[0] - h//2), int(p.y*frame.shape[0] + h//2)

            if x1 > 0 and y1>0 and x2 < frame.shape[1] and y2 < frame.shape[0]:

                alpha = dice[:,:,3:]/255

                frame[y1:y2+1,x1:x2+1] = frame[y1:y2+1,x1:x2+1]*(1-alpha)+dice[:,:,:3]*alpha

    
    cv.imshow('MediaPipe Face AR',cv.flip(frame,1))
    if cv.waitKey(5) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
