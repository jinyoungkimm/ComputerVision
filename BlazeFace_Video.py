import cv2 as cv
import mediapipe as mp


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


face_detection = mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)

cap = cv.VideoCapture(0,cv.CAP_DSHOW)


while True:

    ret,frame = cap.read()
    if not ret:
        print("프레임 획득 실패")
        break
    
    res = face_detection.process(cv.cvtColor(frame,cv.COLOR_BGR2RGB))    

    if res.detections:
        for detection in res.detections:
            mp_drawing.draw_detection(frame,detection)

    cv.imshow("Face Detection By BlazeFace",frame)


    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
