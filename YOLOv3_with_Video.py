import numpy as np
import cv2 as cv
import sys


def construct_yolo_v3():

    file = open("./coco_names.txt",'r') #yolov3는 80부류의 coco dataset으로 학습을 하였다. 
    
    class_names = [line.strip() for line in file.readlines()]

    model = cv.dnn.readNet('./yolov3.weights','./yolov3.cfg') # yolov3.cfg : yolov3의 신경망 구조에 대한 정보
    
    layer_names = model.getLayerNames() # 모델, 층, 부류 이름을 담은 객체 반환

    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()] # yolov3의 3개의 outputlayer 추출([컴퓨터 비전과 딥러닝p386 참조])

    return model, out_layers, class_names


def yolo_detect(img,yolo_model,out_layers):


    height, width = img.shape[0], img.shape[1] # 코드 라인 ~에서 정규화된 좌표값을 원래 값으로 복원할 때 사용된다. 
    test_img = cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True) # swapRB=True : RGB를 BGR로 변환


    yolo_model.setInput(test_img) # yolov3 모델에 Input Image(test_img) 입력
    output3 = yolo_model.forward(out_layers) # Input Image에 대한 forward computation(전방 계산)!
                                            # 결과적으로, [BBox 정보, 신뢰도, 부류 확률]에 대한 정보를 최종 output 받는다. 
    
    box,conf,id = [],[],[]
    for output in output3:
        for vec85 in output:
            scores = vec85[5:] # index 0 ~ 4는 [BBox 좌표(0~3)]와 [신뢰도(4)]에 해당
            class_id = np.argmax(scores) # 부류 확률 중 최대값인 index를 반환!
            confidence = scores[class_id]
            
            if confidence >= 0.5: #신뢰도 0.5 미만에 대한 결과는 skip
                center_x,center_y = int(vec85[0]*width), int(vec85[1]*height)
                w,h = int(vec85[2]*width), int(vec85[3]*height)
                x,y = int(center_x - w/2), int(center_y - h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)

    ind = cv.dnn.NMSBoxes(box,conf,0.5,0.2) # 특정 A 물체에 대한 BBox 박스가 중복되 있는 경우, 그 중복을 제거하는 역할
    objects = [box[i] + [conf[i]] + [id[i]] for i in range(len(box)) if i in ind]

    return objects


model, out_layers, class_names = construct_yolo_v3() # YOLO 모델 생성
colors = np.random.uniform(0,255,size=(len(class_names),3)) # 부류마다 색깔 지정


cap = cv.VideoCapture(0,cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit("카메라 연결 실패")

import time

# 비디오의 FPS를 가져오기
fps = cap.get(cv.CAP_PROP_FPS)
n_frame = 0
start = time.time()

while True:

    ret,frame = cap.read()
    
    if not ret:
        sys.exit("Frame 획득에 실패하여 Loop를 나갑니다.")
    
    result = yolo_detect(frame,model,out_layers)

    for i in range(len(result)):
        
        x1,y1,x2,y2,confidence,id = result[i]

        text = str(class_names[id]+'%.3f'%confidence)

        cv.rectangle(frame,(x1,y1),(x2,y2),colors[id],2)

        cv.putText(frame,text,(x1,y1+30),cv.FONT_HERSHEY_PLAIN,1.5,colors[id],2)
    
    n_frame += 1
    end = time.time()
     # 프레임 속도를 표시하기 위해 프레임에 텍스트 추가
    cv.putText(frame, f"FPS: {(n_frame / (end-start))}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("Object detection from video by YOLO v.3",frame)
    

    key = cv.waitKey(1)

    if key == ord('q'):
        break





print('처리한 프레임 수 ',n_frame ,', 경과 시간', end-start, "\n초당 프레임 수=",n_frame/(end-start) )

cap.release()
cv.destroyAllWindows()










    