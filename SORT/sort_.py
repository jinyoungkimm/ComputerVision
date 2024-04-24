import numpy as np
import cv2 as cv
import sys


# 추적 알고리즘 = 검출 + 쌍맺기
# 검출 : YOLO 알고리즘 사용
# 쌍맺기 : SORT 알고리즘 사용


def construct_yolo_v3():

    f = open('./SORT/coco_names.txt','r')
    class_names = [line.strip() for line in f.readlines()]

    model = cv.dnn.readNet('./SORT/yolov3.weights','./SORT/yolov3.cfg')
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]

    return model,out_layers,class_names




def yolo_detect(img,yolo_model,out_layers):

    height,width = img.shape[0], img.shape[1]
    test_img = cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)

    yolo_model.setInput(test_img)
    output3 = yolo_model.forward(out_layers)

    boxes,conf,id = [],[],[]
    for output in output3:
        for vec85 in output:
            scores = vec85[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(vec85[0]*width), int(vec85[1]*height)
                w,h = int(vec85[2]*width), int(vec85[3]*height)
                x,y = int(center_x-w/2), int(center_y-h/2)
                
                boxes.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)

    
    ind = cv.dnn.NMSBoxes(boxes,conf,0.5,0.4)
    objects = [boxes[i] + [conf[i]] + [id[i]] for i in range(len(boxes)) if i in ind ]

    return objects


model,out_layers,class_names = construct_yolo_v3()
colors = np.random.uniform(0,255,size=(100,3))


from sort import Sort

sort = Sort()

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
if not cap.isOpened(): 
    sys.exit("카메라 연결 실패")

while True:

    ret,frame = cap.read()
    if not ret:
        sys.exit("프레임 획득에 실패")
    
    
    res  = yolo_detect(frame,model,out_layers)
    persons = [res[i] for i in range(len(res)) if res[i][5] == 0]

    if len(persons) == 0: # 현재 Frame에 Person이 전혀 탐지가 안 된 경우
        tracks = sort.update()

    else: # 현재 Frame에 Person이 탐지된 경우
        tracks = sort.update(np.array(persons))
    

    for i in range(len(tracks)):
        x1,y1,x2,y2,track_id = tracks[i].astype(int)
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2)
        cv.putText(frame,str(track_id),(x1+10,y1+40),cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)
    
    cv.imshow("Person tracking by SORT",frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
