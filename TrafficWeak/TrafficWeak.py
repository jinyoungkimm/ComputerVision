import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound


class TrafficWeak(QMainWindow):

    def __init__(self):

        super().__init__()
        self.setWindowTitle("교통약자 보호")
        self.setGeometry(200,200,700,200)

        signButton = QPushButton("표지판 등록",self)
        roadButton = QPushButton("도로 영상 불러옴",self)
        recognitionButton = QPushButton("인식",self)
        quitButton = QPushButton("나가기",self)
        self.label = QLabel("환영합니다",self)

        signButton.setGeometry(10,10,100,30)
        roadButton.setGeometry(110,10,100,30)
        recognitionButton.setGeometry(210,10,100,30)
        quitButton.setGeometry(510,10,100,30)
        self.label.setGeometry(10,40,600,170)

        signButton.clicked.connect(self.signFuntion)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionButton)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [["child.png","어린이"],["elder.png","노인"],["disabled.jpg","장애인"]] #표지판 모델 영상
        self.signImgs=[] #표지판 모델 영상 저장


    def signFuntion(self):
        self.label.clear()
        self.label.setText("교통약자 표지판을 등록합니다.")

        for fileName,_ in self.signFiles:
            self.signImgs.append(cv.imread(fileName))
            cv.imshow(fileName,self.signImgs[-1])

    def roadFunction(self):

        if self.signImgs==[]:
            self.label.setText("먼저 표지판을 등록하세요")

        else:
            
            fileName = QFileDialog.getOpenFileName(self,"파일 읽기","./")
            self.roadImg = cv.imread(fileName[0])
            
            if self.roadImg is None:
                sys.exit("해당 파일을 찾을 수가 없습니다.")

            cv.imshow("Road Scene",self.roadImg)


    def recognitionButton(self):

        if self.roadImg is None:
            self.label.setText("먼저 도로 영상을 입력하세요")

        else: 
            sift = cv.SIFT_create() # Key Point와 Descriptor 추출을 위함.

            KD = [] # 3개의 [모델 영상(=표지판)]의 Key Point와 Descriptor 저장
            for img in self.signImgs:
                gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray,None)) # [모델 영상]의 Key Point와 Descriptor 추출

            # [장면 영상(=도로)]Key Point와 Descriptor 추출
            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY) # SIFT의 입력값은 [명암 영상]
            road_kp,road_des = sift.detectAndCompute(grayRoad,None) 

            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED) #Descriptor를 이용하여,  
            GM = [] #Key Point 중에서도 더 좋은 Key Point 찾기
            for (sign_kp,sign_des) in KD:
                knn_match = matcher.knnMatch(sign_des,road_des,2)
                T = 0.7
                good_match=[]
                for nearest1,nearest2 in knn_match:
                    if(nearest1.distance / nearest2.distance)<T:
                        good_match.append(nearest1)
                GM.append(good_match)
            best = GM.index(max(GM,key=len)) #매칭쌍 개수가 최대인 [모델 영상(=표지판)]

            print("asdfasdfff")

            if len(GM[best]) < 4: # 매칭쌍이 4개 미만이면 실패라고 판단할 거임!
                self.label.setText("표지판이 없습니다.")
            else:
                sign_kp = KD[best][0]
                good_match = GM[best]
            
                point1 = np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])
                point2 = np.float32([road_kp[gm.trainIdx].pt for gm in good_match])

                H,_ = cv.findHomography(point1,point2,cv.RANSAC)
                
                h1,w1 = self.signImgs[best].shape[0],self.signImgs[best].shape[1]
                h2,w2 = self.roadImg.shape[0], self.roadImg.shape[1]

                box1 = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(4,1,2)
                box2 = cv.perspectiveTransform(box1,H)

                self.roadImg=cv.polylines(self.roadImg,[np.int32(box2)],True,(0,255,0),4)

                print("asdfasdfff1111111111")

                img_match = np.empty((max(h1,h2),w1+w2,3),dtype=np.uint8)
                cv.drawMatches(self.signImgs[best],sign_kp,self.roadImg,road_kp,good_match,img_match,flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

                cv.imshow("Matches and Homography",img_match)

                self.label.setText(self.signFiles[best][1]+'보호구역입니다.')
                winsound.Beep(1000,300)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = TrafficWeak()
win.show()
app.exec_()




    


