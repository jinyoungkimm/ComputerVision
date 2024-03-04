
from PyQt5.QtWidgets import *
import sys
import cv2 as cv
import winsound

class Video(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("비디오에서 Frame 수집")
        self.setGeometry(200,200,500,100)

        videoButton = QPushButton("비디어 켜기",self) #버튼 생성
        captureButton = QPushButton("프레임 잡기",self)
        saveButton = QPushButton("프레임 저장",self)
        quitButton = QPushButton("나가기",self)

        videoButton.setGeometry(10,10,100,30) #각 버튼의 위치 및 크기 지정
        captureButton.setGeometry(110,10,100,30)
        saveButton.setGeometry(210,10,100,30)
        quitButton.setGeometry(310,10,100,30)

        videoButton.clicked.connect(self.videoFunction) #Call Back 함수 지정
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def videoFunction(self):
            
            self.cap = cv.VideoCapture(0,cv.CAP_DSHOW) # 카메라와 연결 시도
            if not self.cap.isOpened():
                self.close()


            while True:
                (ret,self.frame) = self.cap.read()
                if not ret :
                    break
                cv.imshow("video display",self.frame)
                cv.waitKey(1)
        

    def captureFunction(self):
            self.capturedFrame = self.frame
            cv.imshow("Captured Frame",self.capturedFrame)

    def saveFunction(self):
            file_name = QFileDialog.getSaveFileName(self,"파일저장","./")
            cv.imwrite(file_name[0],self.capturedFrame)

    def quitFunction(self):
            self.cap.release() # [카메라]와 연결을 끊음
            cv.destroyAllWindows()
            self.close()


app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()


1111111111










