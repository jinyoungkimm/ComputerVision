from PyQt5.QtWidgets import *
import winsound
import sys

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("삑 소리 내기")
        self.setGeometry(200,200,500,100)
        
        shortBeppButton = QPushButton("짧게 삑",self) #버튼 생성
        longBeepButton = QPushButton("길게 삑",self)
        quitButton = QPushButton("나가기",self)

        self.label=QLabel("환영합니다.",self) 
        shortBeppButton.setGeometry(10,10,100,30) #각 버튼의 위치와 크기 지정
        longBeepButton.setGeometry(110,10,100,30)
        quitButton.setGeometry(210,10,100,30)
        self.label.setGeometry(10,40,500,70)

        shortBeppButton.clicked.connect(self.shortBeepFunction) # CallBack 함수 지정
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)

    
    def shortBeepFunction(self):
        self.label.setText("주파수 1000으로 0.5초 동안 삑 소리를 냅니다.")
        winsound.Beep(1000,500)


    def longBeepFunction(self):
        self.label.setText("주파수 1000으로 3초 동안 삑 소리를 냅니다.")
        winsound.Beep(1000,3000)

    def quitFunction(self):
        self.close()

app = QApplication(sys.argv)
win=BeepSound()
win.show()
app.exec_()











