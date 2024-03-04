import cv2 as cv
import numpy as np


img = np.array([[0,0,0,0,0,0,0,0,0,0], # 10 x 10
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0],
                [0,0,0,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,0,0,0,0],
                [0,0,0,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]],dtype=np.float32)


ux = np.array([[-1,0,1]])
uy = np.array([-1,0,1]).transpose()


k = cv.getGaussianKernel(3,1) # 3 x 3의 1차원 Gaussian 필터 생성! 
g = np.outer(k,k.transpose()) # 3 x 3의 2차원 Gaussian 필터 생성!


dy = cv.filter2D(img,cv.CV_32F,uy)
dx = cv.filter2D(img,cv.CV_32F,ux)


print(dy.shape) # (10,10) : img의 모든 pixel에 대해 Filtering 
print(dx.shape) # (10,10) 


dyy = dy*dy
dxx = dx*dx
dyx = dy*dx
gdyy = cv.filter2D(dyy,cv.CV_32F,g)
gdxx = cv.filter2D(dxx,cv.CV_32F,g)
gdyx = cv.filter2D(dyx,cv.CV_32F,g)
C = (gdyy*gdxx - gdyx*gdyx) - 0.04*(gdyy+gdxx)*(gdyy+gdxx) # p169의 공식 5-8


for j in range(1,C.shape[0]-1):
    for i in range(1,C.shape[1]-1):
        print(sum(C[j,i] > C[j-1:j+2,i-1:i+2])) # numpy는 axix 0을 우선하여 연산을 한다. 
        if C[j,i] > 0.1 and sum(sum(C[j,i] > C[j-1:j+2,i-1:i+2])) == 8:
            img[j,i] = 9

np.set_printoptions(precision=2)
print(dy)
print(dx)
print(dyy)
print(dxx)
print(dyx)
print(gdyy)
print(gdxx)
print(gdyx)
print(C)
print(img)


popping = np.zeros([160,160],np.uint8) # 10 x 10인 C Map을 160 x 160으로 확대해서 윈도우에 display
for j in range(0,160):
    for i in range(0,160):
        popping[j,i] = np.uint8( (C[j//16,i//16] + 0.06 ) *700 ) # p170 참조

cv.imshow("popping",popping)
cv.waitKey()
cv.destroyAllWindows()
