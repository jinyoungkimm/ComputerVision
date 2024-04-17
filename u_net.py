from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import os
import random
import cv2 as cv

# 데이터 출처 : https://robots.ox.ac.uk/~vgg/data/pets/
input_dir = "./Oxford_Pets(U-Net)/datasets/oxford_pets/images/images/"
target_dir = "./Oxford_Pets(U-Net)/datasets/oxford_pets/annotations/annotations/trimaps/"
img_siz = (160,160) # 모델에 입력되는 영상 크기
n_class = 3 # Segmentation Label - [1] : 물체(고양이 등), [2] : 배경, [3] : 경계(물체와 배경의 경계) 
batch_size = 32

#sorted() : 학습 데이터를 shuffle하기 위함
img_paths = sorted([os.path.join(input_dir,f) for f in os.listdir(input_dir) if f.endswith(".jpg")])
label_paths = sorted([os.path.join(target_dir,f) for f in os.listdir(target_dir) if f.endswith(".png") and not f.startswith(".")])

class OxfordPets(keras.utils.Sequence):

    def __init__(self,batch_size,img_size,img_paths,label_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.label_paths = label_paths
        


    def __len__(self):
        return len(self.label_paths) // self.batch_size # 전체 레이블 개수 / 배치 사이즈
    
    
    #__getitem__()으로 원하는 크기만큼의 Batch를 들고와서 학습을 시키게 하는 역할
    # ex) train_gen[2] , test_gen[2] -> 아래의 코드에서 두 변수가 나온다.
    # 이 예제에서는 __getitem__()이 사용되지 않는다(OxfordPets 객체에 INDEX or Slicing이 사용돼어야 __getitem__()이 호출된다)  
    def __getitem__(self,idx): #__getitem()에 대한 동작 원리는 Tistory 565 참조

        i = idx*self.batch_size

        #batch_img_paths = [train data]의 paths
        batch_img_paths = self.img_paths[i:i+self.batch_size] # OxfordPets()객체 생성시 생성자 변수에 batch_size=32를 넣고, idx가 1이라면, self.img_paths[32:32+32] 범위의 이미지를 train data로 학습시
                                                              # 시키겠다는 것.
        #batch_label_paths = [label data]의 paths 
        batch_label_paths = self.label_paths[i:i+self.batch_size]
         
        x = np.zeros((self.batch_size,) + self.img_size + (3,),dtype="float32") # x.shape == (32,160,160,3)(self.batch_size가 32인 경우!) , (32,160,160,3) == 컬러 이미지(160,160,3)이 32 장
        # [train data]의 paths를 참조하여 실제 train data(이미지)를 불러온다.                                                                        
        for j,path in enumerate(batch_img_paths):
            img = load_img(path,target_size=self.img_size) # self.img_size== (160,160)인 경우, 로드할 img의 size가 (160,160)이 아닌 경우, (160,160)으로 변환해준다. 
            x[j] = img    # x 객체는 train data가 된다.         

        y = np.zeros((self.batch_size,)+self.img_size+(1,),dtype='uint8') # 이미지 각 픽셀에 대한 [분할 레이블 값]을 저장
        for j,path in enumerate(batch_label_paths):
            img = load_img(path,target_size=(self.img_size),color_mode='grayscale')
            y[j] = np.expand_dims(img,2) # 결과 : img.shape == (160,160,1) -> (160,160)이던 [2차원] img를 (160,160,1)의 3차원의 img로 변환을 해준다.(모델의 output값 형태를 3차원으로 정했기 때문!)
            y[j] -= 1                    # 이때 2는 axis=2를 나타내는데, numpy에서는 z축을 담당한다.(이미지에서는 세로x가로x채널or깊이)
        
        return x,y
    

# makle_model()은 OxforPets 클래스 밖에서 정의된 메서드
def make_model(img_size,num_classes):

        inputs = keras.Input(shape=img_size + (3,)) # ex) shape = (160,160,3)

        #U-Net의 다운 샘플링(Contracting Path) - 축소 경로
        x = layers.Conv2D(32,3,strides=2,padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        previous_block_activation = x # 지름길 연결을 위함

        for filters in [64,128,256]:

            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(filters,3,padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(filters,3,padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3,strides=2,padding="same")(x)

            residual = layers.Conv2D(filters,1,strides=2,padding="same")(previous_block_activation)

            x = layers.add([x,residual])  # 지름길 연결이 일어나는 부분
            previous_block_activation = x # 지름길 연결을 위함

        

        for filters in [256,128,64,32]:
            
            x = layers.Activation('relu')(x)
            x = layers.Conv2DTranspose(filters,3,padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.Conv2DTranspose(filters,3,padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.UpSampling2D(2)(x)    

            residual = layers.UpSampling2D(2)(previous_block_activation)
            residual = layers.Conv2D(filters,1,padding="same")(residual)

            x = layers.add([x,residual]) # 지름길 연결이 일어나는 부분
            previous_block_activation = x # 지름길 연결을 위함


        outputs = layers.Conv2D(num_classes,3,activation='softmax',padding="same")(x)
        model = keras.Model(inputs,outputs)

        return model


model = make_model(img_siz,n_class)

random.Random(1).shuffle(img_paths)
random.Random(1).shuffle(label_paths)

# 10%를 Validation Set으로 하겠다!!
test_samples = int (len(img_paths) * 0.1)

#Training Data Set
train_img_paths = img_paths[:-test_samples]
train_label_paths = label_paths[:-test_samples]

#Validation Data Set
test_img_paths = img_paths[-test_samples:]
test_label_paths = label_paths[-test_samples:]


train_Data_Generation = OxfordPets(batch_size,img_siz,train_img_paths,train_label_paths)
test_Data_Generation = OxfordPets(batch_size,img_siz, test_img_paths,test_label_paths)

model.compile(optimizer='adam', loss = "sparse_categorical_crossentropy",metrics=['accuracy'])


call_back = [keras.callbacks.ModelCheckpoint('oxford_seg.h5', save_best_only=True)]
model.fit(train_Data_Generation, epochs=30, validation_data = test_Data_Generation, callbacks=call_back)

predicts = model.predict(test_Data_Generation)

cv.imshow("Sample image",cv.imread(test_img_paths[0]))
cv.imshow("Segmentation label",cv.imread(test_label_paths[0])*64)
cv.imshow("Segmentation prediction",predicts[0])

cv.waitKey()
cv.destroyAllWindows()
