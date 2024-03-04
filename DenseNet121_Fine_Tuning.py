from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import image_dataset_from_directory
import pathlib

data_path = pathlib.Path('./datasets/standford_dogs/imags/images')

train_datasets = image_dataset_from_directory(data_path,validation_split=0.2,subset='training',seed=123,image_size=(224,224),batch_size=16)
test_datasets = image_dataset_from_directory(data_path,validation_split=0.2,subset='validation',seed=123,image_size=(224,224),batch_size=16)


backBorn_model = DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3)) #pre-trained-Model

cnn_model = Sequential()
cnn_model.add(Rescaling(1.0/255.0))
cnn_model.add(backBorn_model) # pre-train-Model
cnn_model.add(Flatten())

cnn_model.add(Dense(1024,activation='relu'))
cnn_model.add(Dropout(0.75))
cnn_model.add(Dense(units=120,activation='softmax'))

cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(learning_rate=0.000001),metrics=['accuracy'])
history = cnn_model.fit(train_datasets,verbose=2)

print('정확률 : ', cnn_model.evaluate(test_datasets,verbose=0)[1] * 100)

cnn_model.save('cnn_for_stanford_dogs.h5')

import pickle
file = open('dog_species_names.txt','wb')
pickle.dump(train_datasets.class_names,file)
file.close()


import matplotlib.pyplot as plt

plt.plot(history.history(['accuracy']))
plt.plot(history.history(['val_accuracy']))
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()


plt.plot(history.history(['loss']))
plt.plot(history.history(['val_loss']))
plt.title('Loss Graph')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()


















