
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

(xtrain,ytrain),(xtest,ytest)=cifar10.load_data()

xtrain=xtrain.reshape(xtrain.shape[0],32,32,3)
xtest=xtest.reshape(xtest.shape[0],32,32,3)

xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')

#normalizing the data by deviding from 255 as max pixels can be 255
xtrain/=255
xtest/=255

xtrain1=(xtrain>0.5)
xtest1=(xtest>0.5)

#one hot encoding using keras
classes=10
print("shape of ytrain before ohe",ytrain.shape)
ytrain=np_utils.to_categorical(ytrain,classes)
ytest=np_utils.to_categorical(ytest,classes)
print("shape of ytrain after ohe",ytrain.shape)

model=Sequential()

model.add(Conv2D(50,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape=(32,32,3)))

model.add(Conv2D(75,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(125,kernel_size=(3,3),strides=(1,1), padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#in the end we will use neural network for output matrix
model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

model.fit(xtrain1,ytrain,batch_size=128,epochs=10,validation_data=(xtest,ytest))




















