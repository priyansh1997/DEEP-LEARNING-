
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train=pd.read_csv('C:/Users/PRIYANSH/Desktop/datasets/Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
scaled_data=sc.fit_transform(training_set) 

xtrain=[]
ytrain=[]
for i in range (60,1258):
    xtrain.append(scaled_data[i-60:i,0])
    ytrain.append(scaled_data[i,0])

xtrain,ytrain=np.array(xtrain),np.array(ytrain)

#converting 2d array to 3d array    
xtrain.shape
xtrain=np.reshape(xtrain,(1198,60,1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model=Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(60,1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True, input_shape=(60,1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=100,batch_size=30)

dataset_test=pd.read_csv('C:/Users/PRIYANSH/Desktop/datasets/Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
xtest=[]
for i in range(60,80):
    xtest.append(inputs[i-60:i,0])
xtest=np.array(xtest)
xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))

pred_stock_price=model.predict(xtest)

pred_stock_price=sc.inverse_transform(pred_stock_price)

plt.plot(pred_stock_price,color='red')
plt.plot(real_stock_price,color='blue')
plt.show()
 


























