
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[24]:


df=pd.read_csv(open("E:/ML/rnn/IBM_2006-01-01_to_2018-01-01.csv"),index_col='Date',)


# In[25]:


df


# In[14]:


df.head()
df.columns
df.info()


# In[8]:


df.isnull().sum()


# In[15]:


import matplotlib.pyplot as plt


# In[28]:


df['High'][:'2016'].plot(legend=True,figsize=(16,4))
df['High']['2017':].plot(legend=True,figsize=(16,4))
plt.show()


# In[26]:


xtrainset=df[:'2016'].iloc[:,1:2].values
xtestset=df['2017':].iloc[:,1:2].values


# In[72]:


xtestset.shape


# In[30]:


from sklearn.preprocessing import  MinMaxScaler


# In[71]:


sc=MinMaxScaler(feature_range=(0,1))
xtrainset_sc=sc.fit_transform(xtrainset)
xtrainset_sc.shape


# In[59]:


xtrain=[]
ytrain=[]
for i in range(60,2517):
    xtrain.append(xtrainset_sc[i-60:i,0])
    ytrain.append(xtrainset_sc[i,0])

xtrain=np.array(xtrain)
ytrain=np.array(ytrain)

xtrain=np.reshape(xtrain,(2457,60,1))
xtrain.shape


# In[64]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
model=Sequential()


# In[66]:


model.add(LSTM(units=50,input_shape=(60,1),return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
model.compile(optimizer='rmsprop',loss='mean_squared_error')
model.fit(xtrain,ytrain,epochs=50,batch_size=32)


# In[79]:


dataset_total = pd.concat((df["High"][:'2016'],df["High"]['2017':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(xtestset) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)

X_test = []
for i in range(60,311):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = model.predict(X_test)


# In[88]:


predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(df['2017':].iloc[:,1:2].values,color='r')
plt.plot(predicted_stock_price,color='b')


# In[97]:


plt.plot(df['2017':].iloc[:,1:2].values,color='r')
plt.plot(predicted_stock_price,color='b')

