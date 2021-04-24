
import  pandas  as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(open('E:/ML/ann/weatherAUS.csv','rb'))
df.head()
df.columns
df.describe()
df.info()
df.dtypes

import seaborn as sns
sns.countplot(x=df['RainTomorrow'])

corrmat=df.corr()
sns.heatmap(corrmat,annot=True,fmt='.1f')

df['Date']=pd.to_datetime(df["Date"])
df['year']=pd.DatetimeIndex(df['Date']).year
df["month"]=pd.DatetimeIndex(df['Date']).month
df["day"]=pd.DatetimeIndex(df["Date"]).day

df["month_normalized"]=2*np.pi*df["month"]/df["month"].max()
df["month_cosx"]=np.cos(df["month_normalized"])
df["month_sinx"]=np.sin(df["month_normalized"])

df["day_normalized"]=2*np.pi*df["day"]/df["day"].max()
df["day_cosx"]=np.cos(df["day_normalized"])
df["day_sinx"]=np.sin(df["day_normalized"])


string_var=(df.dtypes=='object')

obj_col=list(string_var[string_var].index)


for i in obj_col :
    print(i,df[i].isnull().sum())

for i in obj_col:
    df[i].fillna(df[i].mode()[0],inplace=True)

t = (df.dtypes == "float64")
num_cols = list(t[t].index)

for i in num_cols:
    print(i, df[i].isnull().sum())

for i in num_cols:
    df[i].fillna(df[i].median(), inplace=True)
    
df.info()

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
for i in obj_col:
    df[i]=label_encoder.fit_transform(df[i])
    
df.info()

x=df.drop(['RainTomorrow','Date','day','month','month_normalized','day_normalized'],axis=1)
y=df['RainTomorrow']

x_col=list(x.columns)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features=sc.fit_transform(x)

features=pd.DataFrame(features,columns=x_col)
sns.boxplot(data=features)
plt.xticks(rotation=90)

#Dropping the outlier

features = features[(features["MinTemp"]<2.3)&(features["MinTemp"]>-2.3)]
features = features[(features["MaxTemp"]<2.3)&(features["MaxTemp"]>-2)]
features = features[(features["Rainfall"]<4.5)]
features = features[(features["Evaporation"]<2.8)]
features = features[(features["Sunshine"]<2.1)]
features = features[(features["WindGustSpeed"]<4)&(features["WindGustSpeed"]>-4)]
features = features[(features["WindSpeed9am"]<4)]
features = features[(features["WindSpeed3pm"]<2.5)]
features = features[(features["Humidity9am"]>-3)]
features = features[(features["Humidity3pm"]>-2.2)]
features = features[(features["Pressure9am"]< 2)&(features["Pressure9am"]>-2.7)]
features = features[(features["Pressure3pm"]< 2)&(features["Pressure3pm"]>-2.7)]
features = features[(features["Cloud9am"]<1.8)]
features = features[(features["Cloud3pm"]<2)]
features = features[(features["Temp9am"]<2.3)&(features["Temp9am"]>-2)]
features = features[(features["Temp3pm"]<2.3)&(features["Temp3pm"]>-2)]

features.shape


sns.boxplot(data=features)
plt.xticks(rotation=90)

from sklearn.model_selection import train_test_split
xtest,xtrain,ytest,ytrain=train_test_split(x,y,test_size=0.2)


from keras.models import Sequential
from keras.layers import Dense,Dropout

model=Sequential()

 
#input layer
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26))

#hidden layers
model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#output layers
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history=model.fit(xtrain, ytrain, batch_size = 32, epochs = 150, validation_split=0.2)
history_df=pd.DataFrame(history.history)
plt.plot(history_df.loc[:,['loss']])
plt.plot(history_df.loc[:,['val_loss']])

plt.plot(history_df.loc[:,['accuracy']])
plt.plot(history_df.loc[:,['val_accuracy']])

ypred=model.predict(xtest)
ypred=(ypred>0.5)

from sklearn.metrics import confusion_matrix, classification_report
cf=confusion_matrix(ytest,ypred)
sns.heatmap(cf/np.sum(cf),annot=True)

classification_report(ytest,ypred)






























