
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

x=np.load('E://ML//ann//X.npy')
y=np.load('E://ML//ann//Y.npy')
y=pd.DataFrame(y)

plt.subplot(1,2,1)
plt.imshow(x[260].reshape(64,64))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x[900].reshape(64,64))
plt.axis('off')



x=np.concatenate((x[204:409],x[822:1027]),axis=0)
y=np.concatenate((np.zeros(205), np.ones(205)), axis=0).reshape(x.shape[0],1)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
xtrainf=xtrain.reshape(xtrain.shape[0],64*64)
xtestf=xtest.reshape(xtest.shape[0],64*64)

xtrain=xtrainf.T
xtest=xtestf.T
ytrain=ytrain.T
#to reduce the time we are transposing the matrix
#cost and loss function are same


from keras.models import Sequential
from keras.layers import Dense

#naming it as classifier. using sequencial module for initialization

#Initializing Neural Network
classifier = Sequential()


#adam is the code for reducing the learning rate when it converges and comes near to the actual value


# Adding the input layer and the first hidden layer
classifier.add(Dense(units= 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4096))
# Adding the second hidden layer
classifier.add(Dense(units= 4, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#optimizer:-use to find optimal set of weights. algo is stochastic gradient descent(SGD).
#we will use adam algo in SGD.
#sgd depnds on loss thus our second parameter is loss.
#our dependent variable is binary so we have to use binary_crossentropy.
#for more than 2 category use categorical_crossentropy.
#to improve performance of neural network add metrics as accuracy


# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



classifier.fit(xtrainf, ytrain, batch_size = 50, epochs = 60)
#here xtrainf and ytrain must have same number of rows
#predicting the test result.
#prediction function gives us the probability of the customer leaving the company.
 
# Predicting the Test set results
y_pred = classifier.predict(xtestf)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(ytest, y_pred)
acc = accuracy_score(ytest, y_pred)
print("the accuracy of my model is:-",acc)

#finding the accuracy of model.

