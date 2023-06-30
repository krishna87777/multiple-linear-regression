# Multiple Linear Regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0) # X_train - independent variables of the training set, Y_train - dependent; teest size - 20 perent of these all oberservation go insto test set

#class of skitlearn will take care of everything - like backward elimination,dummy varaiable trap
from sklearn.linear_model import LinearRegression #LinearRegression is a instance or an object of sklearn library
# regression and classification diffrence, regress. is predict the  real value like salary,classification is predict the real value like category or class
regressor = LinearRegression()
# we use fit method to train regression model;
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2) #calling numpy this will display any numerical value with only two decimals after diploma
print(np.concatenate((Y_pred.reshape(len(Y_pred),1),Y_test.reshape(len(Y_test),1)),1)) #(displaying two vectors for that we use concatinate function, reshape iis an attribute function thagt that allow to reshape vectors or array

