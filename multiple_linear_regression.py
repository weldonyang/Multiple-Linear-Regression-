# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
# sklearn will avoid dummy variable trap
# sklearn will pick the model with the lowest p-value 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# NOTE: Due to there being more than one independent variable, we cannot simply plot it on a 2D graph (in this case, we would need a 5D graph)
#       so we will display 2 vectors: 1. real profits of test set and 
#                                     2. predicted profits of test set 

# Predicting the Test set results
# np.set_printoptions(precision=2) displays any numerical values with 2 decimals 
# np.concatenate() expects 2 args: 1. tuple of arrays 
#                                  2. axis=0 or 1 (0 means vertical concatenation, 1 means horizontal )
# .reshape() is a function of regressor that takes 2 args: 1. # of rows
#                                                          2. # of columns and reshapes the vector
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)                                                      
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))                           

# Making a single prediction example
# R&D = 160,000
# Admin = 130,000
# Marketing = 300,000
# State = California [1, 0, 0]
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))