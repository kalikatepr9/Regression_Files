#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:59:53 2020

@author: pranavkalikate
"""

                       #SIMPLE LINEAR REGRESSION

#data preprocessing

#Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,[0]].values                
y=dataset.iloc[:,1].values                  

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)           

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler      #Scaling not required in SLR..Library will take care of it
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
#Scaling not required in SLR..Library will take care of it"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression           
regressor = LinearRegression()                              
regressor.fit(x_train, y_train)       

"""# Applying k-Fold Cross Validation (model evaluation)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()"""                      
                                                         
#Predicting the test set results
y_pred=regressor.predict(x_test)                  

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')                     
plt.plot(x_train, regressor.predict(x_train), color = 'blue')            
plt.title('Salary vs Experience (Training set)')                         
plt.xlabel('Years of Experience')                  
plt.ylabel('Salary')
plt.show()                          

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')                              
plt.plot(x_train, regressor.predict(x_train), color = 'blue')           
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show

#New Prediction
new_observation=regressor.predict([[6.5]])