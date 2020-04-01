#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 22:55:09 2019

@author: pranavkalikate
"""
                    #              ""MLR""
#  Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#  Importing Dataset
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values                
y=dataset.iloc[:,4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder            
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')     
x= np.array(ct.fit_transform(x), dtype=np.float)
                     
# Avoiding the Dummy Variable Trap
x = x[:, 1:]         

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)           

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression     
regressor = LinearRegression()           
regressor.fit(x_train, y_train)         

# Predicting the Test set results
y_pred = regressor.predict(x_test)       

#Building the optiomal model using Backward Elimination
#import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
#np.ones create 1 column with only 1's
#axis=1 for column,0 for rows

#actual backward elimination
#Compare x and x_opt index 
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y ,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y ,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y ,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y ,exog=x_opt).fit()
regressor_OLS.summary()

'''
x_opt=x[:,[0,3]]
regressor_OLS=sm.OLS(endog=y ,exog=x_opt).fit()
regressor_OLS.summary()
'''
#use variables obtained from backward elimination to build another model.