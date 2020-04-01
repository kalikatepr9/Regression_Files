#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:34:04 2019

@author: pranavkalikate
"""

#Import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values                
y=dataset.iloc[:,2].values                  



"""# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)           #20-10 for train set , test set resp
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


# Fitting Linear Regression to the dataset  #For comparison
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures          
poly_reg = PolynomialFeatures( degree = 4 )     
x_poly = poly_reg.fit_transform(x)              
poly_reg.fit(x_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')           
plt.plot(x, lin_reg.predict(x), color = 'blue')      
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results       FOR much better prediction
x_grid=np.arange(min(x),max(x),0.1)   #for better curve
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6]])                            

# Predicting a new result with Polynomial Regression  
lin_reg_2.predict(poly_reg.fit_transform([[6]]))       
