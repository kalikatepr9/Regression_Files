#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:18:03 2020

@author: pranavkalikate
"""

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
x=dataset.iloc[:,[0,2]].values                
y=dataset.iloc[:,4].values                  
                                             
#dummy variable must be created
#'STATE' is a categorical variables which consists of 'TEXTS' 
#we need to change it in numbers to avoid any issue"""

"""# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder            
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')     
x= np.array(ct.fit_transform(x), dtype=np.float)
                     
# Avoiding the Dummy Variable Trap
x = x[:, 1:]   
"""      

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

#predict new obseravation
new_observation=regressor.predict([[9000,10000]])
"""
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
plt.show()
"""