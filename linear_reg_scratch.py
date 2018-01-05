# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:26:32 2017

@author: lalit

"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pprint as pp

#Linear Regression from Scratch

data_reg = pd.read_csv("C:/Users/lalit/Desktop/linear_regression_test_data.csv")
data_reg

#removing first column (serial number #) from pandas df
data_reg=data_reg.loc[:,'x':'y_theoretical']
data_reg

#X values
X=data_reg['x'].values
X

#Y values
Y=data_reg['y'].values
Y

#computing mean
def mean(x):
    m = sum(x)/len(x)
    return m                  

mean(X)
mean(Y)

#computing variance
def variance(x):
    
        mean_x = mean(x)
        len1 = len(x)
        temp = 0
        
        for i in range(len1):
            temp += (x[i] - mean_x) * (x[i] - mean_x) 
        return temp / len1;

variance(X)

variance(Y)

#computing covariance
def covariance(x, y):
    size = len(x)
    covariance = 0.0
    for i in xrange(0, size):
        covariance += (x[i] - mean(x)) * (y[i] - mean(y))
    return covariance / float(size-1)


covariance(X,Y)

#computing coefficients b0 and b1
def coefficients(x,y):
    x_mean, y_mean = mean(x), mean(y)
    b1 = (covariance(x, y)) / variance(x)
    b0 = y_mean - (b1 * x_mean)
    return [b0, b1]


#assigning and printing b0 and b1
b0 = coefficients(X,Y)[0]
b0
b1 = coefficients(X,Y)[1]
b1

#building linear regression line
max = np.max(X) + 50
min = np.min(X) - 50
x_reg = np.linspace(min, max, 100)
y_reg = b0 + b1 * x_reg

#plotting the regression line
plt.plot(x_reg, y_reg, color='red', label='Regression Line')
plt.xlim(-2.5, 2.5), plt.ylim(-5, 5)
plt.legend()
plt.show()


#plotting y vs x; y theoretical vs x; pca axis and the regression line
plt.scatter(np.asarray(x), np.asarray(y), color="red", alpha=0.5,marker=(5,1), label='y vs x')
plt.scatter(np.asarray(x), np.asarray(y_theoretical), color="blue", alpha=0.5, marker="+", label='y theoretical vs x')
plt.plot([0, 10*pca_result['loadings'][0,0]], [0,10*pca_result['loadings'][1,0]],
            color='yellow', linewidth=2)
plt.plot([0, -10*pca_result['loadings'][0,0]], [0,-10*pca_result['loadings'][1,0]],
            color='yellow', linewidth=2, label='PC axis')
plt.title('y vs x; y theoretical vs x; pc1 axis and the regression line')
plt.plot(x_reg, y_reg, color='#58b970', label='Regression Line')
plt.xlim(-2.5, 2.5), plt.ylim(-5, 5)
plt.legend()
plt.show()