# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:52:02 2017

@author: lalit
"""
#divison was giving an approximation error 
#imported divison from python3
from __future__ import division
#importing iris dataset to pandas DataFrame
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np




#sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

theta = np.zeros((3,1))
theta

#cost function for logistic regression
def cost_function(data_logistic,data_logistic_target,theta):
    m = len(data_logistic_target)
    h = sigmoid(np.dot(np.transpose(data_logistic),theta))
    cost = (-1.0/m) * ((np.dot(np.transpose(data_logistic_target),np.log10(h))) + (np.dot(np.transpose(1.0-data_logistic_target),np.log10(1-h))))
    return cost

#computing partial derivate theta (gradient)
def compute_partial_derivate_theta(data_logistic,data_logistic_target,theta):
    m = len(data_logistic_target)
    h = sigmoid(np.dot(np.transpose(data_logistic),theta))
    delta = h-data_logistic_target    
    partial_derivative_theta = np.dot(np.transpose(delta),np.transpose(data_logistic))
    partial_derivative_theta  = partial_derivative_theta/m
    return partial_derivative_theta

#function to predict
def predict(data_logistic,theta):
    h_predict = sigmoid(np.dot(np.transpose(data_logistic),theta))
    if (h_predict >= 0.5):
        return 1
    else:
        return 0    
    
#loading the dataset
data = load_iris()
data_1 = pd.DataFrame(data.data, columns=data.feature_names)
data_1['target'] = data['target']
data_1.head()
data_1.tail()

#dropping columns sepal length and sepal width
data_iris = data_1.drop(['sepal length (cm)', 'sepal width (cm)'], axis=1)
data_iris.head()

#dropping rows of setosa flower
data_iris = data_iris.drop(data_iris.index[:50])
data_iris.head()


#resetting the index of the DataFrame
data_iris = data_iris.reset_index(drop=True)
data_iris.head()

#performing min-max normalization
data_iris['petal length (cm)'] = (data_iris['petal length (cm)'] - min(data_iris['petal length (cm)']))/(max(data_iris['petal length (cm)']) - min(data_iris['petal length (cm)']))
data_iris['petal width (cm)'] = (data_iris['petal width (cm)'] - min(data_iris['petal width (cm)']))/(max(data_iris['petal width (cm)']) - min(data_iris['petal width (cm)']))
data_iris


#changing the targets to 0 for versicolor and 1 for virginica
data_iris.loc[data_iris['target'] == 1, 'target'] = 0
data_iris.loc[data_iris['target'] == 2, 'target'] = 1
data_iris



def main():
    #initialise list for error
    error =[]
    
    #leave one out cross validation    
    for value in range(100):
        
        print('Iteration',value)
        #selecting the row for test based on the index in range of 100    
        data_test_X = data_iris[['petal length (cm)','petal width (cm)']].iloc[value].values
        #setting the index of data_test_X to data_slice
        data_slice = data_iris.index.isin([value])
        #selecting other indices other than data_slice
        data_train_X = data_iris[~data_slice]
        #setting index of dataset to test
        data_test_Y = data_iris['target'][value]
        #selecting values of y in other indices for training
        data_train_Y = data_iris['target'][~data_slice] 
        
        #adding the bias term to the test data input
        data_test_X = np.concatenate([np.ones(1),data_test_X])
        data_test_X
    
        #intialise theta as zeros
        theta = np.zeros((3,1))
        #preparing the input dataset
        data_logistic = np.concatenate([np.ones(len(data_train_X['target'])),data_train_X['petal length (cm)'].values, data_train_X['petal width (cm)'].values]).reshape(3,data_train_X.shape[0])
        data_logistic.shape
    

        #output matrix
        data_logistic_target = np.matrix(data_train_Y).reshape(data_train_X.shape[0],1)
        data_logistic_target.shape
    
    
    
        #list store cost over the iterations
        list_cost_logistic = []
        
        for i in range(1500):
            learning_rate = 1.35
            
            #computing cost
            cost_logistic = cost_function(data_logistic,data_logistic_target,theta)
            #appending cost to list
            list_cost_logistic.append(cost_logistic.item())
            #calculating gradient
            partial_derivative_theta = compute_partial_derivate_theta(data_logistic,data_logistic_target,theta)
            #updating theta
            theta-= np.multiply(learning_rate,partial_derivative_theta.reshape(3,1))
            #print(list_cost_logistic)            
        #calling prediction function on test data and theta after training for 1500 iteration
        prediction = predict(data_test_X,theta)
        #calculating error in each fold of leave one out validation
        error_each_fold = np.abs(data_test_Y - prediction)
        error.append(error_each_fold)
        #print(error_each_fold)
    error_matrix = np.matrix(error)
    error_rate = (int(round(np.sum(error_matrix))))/100
    print('Average Error Rate:',error_rate)

