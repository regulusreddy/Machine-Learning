# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:23:10 2017

@author: lalit
"""
from __future__ import division

#importing iris dataset to pandas DataFrame
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np



#creating a sigmoid function to be applied at each unit in layer 2 except bias unit and layer 3
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

#function to calculate a_2
def calculate_a_2(theta_1,a_1):
    z_2 = np.matmul(theta_1,a_1)
    a_2 = np.concatenate([sigmoid(z_2[0]),sigmoid(z_2[1])]).reshape(2,a_1.shape[1])
    return a_2

#creating new a_2 by adding bias term to a_2 to use to calculate a_3
def calculate_new_a_2(theta_1,a_1):
    a_2 = calculate_a_2(theta_1,a_1)
    new_a_2 = np.concatenate([np.ones(a_1.shape[1]).reshape(a_1.shape[1],1),a_2[0].reshape(a_1.shape[1],1),a_2[1].reshape(a_1.shape[1],1)]).reshape(3,a_1.shape[1])
    return new_a_2

#function to calculate a_3 from new_a_2
def calculate_a_3(theta_2,new_a_2,a_1):
    z_3 = np.matmul(theta_2,new_a_2)
    a_3 = np.matrix(sigmoid(z_3).reshape(1,a_1.shape[1]))
    return a_3

#compute cost function for logistic regression
def compute_cost(y,theta_2,new_a_2,a_1):
    m=99
    a_3 = calculate_a_3(theta_2,new_a_2,a_1)
    cost = np.dot(np.log10(a_3),y) + np.dot(np.log10(1.0 - a_3),(1.0-y))
    cost = -cost/m
    return cost

#computing the gradient or partial derivative term for theta_2
def compute_partial_derivate_theta_2(theta_1,theta_2,a_1,new_a_2,y):
    m=99
    a_3 = calculate_a_3(theta_2,new_a_2,a_1)
    delta_2 = (a_3-y.reshape(1,a_1.shape[1]))
    partial_derivative_theta_2 = np.dot((delta_2),np.transpose(new_a_2))
    partial_derivative_theta_2  = partial_derivative_theta_2/m
    return partial_derivative_theta_2

#computing the gradient or partial derivative term for theta_1
def compute_partial_derivate_theta_1(theta_1,theta_2,a_1,new_a_2,y):
    m=99
    new_a_2 = calculate_new_a_2(theta_1,a_1)
    a_3 = calculate_a_3(theta_2,new_a_2,a_1)
    delta_2 = (a_3-y.reshape(1,a_1.shape[1]))
    delta_1 = np.multiply(np.dot(np.transpose(theta_2),delta_2),np.multiply(new_a_2,(1-new_a_2)))
    delta_1
    #check the delta 1 first column 
    # we can remove this row as there is no change needed for bias input
    delta_1 = delta_1[1:3,:]
    delta_1.shape
    partial_derivative_theta_1 = np.dot((delta_1),np.transpose(a_1))
    partial_derivative_theta_1 = partial_derivative_theta_1/m
    return partial_derivative_theta_1

#function to predict output using forward propogation
def predict(theta_1,theta_2,a_1):
    new_a_2_predict = calculate_new_a_2(theta_1,a_1)
    a_3_predict = calculate_a_3(theta_2,new_a_2_predict,a_1)
    if (a_3_predict >= 0.5):
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
    #list to store error
    error =[]
    
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

        #adding the bias term to the input for test
        data_test_X = np.concatenate([np.ones(1),data_test_X])
        data_test_X
        
        #adding bias term to input for training
        a_1 = np.concatenate([np.ones(len(data_train_X)),data_train_X['petal length (cm)'].values, data_train_X['petal width (cm)'].values]).reshape(3,99)
        a_1.shape

        #output matrix
        y = np.matrix(data_train_Y).reshape(99,1)
        y.shape

        #creating a 2X3 NUMPY MATRIX Theta_1 for weights 
        theta_1 = np.random.rand(2,3)
        theta_1 = np.asmatrix(theta_1)
        theta_1

        #creating a 1X3 NUMPY MATRIX Theta_2 for weights 
        theta_2 = np.random.rand(1,3)
        theta_2 = np.asmatrix(theta_2)
        theta_2


        #creating numpy matrix with initial theta values for each theta_1 and theta_2
        theta_1_history = np.matrix(theta_1.ravel())
        theta_2_history = np.matrix(theta_2.ravel())


        #list to store cost over each iteration
        list_cost = []
        
        for i in range(1500):
        
            #CREATING A NUMPY MATRIX A_1 WITH bias as A_1[0] and input x1 and x2 as A_1[1] and A_1[2] respectively
            learning_rate = 1.2
            #calling calculate_a_2 function
            calculate_a_2(theta_1,a_1)
    
            #calling calculate_new_a_2 function and storing value to new_a_2
            new_a_2 = calculate_new_a_2(theta_1,a_1)
            
            #calling calculate_a_3 function
            calculate_a_3(theta_2,new_a_2,a_1)
        
            #computing cost and storing it to cost
            cost = compute_cost(y,theta_2,new_a_2,a_1)
            
            #appending cost at each iteration to list_cost
            list_cost.append(cost.item())
            
            #computing big_delta_3 and big_delta_2
            partial_derivative_2 = compute_partial_derivate_theta_2(theta_1,theta_2,a_1,new_a_2,y)
            partial_derivative_1 = compute_partial_derivate_theta_1(theta_1,theta_2,a_1,new_a_2,y)
           
            #updating theta_1 and theta_2 values
            theta_2 -= np.multiply(learning_rate,partial_derivative_2)
            theta_1 -= np.multiply(learning_rate,partial_derivative_1)
           
            #updating the theta_1_history and theta_2_history matrix
            theta_2_history = np.concatenate((theta_2_history,theta_2.ravel()),axis=0)
            theta_1_history = np.concatenate((theta_1_history,theta_1.ravel()),axis=0)
            #print(cost)
        #calling prediction function and storing     
        prediction = predict(theta_1,theta_2,data_test_X.reshape(3,1))
        #calculate error for each fold
        error_each_fold = np.abs(data_test_Y - prediction)
        #append error to list
        error.append(error_each_fold)
    error_matrix = np.matrix(error)
    error_rate = (int(round(np.sum(error_matrix))))/100
    print('Average Error Rate:',error_rate)

 