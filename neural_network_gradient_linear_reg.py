# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:54:05 2017

@author: lalit
"""

#neural network from scratch
#Network with input layer, one hidden layer, and output layer
#Input layer has two units and one bias unit
#Hidden layer has two units and one bias unit
#Output layer has two units

import numpy as np
import math as mt
import matplotlib.pyplot as plt
from itertools import chain




#creating a sigmoid function to be applied at each unit in layer 2 except bias unit and layer 3
def sigmoid(z):
  return 1 / (1 + mt.exp(-z))

#function to calculate a_2
def calculate_a_2(theta_1,a_1):
    z_2 = np.matmul(theta_1,a_1)
    a_2 = np.matrix([sigmoid(z_2[0].item()),sigmoid(z_2[1].item())]).reshape(2,1)
    return a_2

#creating new a_2 by adding bias term to a_2 to use to calculate a_3
#using numpy.random.rand(1).item(0) to get the value of the randomly created array of length 1 and setting this to a_0_2 which is the bias term
def calculate_new_a_2(theta_1,a_1):
    new_a_2 = np.matrix([1,calculate_a_2(theta_1,a_1)[0].item(),calculate_a_2(theta_1,a_1)[1].item()]).reshape(3,1)
    return new_a_2

#function to calculate a_3 from new_a_2
def calculate_a_3(theta_2,new_a_2):
    z_3 = np.matmul(theta_2,new_a_2)
    a_3 = np.matrix([sigmoid(z_3[0].item()),sigmoid(z_3[1].item())]).reshape(2,1)
    return a_3

#Using mean square error as the cost function
def compute_cost(y,theta_2,new_a_2):
    J_theta = np.multiply(0.5,((y[0].item(0)-calculate_a_3(theta_2,new_a_2)[0].item())**2+(y[1].item()-calculate_a_3(theta_2,new_a_2)[1].item())**2))
    return J_theta

#function to calculate small delta_3 and big delta_3
def compute_delta_3(theta_2,new_a_2,y):
    a_3 = calculate_a_3(theta_2,new_a_2)
    #computing small delta_3 for layer 3
    J_theta_wrt_z_1_3 = (a_3[0].item()-y[0].item())*((a_3[0].item())*(1-a_3[0].item()))
    J_theta_wrt_z_2_3 = (a_3[1].item()-y[1].item())*((a_3[1].item())*(1-a_3[1].item()))
    small_delta_3 = np.matrix([J_theta_wrt_z_1_3,J_theta_wrt_z_2_3]).reshape(2,1)
    small_delta_3
    #computing big delta_3 for layer 3
    big_delta_3= np.dot(small_delta_3,np.transpose(new_a_2))
    big_delta_3
    return big_delta_3


def compute_delta_2(theta_1,theta_2,a_1,new_a_2,y):
    a_3 = calculate_a_3(theta_2,new_a_2)
    a_2 = calculate_a_2(theta_1,a_1)
    
    #computing small delta_2 for layer 2
    # J1 with respect to z_1_2
    J1_theta_wrt_z_1_2 = (a_3[0].item()-y[0].item())*((a_3[0].item())*(1-a_3[0].item()))*(theta_2[0,1])*((a_2[0].item())*(1-a_2[0].item()))
    J1_theta_wrt_z_1_2
    # J2 with respect to z_1_2
    J2_theta_wrt_z_1_2 = (a_3[1].item()-y[1].item())*((a_3[1].item())*(1-a_3[1].item()))*(theta_2[1,1])*((a_2[0].item())*(1-a_2[0].item()))
    J2_theta_wrt_z_1_2
    # J with respect to z_1_2
    J_theta_wrt_z_1_2 = J1_theta_wrt_z_1_2 + J2_theta_wrt_z_1_2 
    J_theta_wrt_z_1_2 
    
    # J1 with respect to z_2_2
    J1_theta_wrt_z_2_2 = (a_3[0].item()-y[0].item())*((a_3[0].item())*(1-a_3[0].item()))*(theta_2[0,2])*((a_2[1].item())*(1-a_2[1].item()))
    J1_theta_wrt_z_2_2

    # J2 with respect to z_1_2
    J2_theta_wrt_z_2_2 = (a_3[0].item()-y[0].item())*((a_3[0].item())*(1-a_3[0].item()))*(theta_2[1,2])*((a_2[1].item())*(1-a_2[1].item()))
    J2_theta_wrt_z_2_2
    
    # J with respect to z_2_2
    J_theta_wrt_z_2_2 = J1_theta_wrt_z_2_2 + J2_theta_wrt_z_2_2  
    J_theta_wrt_z_2_2 

    #computing small delta_2 for layer 2
    small_delta_2 = np.matrix([J_theta_wrt_z_1_2,J_theta_wrt_z_2_2]).reshape(2,1)
    small_delta_2
    #computing big delta_2 for layer 2
    big_delta_2= np.dot(small_delta_2,np.transpose(a_1))
    big_delta_2
    
    return big_delta_2


#list for cost_computation
def main():
    #CREATING A NUMPY MATRIX A_1 WITH bias as A_1[0] and input x1 and x2 as A_1[1] and A_1[2] respectively
    a_1 = np.matrix([1,0.05,0.1]).reshape(3,1)
    a_1

    #output matrix
    y = np.matrix([0.01,0.99]).reshape(2,1)
    y


    #creating a 2X3 NUMPY MATRIX Theta_1 for weights 
    theta_1 = np.random.rand(2,3)
    theta_1 = np.asmatrix(theta_1)
    theta_1


    #creating a 2X3 NUMPY MATRIX Theta_2 for weights 
    theta_2 = np.random.rand(2,3)
    theta_2 = np.asmatrix(theta_2)
    theta_2

    #creating a empty list to store cost of each iteration
    list_cost = []
   
    #creating numpy matrix with initial theta values for each theta_1 and theta_2
    theta_1_history = np.matrix(theta_1.ravel())
    theta_2_history = np.matrix(theta_2.ravel())
    
    #forward propogation and backward propogation iterations
    for i in range(25000):
        #setting the learning as 0.14, learning rate can be between 0 and 1
        learning_rate = 0.14
        
        #calling calculate_a_2 function
        calculate_a_2(theta_1,a_1)
       
        #calling calculate_new_a_2 function and storing value to new_a_2
        new_a_2 = calculate_new_a_2(theta_1,a_1)
        
        #calling calculate_a_3 function
        calculate_a_3(theta_2,new_a_2)
        
        #computing cost and storing it to cost
        cost = compute_cost(y,theta_2,new_a_2)
        
        #appending cost at each iteration to list_cost
        list_cost.append(cost)
        
        #computing big_delta_3 and big_delta_2
        big_delta_3 = compute_delta_3(theta_2,new_a_2,y)
        big_delta_2 = compute_delta_2(theta_1,theta_2,a_1,new_a_2,y)
       
        #updating theta_1 and theta_2 values
        theta_1 -= np.multiply(learning_rate,big_delta_2)
        theta_2 -= np.multiply(learning_rate,big_delta_3)
        
        #updating the theta_1_history and theta_2_history matrix
        theta_1_history = np.concatenate((theta_1_history,theta_1.ravel()),axis=0)
        theta_2_history = np.concatenate((theta_2_history,theta_2.ravel()),axis=0)
        
        
        #because of large iterations you cant comment the following print statment as it will take less time 
        print list_cost
    
    #printing the theta_1_history     
    print (theta_1_history)
    
    #printing the theta_2_history     
    print (theta_2_history)
    
    #number of iterations
    x= range(1,25001)
    
    #iterations for thetas is 25002 because we are appending intialvalue of theta to matrix which is why its value is one more than x
    y= range(1,25002)
    
    #plotting total cost vs iterations      
    plt.plot(x,list_cost)
    
    #just getting the values from numpy matrix
    theta_10_1= list(chain.from_iterable(theta_1_history[:,0].ravel().tolist()))
    theta_11_1= list(chain.from_iterable(theta_1_history[:,1].ravel().tolist()))
    theta_12_1= list(chain.from_iterable(theta_1_history[:,2].ravel().tolist()))
    theta_20_1= list(chain.from_iterable(theta_1_history[:,3].ravel().tolist()))
    theta_21_1= list(chain.from_iterable(theta_1_history[:,4].ravel().tolist()))
    theta_22_1= list(chain.from_iterable(theta_1_history[:,5].ravel().tolist()))
    theta_10_2= list(chain.from_iterable(theta_2_history[:,0].ravel().tolist()))
    theta_11_2= list(chain.from_iterable(theta_2_history[:,1].ravel().tolist()))
    theta_12_2= list(chain.from_iterable(theta_2_history[:,2].ravel().tolist()))
    theta_20_2= list(chain.from_iterable(theta_2_history[:,3].ravel().tolist()))
    theta_21_2= list(chain.from_iterable(theta_2_history[:,4].ravel().tolist()))
    theta_22_2= list(chain.from_iterable(theta_2_history[:,5].ravel().tolist()))
   
    
   
    
    #plotting theta_10_1 vs iterations
    plt.plot(y,theta_10_1)
    plt.xlabel('iterations')
    plt.ylabel('theta_10_1')
    plt.title('theta_10_1 vs iterations')
    plt.show()
    
    
 
    
    #plotting theta_12_1 vs iterations
    plt.plot(y,theta_12_1)
    plt.xlabel('iterations')
    plt.ylabel('theta_12_1')
    plt.title('theta_12_1 vs iterations')
    plt.show()

    #plotting theta_20_1 vs iterations
    plt.plot(y,theta_20_1)
    plt.xlabel('iterations')
    plt.ylabel('theta_20_1')
    plt.title('theta_20_1 vs iterations')
    plt.show()
    
    #plotting theta_21_1 vs iterations
    plt.plot(y,theta_21_1)
    plt.xlabel('iterations')
    plt.ylabel('theta_21_1')
    plt.title('theta_21_1 vs iterations')
    plt.show()
    
    #plotting theta_22_1 vs iterations
    plt.plot(y,theta_22_1)
    plt.xlabel('iterations')
    plt.ylabel('theta_22_1')
    plt.title('theta_22_1 vs iterations')
    plt.show()
    
    #plotting theta_10_2 vs iterations
    plt.plot(y,theta_10_2)
    plt.xlabel('iterations')
    plt.ylabel('theta_10_2')
    plt.title('theta_10_2 vs iterations')
    plt.show()
    
    #plotting theta_11_2 vs iterations
    plt.plot(y,theta_11_2)
    plt.xlabel('iterations')
    plt.ylabel('theta_11_2')
    plt.title('theta_11_2 vs iterations')
    plt.show()
   
    #plotting theta_12_2 vs iterations
    plt.plot(y,theta_12_2)
    plt.xlabel('iterations')
    plt.ylabel('theta_12_2')
    plt.title('theta_12_2 vs iterations')
    plt.show()

    #plotting theta_20_2 vs iterations
    plt.plot(y,theta_20_2)
    plt.xlabel('iterations')
    plt.ylabel('theta_20_2')
    plt.title('theta_20_2 vs iterations')
    plt.show()
    
    #plotting theta_21_2 vs iterations
    plt.plot(y,theta_21_2)
    plt.xlabel('iterations')
    plt.ylabel('theta_21_2')
    plt.title('theta_21_2 vs iterations')
    plt.show()
    
    #plotting theta_22_2 vs iterations
    plt.plot(y,theta_22_2)
    plt.xlabel('iterations')
    plt.ylabel('theta_22_2')
    plt.title('theta_22_2 vs iterations')
    plt.show()


    #plot of theta_1 values vs iterations
    plt.plot(y,theta_10_1)
    plt.plot(y,theta_11_1)
    plt.plot(y,theta_12_1)
    plt.plot(y,theta_20_1)
    plt.plot(y,theta_21_1)
    plt.plot(y,theta_22_1)
    plt.xlabel('iterations')
    plt.ylabel('theta_1 values')
    plt.show()
    
    
    #plot of theta_2 values vs iterations
    plt.plot(y,theta_10_2)
    plt.plot(y,theta_11_2)
    plt.plot(y,theta_12_2)
    plt.plot(y,theta_20_2)
    plt.plot(y,theta_21_2)
    plt.plot(y,theta_22_2)
    plt.xlabel('iterations')
    plt.ylabel('theta_2 values')
    plt.show()