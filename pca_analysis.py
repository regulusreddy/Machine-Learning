# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:46:52 2017

@author: lalit

"""

#PCA Analysis

import numpy as np
from numpy import genfromtxt
from numpy import linalg as lg
from matplotlib import pyplot as plt
import pprint as pp


#reading data from csv
data = genfromtxt("C:/Users/lalit/Desktop/linear_regression_test_data.csv", delimiter=",")
data


#deleting NAN in numpy array
data = np.delete(data,(0),axis=0)
data = np.delete(data,(0),axis=1)
data


#convert numpy array to matrix
data= np.asmatrix(data)
data

#x values
x=data[:,0]
x

#y values
y=data[:,1]
y

#y theoretical values
y_theoretical=data[:,2]
y_theoretical


#deleting y theoricatal values
data_one = np.delete(data,(2),axis=1)
data_one


#computing means for each
col_mean = data_one.mean(axis=0)
col_mean


#repeating column mean to match data_one dimension
col_mean_data = np.tile(col_mean, reps=(data_one.shape[0],1))
col_mean_data


#computing mean center
data_one_mean_center = data_one - col_mean_data
data_one_mean_center




#computing covariance matrix
covariance_matrix=np.cov(data_one_mean_center,rowvar=False)
covariance_matrix



#sorting eigen values and corresponding eigen vectors in descending order
n_components=2
eigenvalues, eigenvectors = lg.eig(covariance_matrix)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx][:n_components]
eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]
#return eigenvalues,eigenvectors
print'Eigen Values \n %s' %eigenvalues
print 'Eigen Vectors \n %s' %eigenvectors




#computing pca scores
scores= np.dot(data_one_mean_center,eigenvectors)
scores


#computing explained variance
total = sum(eigenvalues)
var_exp = [(i / total)*100 for i in sorted(eigenvalues, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
cum_var_exp

#plotting prinicpal components
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(7, 7))
    plt.bar(range(2), var_exp, alpha=0.5, align='center',label='individual explained variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components:')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


#dictionary for pca results
pca_result = {}
pca_result['data']=data_one
pca_result['mean centered data']=data_one_mean_center
pca_result['PC variance']= eigenvalues
pca_result['loadings']= eigenvectors
pca_result['scores']= scores


#pretty printing pca results
print "data:"
pp.pprint (pca_result['data'])
print "Mean Centerd Data:"
pp.pprint (pca_result['mean centered data'])
print "PC Variance:"
pp.pprint (pca_result['PC variance'])
print "Loadings:"
pp.pprint (pca_result['loadings'])
print "PC Scores:"
pp.pprint (pca_result['scores'])



#transforming data to new axis
data_one_transformed = data_one_mean_center.dot(eigenvectors)
data_one_transformed.shape


#plotting y vs x; y theoretical vs x and pca axis
plt.scatter(np.asarray(x), np.asarray(y), color="red", alpha=0.5,marker=(5,1), label='y vs x')
plt.scatter(np.asarray(x), np.asarray(y_theoretical), color="blue", alpha=0.5, marker="+", label='y theoretical vs x')
plt.plot([0, 10*pca_result['loadings'][0,0]], [0,10*pca_result['loadings'][1,0]],
            color='yellow', linewidth=2)
plt.plot([0, -10*pca_result['loadings'][0,0]], [0,-10*pca_result['loadings'][1,0]],
            color='yellow', linewidth=2, label='PC axis')
plt.title('y vs x; y theoretical vs x; pc1 axis')
plt.xlim(-2.5, 2.5), plt.ylim(-5, 5)
plt.legend()
plt.show()
