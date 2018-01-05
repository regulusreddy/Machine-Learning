# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:27:43 2017

@author: lalit
"""

import numpy as np
from numpy import linalg as lg
import pandas as pd
import pprint as pp
from matplotlib import pyplot as plt

#reading the dataset into pandas dataframe
data= pd.read_csv("dataset_1.csv")
data

#Question 1.1
#Plotting V2 vs V1 
plt.scatter(data['V1'], data['V2'], marker = 'o', color = 'blue', alpha = 0.9)
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Plotting V2 vs V1')
plt.show()

#Question 1.2
#Applying own PCA function to this dataset

#Ignoring class labels and taking only V1 and V2 for PCA
data_pca = data.drop('label',axis=1)
data_pca

#calculating column mean i.e. calculating mean for each column
mean_data_pca = data_pca.mean(axis=0)
mean_data_pca

#creating mean data frame from numpy array
mean_data_pca = np.array([['','V1','V2'],
                ['0',mean_data_pca[0],mean_data_pca[1]]])
                
mean_data_pca_1 = pd.DataFrame(data=mean_data_pca[1:,1:],
                  index=mean_data_pca[1:,0],
                  columns=mean_data_pca[0,1:])
mean_data_pca_1

#changing the dimension of mean_data_pca_1 to match dimension of data_pca to perform mean centering
mean_data_pca_replicate = pd.DataFrame(pd.np.tile(mean_data_pca_1, (60, 1)),columns=mean_data_pca[0,1:])     
mean_data_pca_replicate = mean_data_pca_replicate.astype(float)       
mean_data_pca_replicate


#mean centering the data
mean_center_data_pca = data_pca.subtract(mean_data_pca_replicate)
mean_center_data_pca


#computing covariance matrix
covariance_matrix=np.cov(mean_center_data_pca,rowvar=False)
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
scores= np.dot(mean_center_data_pca,eigenvectors)
scores


#Question 1.7
#computing explained variance
#Variance attributed to each eigen value
total_pca = sum(eigenvalues)
var_exp_pca = [(i / total_pca)*100 for i in sorted(eigenvalues, reverse=True)]
print('Variance explained by each eigen value', var_exp_pca)
cum_var_exp_pca = np.cumsum(var_exp_pca)
print('Cumulative Variance Explained:',cum_var_exp_pca)


#dictionary for pca results
pca_result = {}
pca_result['data']=data_pca
pca_result['mean centered data']=mean_center_data_pca
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



#Question 1.2
#projecting data to PC1 axis
projection_pc1_axis = data_pca.dot(eigenvectors[0])
projection_pc1_axis

plt.scatter(projection_pc1_axis,np.zeros(len(projection_pc1_axis)),linestyle='None', color="blue", alpha=0.5,marker=(5,1), label='1')
plt.title('Projecting raw PCA data to PC1 axis')
plt.show()



#Question 1.3
#Adding the PC1 axis to question 1.1
plt.scatter(data['V1'][0:30], data['V2'][0:30], marker = 'o', color = 'red', alpha = 0.5, label= 'Label 1')
plt.scatter(data['V1'][30:60], data['V2'][30:60], marker = 'o', color = 'blue', alpha = 0.5, label= 'Label 2')
plt.plot([0, 50*pca_result['loadings'][0,0]], [0,50*pca_result['loadings'][1,0]],
            color='black', linewidth=2)
plt.plot([0, -50*pca_result['loadings'][0,0]], [0,-50*pca_result['loadings'][1,0]],
            color='black', linewidth=1, label='PC axis')
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Plotting V2 vs V1 and the PC1 axis')
plt.legend()
plt.xlim(0,35)
plt.ylim(0,40)
plt.show()

#------------------------------------------------------------------------------
#Question 1.4
#LDA algorithm implementation
#reading data into pandas dataframe 
data_lda= pd.read_csv("dataset_1.csv")
data_lda

#removing rownames and  class label
data_lda = data_lda.drop('label',axis=1)
data_lda.shape

#splitting data class-wise into  class 1 and class 2
data_class_I, data_class_II = data_lda.iloc[0:30,:], data_lda.iloc[30:60,:] 
data_class_I
data_class_II

#mean of data column-wise for class I  
mean_mu_I = data_class_I.mean(axis=0)
mean_mu_I

#mean of data column-wise for class II SCLC
mean_mu_II = data_class_II.mean(axis=0)
mean_mu_II

#column-wise mean of overall data
mean_overall = data_lda.mean(axis=0)
mean_overall

#reshaping mean vectors to calculate within class scatter matrix
mean_mu_I = mean_mu_I.values.reshape(1,2)
mean_mu_I
mean_mu_I = np.repeat(mean_mu_I,30,axis =0)
mean_mu_I.shape

mean_mu_II = mean_mu_II.values.reshape(1,2)
mean_mu_II
mean_mu_II = np.repeat(mean_mu_II,30,axis =0)
mean_mu_II.shape

#computing within class scatter matrix
scatter_within_class = np.zeros((2,2))

#computing scatter_within s1 for class 1 
SW_I = np.zeros((2,2))
SW_I = np.matmul((np.transpose(data_class_I - mean_mu_I)),(data_class_I - mean_mu_I))
SW_I
SW_I.shape

#computing scatter_within s2 for class 2
SW_II = np.zeros((2,2))
SW_II = np.matmul((np.transpose(data_class_II - mean_mu_II)),(data_class_II - mean_mu_II))
SW_II
SW_II.shape

#computing within scatter matrix sw1+sw2
scatter_within_class = np.add(SW_I,SW_II)
scatter_within_class


#manipulating mean vectors to calculate between class scatter matrix
#mean of data column-wise for class I  
mean_mu_I = data_class_I.mean(axis=0)
mean_mu_I = np.asarray(mean_mu_I).reshape(2,1)
mean_mu_I.shape

#mean of data column-wise for class II
mean_mu_II = data_class_II.mean(axis=0)
mean_mu_II = np.asarray(mean_mu_II).reshape(2,1)
mean_mu_II.shape

#mean of overall data
mean_overall = data_lda.mean(axis=0)
mean_overall = np.asarray(mean_overall).reshape(2,1)
mean_overall.shape

#computing between class scatter matrix SB_1 and SB_2
SB_I =np.multiply(len(data_class_I),np.outer((mean_mu_I - mean_overall),(mean_mu_I - mean_overall)))
SB_I
SB_II =np.multiply(len(data_class_II),np.outer((mean_mu_II - mean_overall),(mean_mu_II - mean_overall)))
SB_II

#computing scatter_between sb_1 + sb_2
scatter_between = np.add(SB_I,SB_II)
scatter_between

#computing the eigen vector, eigen value pairs for np.dot(inv(scatter_within_class), scatter_between) 
eig_values, eig_vectors = np.linalg.eig(np.dot(lg.inv(scatter_within_class),scatter_between))
eig_pairs = [(np.abs(eig_values[eig]), eig_vectors[:,eig].reshape(2,1)) for eig in range(len(eig_values))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
print (eig_pairs)


#Question 1.8
#Variance of the projection of W axis
#computing explained variance for eigen values obtained from LDA
total_lda = sum(eig_values)
var_exp_lda = [(i / total_lda)*100 for i in sorted(eig_values, reverse=True)]
print('Variance explained by each eigen value', var_exp_lda)
cum_var_exp_lda = np.cumsum(var_exp_lda)
print('Cumulative Variance Explained:',cum_var_exp_lda)



#Question 1.4
#now computing the  matrix W
eig_pairs = [(np.abs(eig_values[eig]), eig_vectors[:,eig]) for eig in range(len(eig_values))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
W_matrix = eig_pairs[0][1].reshape(2,1)
print('Matrix W:\n', W_matrix.real)       


#Question 1.5 
#Projecting the raw onto W
projection_lda = data_lda.dot(W_matrix)
projection_lda.values.real
projection_lda.shape
#Plotting the projection on to W axis
plt.scatter(projection_lda.iloc[0:30,:],np.zeros(len(data_class_I)),linestyle='None', color="blue", alpha=0.5,marker=(5,1), label='1')
plt.scatter(projection_lda.iloc[30:60,:],np.zeros(len(data_class_II)),linestyle='None',color="red", alpha=0.5,marker=(5,1), label='0')
plt.title('Projection of raw data to W axis of LDA')
plt.legend()
plt.show()



#Question 1.6
#plotting raw data points along with PC1 axis and W axis 
plt.scatter(data['V1'][0:30], data['V2'][0:30], marker = 'o', color = 'red', alpha = 0.5, label= 'Label 1')
plt.scatter(data['V1'][30:60], data['V2'][30:60], marker = 'o', color = 'blue', alpha = 0.5, label= 'Label 2')
plt.plot([0, 50*pca_result['loadings'][0,0]], [0,50*pca_result['loadings'][1,0]],
            color='black', linewidth=2)
plt.plot([0, -50*pca_result['loadings'][0,0]], [0,-50*pca_result['loadings'][1,0]],
            color='black', linewidth=2, label='PC axis')
plt.plot(150* W_matrix,label ='W axis')
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Plotting V2 vs V1, PC1 axis and W axis')
plt.legend() 
plt.xlim(0,40)
plt.ylim(0,40)
plt.show()


