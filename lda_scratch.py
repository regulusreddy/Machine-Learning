# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 12:06:16 2017

@author: lalit
"""
#LDA on dataset SCLC study output filtered 2.csv

import numpy as np
import pandas as pd
from numpy import linalg as lg
from matplotlib import pyplot as plt



#LDA algorithm implementation
#reading data into pandas dataframe 
data= pd.read_csv('SCLC_study_output_filtered_2.csv')    
data

#removing rownames and  class label sclc and nsclc
data= data.loc[:, ~data.columns.str.contains('^Unnamed')]
data.shape

#splitting data class-wise into  class 1 and class 2
data_class_I, data_class_II = data.iloc[0:20,:], data.iloc[20:41,:] 
data_class_I
data_class_II

#mean of data column-wise for class I  NSCLC
mean_mu_I = data_class_I.mean(axis=0)
mean_mu_I

#mean of data column-wise for class II SCLC
mean_mu_II = data_class_II.mean(axis=0)
mean_mu_II

#column-wise mean of overall data
mean_overall = data.mean(axis=0)
mean_overall

#reshaping mean vectors to calculate within class scatter matrix
mean_mu_I = mean_mu_I.values.reshape(1,19)
mean_mu_I
mean_mu_I = np.repeat(mean_mu_I,20,axis =0)
mean_mu_I.shape

mean_mu_II = mean_mu_II.values.reshape(1,19)
mean_mu_II
mean_mu_II = np.repeat(mean_mu_II,20,axis =0)
mean_mu_II.shape

#computing within class scatter matrix
scatter_within_class = np.zeros((19,19))

#computing scatter_within s1 for class 1 
SW_I = np.zeros((19,19))
SW_I = np.matmul((np.transpose(data_class_I - mean_mu_I)),(data_class_I - mean_mu_I))
SW_I
SW_I.shape

#computing scatter_within s2 for class 2
SW_II = np.zeros((19,19))
SW_II = np.matmul((np.transpose(data_class_II - mean_mu_II)),(data_class_II - mean_mu_II))
SW_II
SW_II.shape

#computing within scatter matrix sw1+sw2
scatter_within_class = np.add(SW_I,SW_II)
scatter_within_class


#manipulating mean vectors to calculate between class scatter matrix
#mean of data column-wise for class I  NSCLC
mean_mu_I = data_class_I.mean(axis=0)
mean_mu_I = np.asarray(mean_mu_I).reshape(19,1)
mean_mu_I.shape

#mean of data column-wise for class II SCLC
mean_mu_II = data_class_II.mean(axis=0)
mean_mu_II = np.asarray(mean_mu_II).reshape(19,1)
mean_mu_II.shape

#mean of overall data
mean_overall = data.mean(axis=0)
mean_overall = np.asarray(mean_overall).reshape(19,1)
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
for eig in range(len(eig_values)):
    eigvec_scatter = eig_vectors[:,eig].reshape(19,1)   
    print('\n Eigen vector:\n', eigvec_scatter.real)
    print('\n Eigen value:\n', eig_values[eig].real)


#eigen values in decreasing order
for eig in range(len(eig_values)):
    eig_pairs = [(np.abs(eig_values[eig]), eig_vectors[:,eig])]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    for i in eig_pairs:
        print ('eigenvalue {}:\t{}\n '.format(eig+1, i[0]))

#creating a list of eigen pairs and sorting it in decreasing order and then computing variance explained by each of the eigen value.
for eig in range(len(eig_values)):
    eig_pairs = [(np.abs(eig_values[eig]), eig_vectors[:,eig])]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    eigv_sum = sum(eig_values)
    for a,b in enumerate(eig_pairs):
        print('Variance explained:')
        print ('eigenvalue {0:}: {1:.15%} \n'.format(eig+1, (b[0]/eigv_sum).real))
#It can be noted that the first eigen pair is the most informative at 99.999999999998167% explained variance 
#and we wont loose much information if we ignore other eigen pairs

#now computing the Eigen vector matrix W, choosing the eigen vectors with largest eigen values
eig_pairs = [(np.abs(eig_values[eig]), eig_vectors[:,eig]) for eig in range(len(eig_values))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
W_matrix = eig_pairs[0][1].reshape(19,1)
print('Matrix W:\n', W_matrix.real)       


#Projection, transforming data into new subspace where projection= data X W_matrix,
#here our data is 40X19 and W_matrix is 19X1 our projected data dimension is 40X1
projection_lda = data.dot(W_matrix)
projection_lda.values.real
projection_lda.shape

#plotting the new projection
#plotting y vs x; y theoretical vs x and pca axis
plt.scatter(projection_lda.iloc[0:20,:],np.zeros(len(data_class_I)),linestyle='None', color="blue", alpha=0.5,marker=(5,1), label='NSCLC')
plt.scatter(projection_lda.iloc[20:40,:],np.zeros(len(data_class_II)),linestyle='None',color="red", alpha=0.5,marker=(5,1), label='SCLC')
plt.title('Results from applying own implementation of LDA to cell line data')
plt.legend()
plt.show()



