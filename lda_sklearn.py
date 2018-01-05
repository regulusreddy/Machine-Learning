# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:10:18 2017

@author: lalit
"""
import numpy as np
from matplotlib import pyplot as plt


#LDA usingsklearn
# apply sklearn LDA to cell line data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda_sk_module
from numpy import genfromtxt

#reading dataset into numpy array
data_sklearn = genfromtxt('SCLC_study_output_filtered_2.csv', delimiter = ',') 

#deleting NAN in numpy array
data_sklearn_X = np.delete(data_sklearn,(0),axis=0)
data_sklearn_X = np.delete(data_sklearn_X,(0),axis=1)
data_sklearn_X
data_sklearn_X.shape

class_NSCLC = np.zeros(data_sklearn.shape[0]//2)
class_NSCLC = class_NSCLC.reshape((20,1))

class_SCLC = np.ones(data_sklearn.shape[0]//2)
class_SCLC= class_SCLC.reshape((20,1))


data_sklearn_class_Y= np.concatenate((class_NSCLC,class_SCLC),axis=0)
data_sklearn_class_Y = data_sklearn_class_Y.astype(int).reshape(1,40).ravel()
data_sklearn_class_Y.shape




lda= lda_sk_module(n_components=1)
sklearn_LDA_projection = lda.fit_transform(data_sklearn_X,data_sklearn_class_Y )

II_class_0 = np.where(data_sklearn_class_Y==0)
II_class_1 = np.where(data_sklearn_class_Y==1)

II_class_0 = II_class_0[0]
II_class_1 = II_class_1[0]

# plot the projections
plt.title('Results from applying sklearn LDA to cell line data')
plt.xlabel('LDA 1')
plt.ylabel('')
plt.scatter(sklearn_LDA_projection[II_class_0], np.zeros(len(II_class_0)),linestyle='None', color="blue", alpha=0.5,marker=(5,1), label='NSCLC')
plt.scatter(sklearn_LDA_projection[II_class_1], np.zeros(len(II_class_1)),linestyle='None',color="red", alpha=0.5,marker=(5,1), label='SCLC')
plt.legend()
plt.show()
