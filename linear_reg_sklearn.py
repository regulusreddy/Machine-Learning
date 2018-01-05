# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:15:13 2017

@author: lalit

"""

from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics


#loading the dataset diabetes
diabetes = datasets.load_diabetes()

#subsetting dataset to get x
diabetes_x = diabetes.data[:,2]
diabetes_x


#subsetting dataset to get y
diabetes_y = diabetes.target
diabetes_y

#converting numpy array to pd dataframes
diabetes_data = pd.DataFrame({'x': diabetes_x, 'y': diabetes_y})
print (diabetes_data)

#splitting the data by randomly selecting 20 points from the data
diabetes_random_20 = diabetes_data.sample(n=20)

#checking the points
diabetes_random_20

#getting the rest of the 422 points from the data
diabetes_rest_422 = diabetes_data.loc[~diabetes_data.index.isin(diabetes_random_20.index)]
diabetes_rest_422


#setting test data of x from the random 20 points
diabetes_test_x = diabetes_random_20['x']
diabetes_test_x

#settings test data of y from the random 20 points
diabetes_test_y = diabetes_random_20['y']
diabetes_test_y

#settings train data of x from the rest 422 points
diabetes_train_x = diabetes_rest_422['x']
diabetes_train_x


#settings train data of y from the rest 422 points
diabetes_train_y = diabetes_rest_422['y']
diabetes_train_y

#reshaping the data to get 2d array
diabetes_train_x = diabetes_train_x.reshape(-1,1)
diabetes_test_x = diabetes_test_x.reshape(-1,1)

#creating the regression object
lm_reg = linear_model.LinearRegression()

#training the model using training data
lm_reg.fit(diabetes_train_x, diabetes_train_y)


#predicting Y from the model linear regression model lm_reg
diabetes_predict_y = lm_reg.predict(diabetes_test_x)
diabetes_predict_y
#plotting testing x vs testing y and testing x vs predicted y in the same plot
plt.scatter(diabetes_test_x, diabetes_test_y,  color='blue', label = "testing x vs testing y")
plt.plot(diabetes_test_x, diabetes_predict_y, color='black', linewidth=2, label= "testing x vs predicted y")
plt.legend()
plt.show()

#explained variance by the model: rsquared
print"variance explained or R-squared :"
lm_reg.score(diabetes_train_x, diabetes_train_y)

#printing the mean squared error of the model
print("rmse:")
print(metrics.mean_squared_error(diabetes_test_y, diabetes_predict_y))