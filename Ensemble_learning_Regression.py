# Data Preprocessing
# Importing the libraries
import pandas as pd
import numpy as np

#Read in the excel data
df = pd.read_excel (r'Training Set.xlsx', sheet_name='Sheet 1')
print (df)
#statistical information on the Data frame
df.describe() 
#view the 1st few rows of the data frame
df.head()
#find the data type of variables
df.dtypes.value_counts() # No categorical features in our data

#####Data preprocessing
#identify columns with missing values
#true if data is missing false if data is present in particular cell
missing_val = df.isnull()
#number of missing values in numerical and categorical dimensions
missing_val_sum = df.isnull().sum().sort_values(ascending=False)
print(missing_val_sum) #No missing in the data


#identify correlation between the predictors using Pearson correlation
corrdata = df.iloc[:, 1:-1]
correlation_ = corrdata.corr(method ='pearson')

#Dimensionality reduction using PCA
from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(corrdata)
x_pca = pd.DataFrame(x_pca)
x_pca.head()
explained_variance = pca.explained_variance_ratio_
explained_variance #variance explained by each of the PCA components

x_pca.columns = ['PC1', 'PC2', 'PC3','PC4','PC5','PC6','PC7','PC8','PC9','P10','P11']
x_pca.head()

#define the predictors and target variable
#Select the first 4 PCA components only since others are not significant enough
X = x_pca.iloc[:, :-7].values
#define target variable
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Libraries for comparing machine learning models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from matplotlib import pyplot
from sklearn import tree

###Ensemble model - Averaging
model1 = tree.DecisionTreeRegressor()
model2 = KNeighborsRegressor()
#model fitting
model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
#Prediction
pred1=model1.predict(X_test)
pred2=model2.predict(X_test)
#Model averaging
finalpred=(pred1+pred2)/2
print(finalpred)

#Model performance
# Root mean squared error
errors = mean_squared_error(y_test, finalpred, squared=False)
errors


#######################Test set
#Read in the excel data
df_ = pd.read_excel (r'Test Set.xlsx', sheet_name='Sheet 1')
print (df_)
df_.describe()

#define the predictors and target variable
X1 = df_.iloc[:, 1:-1].values

#reduce the test data using PCA
xpca = pca.fit_transform(X1)
xpca = pd.DataFrame(xpca)
xpca.head()
explainedvariance = pca.explained_variance_ratio_
explainedvariance

xpca.columns = ['PC1', 'PC2', 'PC3','PC4','PC5','PC6','PC7','PC8','PC9','P10','P11']
xpca.head()

#define the predictors and target variable
X_1 = xpca.iloc[:, :-7].values

#predict the electricity price for the test data using the trained model
pred_1=model1.predict(X_1)
pred_2=model2.predict(X_1)
final_pred=(pred_1+pred_1)/2
#Final predicted electricity price
print(final_pred)





