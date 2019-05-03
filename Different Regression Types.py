# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Research Data - Sheet1.csv')
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Regression Model to the dataset
# Type 1
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)

# Type 2
from sklearn.svm import SVR
regressor3 = SVR(kernel='poly')
regressor3.fit(X_train, y_train)

# Type 3
from sklearn.linear_model import LinearRegression
regressor2 = LinearRegression()
regressor2.fit(X, y)

