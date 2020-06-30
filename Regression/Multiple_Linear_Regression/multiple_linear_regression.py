import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT DATASET
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)

# ENCODING CATEGORICAL DATA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

print(x)

# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# TRAINING THE MULTIPLE LINEAR REGRESSION MODEL ON THE TRRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Building the multiple linear regression model
regressor.fit(x_train, y_train) # Train the multiple linear regession model previously created

# PREDICTING THE TEST RESULTS
y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2) #np.setprintoptions(precision = number of decimals)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)) # Concatanate our 2 vectors vertical