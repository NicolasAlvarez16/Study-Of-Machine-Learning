import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(x)

# ENCODING CATEGORICAL DATA
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), [3])], remainder = "passthrough")
x = np.array(ct.fit_transform(x))

#print(x)

# SPLITTING THE MODEL INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# TRAINING THE DECISION TREE REGRESSION MODEL ON THE TRAINING SET
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# PREDICTING THE TEST RESULTS
y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# PREDICTING A NEW RESULT
prediction = regressor.predict([[1, 0, 0, 160000, 130000, 300000]])
print(prediction)

# EVALUATINF THE MODEL PERFORMANCE
from sklearn.metrics import r2_score
performance = r2_score(y_test, y_pred)
print(performance)
