import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING DATASET
dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# SPLITTING THE DATASET INTO THE TRAINING SET AND THE TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON THE TRAINING SET
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) # To connect our model to our data

# PREDICTING THE TEST SET RESULTS
y_pred = regressor.predict(x_test)

# VISUALASING THE TRAINING SET RESULTS
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "green")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# VISUALASING THE TEST SET RESULTS
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "green")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# MAKING A SINGLE PREDICTION (FOR EXAMPLE THE SALARY OF AN EPLOYEE WITH 12 YEARS OF EXPERIENCE)
print(regressor.predict([[12]]))

# GETTING THE FINAL LINEAR REGRESSION EQUATION WITH THE VALUES OF THE COEFFICIENTS
print(regressor.coef_)
print(regressor.intercept_)