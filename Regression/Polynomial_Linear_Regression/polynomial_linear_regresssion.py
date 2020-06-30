import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORT THE DATASET
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# TRAINING THE LINEAR REGRESSION MODEL ON THE WHOLE DATASET
from sklearn.linear_model import LinearRegression
line_reg = LinearRegression() # Build the linear regression model
line_reg.fit(x, y) # Train the linear regression model

# TRINING THE POLYNOMIAL REGRESSION MODEL ON THE WHOLE DATASET
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # degree = n in the formula
x_poly = poly_reg.fit_transform(x) # Transform this matrixc of single features into a new matrix of features (new matrix of featutres for the polynomial model)
line_reg_2 = LinearRegression() # New linear regression model to be trained on the new matrix of features
line_reg_2.fit(x_poly, y)

# VISUALISING THE LINEAR REGRESSION RESULTS
plt.scatter(x, y, color = "red")
plt.plot(x, line_reg.predict(x), color = "blue")
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# VISUALISING THE POLYNOMIAL REGRESSION RESULTS
plt.scatter(x, y, color = "red")
plt.plot(x, line_reg_2.predict(x_poly), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# VISUALISING THE POLYNOMIAL REGRESSION RESULTS (FOR HIGHER RESOLUTIONS AND SMOTHER CURVE)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = "red")
plt.plot(x_grid, line_reg_2.predict(poly_reg.fit_transform(x_grid)), color = "blue")
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# PREDICTING A NEW RESULT WITH LINEAR REGRESSION
print(line_reg.predict([[6.5]]))

# PREDICTING A NEW RESULT WITH POLYNOMIAL REGRESSION
print(line_reg_2.predict(poly_reg.fit_transform([[6.5]])))
