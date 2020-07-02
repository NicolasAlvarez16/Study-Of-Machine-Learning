import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)
y = y.reshape(len(y), 1)
print(y)

# FEATURES SCALING
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x) # To compute the mean and the standard deviation of the Position Level
y = sc_y.fit_transform(y) # To compute the mean and the standard deviation of the Salary

print(x)
print(y)

# TRAINING THE SVR MODEL ON THE WHOLE DATASET
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x, y)

# PREDICTING A NEW RESULT
prediction = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]]))) # Scaling our prediction in respec to the data and then using inverse scaling to put everythinh back to normal data
print(prediction)

# VISUALISING THE SVR RESULTS
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = "blue")
plt.title("Truth or Bluff (Support Vector Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# VISUALISING THE SVR RESULTS (FOR HIHG RESOLUTION AND SMOTHER CURVE)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = "red")
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))))
plt.title("Truth or Bluff (Support Vector Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()