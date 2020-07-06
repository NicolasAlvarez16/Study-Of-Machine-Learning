import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# TRAINING THE RANDOM FOREST REGRESSION MODEL ON THE WHOLE DATASET
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

# PREDICTING A NEW RESULT
prediction = regressor.predict([[6.5]])
print(prediction)

# VISUALISING THE RANDOM FOREST REGRESSION RESULTS (HIGHER RESOLUTION)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color = "blue")
plt.title("Truth or Bluff (Random Forest Regession)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()