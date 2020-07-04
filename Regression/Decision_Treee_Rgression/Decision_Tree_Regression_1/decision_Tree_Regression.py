import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# TRAINING THE DECISION TREE REGRESSION MODEL ON THE WHOLE DATASET
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)

# PREDICT A NEW RESULT
prediction = regressor.predict([[6.5]])
print(prediction)

# VISUALISING THE DECISION TREE REGRESSION RESULTS (HIGH RESOLUTIONS)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regressor.predict(x_grid), color = "blue")
plt.title("Truth or Bluff (Decision Tree Regressor)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()