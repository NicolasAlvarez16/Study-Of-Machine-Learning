import numpy as np
import matplotlib.pyplot as ptl
import pandas as pd

# IMPORTING THE DATASET
dataset = pd.read_csv("Data.csv") 
x = dataset.iloc[:, :-1].values # x = dataset.iloc[rows (all row), columns (all column excep last one)].values(taking all values)
y = dataset.iloc[:, -1] # y = dataset.iloc[row (all rows), columns(only the last column)].values(taking all values)

print(x)
print(y)

# TAKING CARE OF THE MISSING DATA
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") # inputer = SimpleImputer(missing values, those missing values will be replaced by the mean of the rest of the value)
imputer.fit(x[:, 1:3]) # The fit() method will search for missing values in the columns of Age and salary (This will connect our imputer to our matrix of features)
x[:, 1:3] = imputer.transform(x[:, 1:3]) # Second and third column missing data will be replaced by the average

print(x)

# ENCODING CATEGORICAL DATA
# 1. ENCODING THE INDEPENDENT VARIABLE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough") # ct = ColumnTransform(transformers = [(encoding the countries into three new columns)], do not delete the rest of the columns)
x = np.array(ct.fit_transform(x))

print(x)

# 2. ENCODING THE DEPENDET VARIABLE
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

# SPLITTING THE DATASET INTO THE TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1) # = train_test_split(matrix variable, dependent variable, split sizem, no randomness(for teaching purposes))
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])  # x_train[row, columns]
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print(x_train)
print(x_test)

