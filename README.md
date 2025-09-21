# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect input features (size, bedrooms, etc.) and target outputs (price, occupants).
2.Train MultiOutputRegressor with SGDRegressor on the dataset.
3.Give new input data (e.g., house details) to the trained model.
4.Predict and display the house price and number of occupants. 

## Program:
```
from sklearn.linear_model import SGDRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split

import numpy as np

# Dummy dataset (features: [size, bedrooms], targets: [price, occupants])
X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 3],
    [2500, 4],
    [3000, 5]
])

y = np.array([
    [150000, 3],
    [220000, 4],
    [280000, 4],
    [350000, 5],
    [450000, 6]
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3))
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Predictions:", y_pred)

# Example new house
new_house = np.array([[1800, 3]])
print("New house prediction:", model.predict(new_house))

```

## Output:
<img width="1043" height="147" alt="image" src="https://github.com/user-attachments/assets/ff835844-ab77-4a4e-8d42-8c2380d50328" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
