import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================
# 1. Load Dataset
# =========================
iris = load_iris()

# Select only Petal Length and Petal Width
X = iris.data[:, [2, 3]]

# Select only Setosa (0) and Versicolor (1)
mask = iris.target != 2
X = X[mask]
y = iris.target[mask]

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)


# =========================
# 2. Visualize Dataset
# =========================
plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0][:, 0],
    X[y == 0][:, 1],
    label="Setosa"
)

plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    label="Versicolor"
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# 3. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# 4. Create Perceptron
# =========================
model = Perceptron(
    max_iter=1000,
    random_state=42
)


# =========================
# 5. Train Model
# =========================
model.fit(X_train, y_train)


# =========================
# 6. Predict
# =========================
y_pred = model.predict(X_test)


# =========================
# 7. Accuracy
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")


# =========================
# 8. Learned Weights & Bias
# =========================
print("\nWeights:")
print(model.coef_)

print("\nBias:")
print(model.intercept_)


# =========================
# 9. Decision Boundary
# =========================
w = model.coef_[0]
b = model.intercept_[0]

x_values = np.linspace(
    X[:, 0].min(),
    X[:, 0].max(),
    100
)

y_values = -(w[0] * x_values + b) / w[1]

plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0][:, 0],
    X[y == 0][:, 1],
    label="Setosa"
)

plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    label="Versicolor"
)

plt.plot(
    x_values,
    y_values,
    label="Decision Boundary"
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.grid(True)

plt.show()