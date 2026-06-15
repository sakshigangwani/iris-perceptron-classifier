# Iris Perceptron Classifier

A machine learning project that uses a [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) to classify Iris flowers. It trains a linear binary classifier to separate the **Setosa** and **Versicolor** species using their petal measurements, then visualizes the data and the learned decision boundary.

## Overview

The script `iris_perceptron.py` walks through a complete, classic ML workflow:

1. **Load data** — uses the built-in Iris dataset from scikit-learn.
2. **Select features & classes** — keeps only *petal length* and *petal width* as features, and only the *Setosa* (0) and *Versicolor* (1) classes (a linearly separable, binary problem).
3. **Visualize** — plots the two classes in feature space.
4. **Split** — 80% train / 20% test split.
5. **Train** — fits a `Perceptron` model.
6. **Evaluate** — reports accuracy on the test set.
7. **Inspect** — prints the learned weights and bias.
8. **Visualize the boundary** — plots the perceptron's decision boundary over the data.

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn

## Installation

```bash
# (optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python iris_perceptron.py
```

Running the script will:

- Print the feature and label shapes, test-set accuracy, and the learned weights and bias to the console.
- Open two plot windows: the raw dataset, and the dataset with the perceptron's decision boundary.

> **Note:** The plots are shown with `plt.show()`. Close each plot window to let the script continue to the next step.

## How It Works

A perceptron learns a linear decision boundary of the form:

```
w[0] * x1 + w[1] * x2 + b = 0
```

where `x1` is petal length, `x2` is petal width, `w` are the learned weights (`model.coef_`), and `b` is the bias (`model.intercept_`). Because Setosa and Versicolor are linearly separable on these two features, the perceptron typically reaches 100% accuracy on the test set.