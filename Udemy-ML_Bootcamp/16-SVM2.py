import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load Data
df = sns.load_dataset('iris')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
# sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'

# Analyze Data
# sns.pairplot(data=df, hue='species')
# plt.show()

# setosa = df[df['species'] == 'setosa']
# sns.kdeplot(data=setosa, x=setosa['sepal_width'], y=setosa['sepal_length'], cmap="plasma", fill=True, thresh=0.05)
# plt.show()

# Preprocess
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Train
svc_model = SVC()
svc_model.fit(X_train, y_train)

# Predict
predictions = svc_model.predict(X_test)

# Evaluate
print("Confustion Matrix:\n", confusion_matrix(y_test, predictions))
print()
print('Classification Report =\n', classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')

# Gridsearch
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

print("SVM + Gridsearch")
print("Best Params:\n", grid.best_params_)
print("Best Estimator:\n", grid.best_estimator_)

grid_predictions = grid.predict(X_test)
print("Confustion Matrix:\n", confusion_matrix(y_test, grid_predictions))
print()
print('Classification Report =\n', classification_report(y_test, grid_predictions))
print('------------------------------------------------------------------------\n')