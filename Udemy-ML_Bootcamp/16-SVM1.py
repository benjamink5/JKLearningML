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
cancer = load_breast_cancer()
#print(cancer.keys())
# print(cancer['DESCR'])
# print(cancer['feature_names'])
'''
'mean radius' 'mean texture' 'mean perimeter' 'mean area'
'mean smoothness' 'mean compactness' 'mean concavity'
'mean concave points' 'mean symmetry' 'mean fractal dimension'
'radius error' 'texture error' 'perimeter error' 'area error'
'smoothness error' 'compactness error' 'concavity error'
'concave points error' 'symmetry error' 'fractal dimension error'
'worst radius' 'worst texture' 'worst perimeter' 'worst area'
'worst smoothness' 'worst compactness' 'worst concavity'
'worst concave points' 'worst symmetry' 'worst fractal dimension'
'''

# Preprocess
df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df_feat.head())
# print(df_feat.info())
df_target = pd.DataFrame(cancer['target'], columns=["Cancer"])

X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)

# Train
model = SVC()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("SVM")
print("Confustion Matrix:\n", confusion_matrix(y_test, predictions))
print()
print('Classification Report =\n', classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')

# Gridsearch
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

grid.fit(X_train,y_train)
print("SVM + Gridsearch")
print("Best Params:\n", grid.best_params_)
print("Best Estimator:\n", grid.best_estimator_)

grid_predictions = grid.predict(X_test)
print("Confustion Matrix:\n", confusion_matrix(y_test, grid_predictions))
print()
print('Classification Report =\n', classification_report(y_test, grid_predictions))
print('------------------------------------------------------------------------\n')
