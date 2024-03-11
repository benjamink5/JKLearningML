
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from six import StringIO
import pydot

# Data Load
df = pd.read_csv('./Data/kyphosis.csv')
print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
# ['Kyphosis', 'Age', 'Number', 'Start']
print('------------------------------------------------------------------------\n')

# Data Analysis
# sns.pairplot(df, hue='Kyphosis', palette='Set1')
# plt.show()

# Preprocess
X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Prediction
print("Decision Tree")
predictions = dtree.predict(X_test)
print('Confusion Matrix =\n', confusion_matrix(y_test, predictions))
print()
print('Classification Report =\n', classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')


# Visualization
features = list(df.columns[1:])
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
#Image(graph[0].create_png()) # TODO: It will be used by the jupyer notebook? Need to import Image package from IPython.display module
graph[0].write_png("kyphosis.png")

print("Random Forests")
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print('Confusion Matrix =\n', confusion_matrix(y_test, rfc_pred))
print()
print('Classification Report =\n', classification_report(y_test, rfc_pred))
print('------------------------------------------------------------------------\n')


