import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Data Load
df = pd.read_csv('./Data/KNN_Project_Data.csv')
print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
# ['XVPM', 'GWYH', 'TRAT', 'TLLZ', 'IGGA', 'HYKR', 'EDFS', 'GUUB', 'MGJM', 'JHZC', 'TARGET CLASS']
print('------------------------------------------------------------------------\n')

# Data Analysis
# sns.pairplot(data=df, hue='TARGET CLASS')
# plt.show()

# Preprocess
X = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']
scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)
# df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])
# print(df_scaled.head())

X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, train_size=0.3, random_state=101)

# Train
print('When K=1')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)

# Evaluation
print('Confustion Matrix=\n', confusion_matrix(y_test, predictions))
print()
print('Classification Report=\n', classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')

# K-value section
# error_rates = []
# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     pred_i = knn.predict(X_test)
#     error_rates.append(np.mean(y_test != pred_i))

# fig = plt.figure(figsize=(10,6))
# plt.plot(range(1, 40), error_rates, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.grid()
# plt.show()

print('When K=30')
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)

# Evaluation
print('Confustion Matrix=\n', confusion_matrix(y_test, predictions))
print()
print('Classification Report=\n', classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')
