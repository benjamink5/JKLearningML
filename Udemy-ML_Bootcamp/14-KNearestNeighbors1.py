import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('./Data/Classified_Data.csv', index_col=0)
print(df.head())
# print(df.info())
# print(df.describe())
#print(df.columns) 
#['WTT', 'PTI', 'EQW', 'SBI', 'LQE', 'QWG', 'FDJ', 'PJF', 'HQE', 'NXJ', 'TARGET CLASS']
print('------------------------------------------------------------------------\n')

# Preprocess
X = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']
scalar = StandardScaler()
scalar.fit(X)
scaled_features = scalar.transform(X)
# df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
# print(df_feat.head())

X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.30, random_state=101)

# Train
print('When K=1')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)

# Evaluation
print('Confustion Matrix = \n', confusion_matrix(y_test, predictions))
print()
print('Classification Report = \n', classification_report(y_test, predictions))

print('------------------------------------------------------------------------\n')

# Choosing a K value
# error_rate = []
# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, y_train)
#     predict_i = knn.predict(X_test)
#     error_rate.append(np.mean(predict_i != y_test))
    
# fig = plt.figure(figsize=(10, 6))
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.grid()
# plt.show()

# With K=23
print('When K=23')
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print('Confustion Matrix = \n', confusion_matrix(y_test, predictions))
print()
print('Classification Report = \n', classification_report(y_test, predictions))
