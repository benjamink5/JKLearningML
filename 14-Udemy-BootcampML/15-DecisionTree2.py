import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Data Load
df = pd.read_csv('./Data/loan_data.csv')
print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
# ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 
#  'days.with.cr.line', 'revol.bal', 'revol.util','inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid']
print('------------------------------------------------------------------------\n')

# Data Analysis
# plt.figure(figsize=(10, 6))
# df[df['credit.policy'] == 1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='Credit.Policy=1')
# df[df['credit.policy'] == 0]['fico'].hist(alpha=0.5, color='red', bins=30, label='Credit.Policy=0')
# plt.legend()
# plt.xlabel('FICO')
# plt.show()

# plt.figure(figsize=(10, 6))
# df[df['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='not.fully.paid=1')
# df[df['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, color='red', bins=30, label='not.fully.paid=0')
# plt.legend()
# plt.xlabel('FICO')
# plt.show()

# Show the counts of loans by purpose
# plt.figure(figsize=(11, 7))
# sns.countplot(x='purpose', hue='not.fully.paid', data=df, palette='Set1')
# plt.show()

# Show the trend between FICO score and interest rate
# sns.jointplot(x='fico', y='int.rate', data=df, color='purple')
# plt.show()

# show the trend differed between not.fully.paid and credit.policy.
# sns.lmplot(y='int.rate', x='fico', data=df, hue='credit.policy', col='not.fully.paid', palette='Set1')
# plt.show()

# Preprocess
# Notice that the purpose column as categorical
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
final_data = pd.get_dummies(df, columns=['purpose'], drop_first=True)
final_data.info()

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Prediction
predictions = dtree.predict(X_test)

# Evaluation
print("Decision Tree")
print("Confustion Matrix:\n", confusion_matrix(y_test, predictions))
print()
print("Classification Report:\n", classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')


print("Random Forests")
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print('Confusion Matrix =\n', confusion_matrix(y_test, rfc_pred))
print()
print('Classification Report =\n', classification_report(y_test, rfc_pred))
print('------------------------------------------------------------------------\n')
