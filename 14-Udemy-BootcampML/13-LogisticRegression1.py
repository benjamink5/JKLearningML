import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
            
train = pd.read_csv('./Data/titanic_train.csv')
# print(train.head())
# print(train.columns)
# print(train.info())
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# Exploratory Data Anaysis

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis') # Show the missing data

# sns.set_style('whitegrid')
# sns.countplot(x='Survived', data=train, palette='RdBu_r')

# sns.set_style('whitegrid')
# sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')

# sns.set_style('whitegrid')
# sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')

# sns.displot(train['Age'].dropna(), kde=False, color='darkred', bins=30)
# plt.show()

# Data Cleaning
# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')
# plt.show()

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show()

train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train = pd.concat([train, sex, embark], axis=1)
# print(train.head())

# Train Test Split
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Predicting
predictions = logmodel.predict(X_test)
# plt.scatter(y_test, predictions)
# sns.displot((y_test - predictions), bins=50)
# plt.show()

# Evaluation
print("Train Score:", logmodel.score(X_train, y_train))
print("Test Score:", logmodel.score(X_test, y_test))

print(classification_report(y_test, predictions))