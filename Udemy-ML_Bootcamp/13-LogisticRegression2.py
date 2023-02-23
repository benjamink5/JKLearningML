'''
In this project we will be working with a fake advertising data set, 
indicating whether or not a particular internet user clicked on an Advertisement. 
We will try to create a model that will predict whether or not they will click on 
an ad based off the features of that user.

This data set contains the following features:

    'Daily Time Spent on Site': consumer time on site in minutes
    'Age': cutomer age in years
    'Area Income': Avg. Income of geographical area of consumer
    'Daily Internet Usage': Avg. minutes a day consumer is on the internet
    'Ad Topic Line': Headline of the advertisement
    'City': City of consumer
    'Male': Whether or not consumer was male
    'Country': Country of consumer
    'Timestamp': Time at which consumer clicked on Ad or closed window
    'Clicked on Ad': 0 or 1 indicated clicking on Ad
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load Data
advertises = pd.read_csv('./DATA/advertising.csv')
print(advertises.head())
print(advertises.info())
print(advertises.describe())
# print(advertises.columns)
# ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Ad Topic Line', 
#  'City', 'Male', 'Country', 'Timestamp', 'Clicked on Ad']

# Exploratory Data Analysis
# sns.set_style('whitegrid')
# sns.displot(advertises['Age'], bins=30)

# g = sns.jointplot(data=advertises, x="Age", y="Area Income")

# g = sns.jointplot(data=advertises, kind="kde", x="Age", y="Daily Time Spent on Site")

# g = sns.jointplot(data=advertises, x="Daily Time Spent on Site", y="Daily Internet Usage")

# sns.pairplot(data=advertises, hue="Clicked on Ad")

# plt.show()

# Select data
X = advertises[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = advertises['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Prediction
predictions = lr.predict(X_test)

# Report
print("Train Score:", lr.score(X_train, y_train))
print("Test Score:", lr.score(X_test, y_test))
print(classification_report(y_test, predictions))