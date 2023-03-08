import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline

# Data Load
yelp = pd.read_csv('./Data/yelp.csv')
print(yelp.head())
# print(yelp.info())
# print(yelp.describe())
# print(yelp.columns)
# 'business_id', 'date', 'review_id', 'stars', 'text', 'type', 'user_id', 'cool', 'useful', 'funny'
print('------------------------------------------------------------------------\n')

yelp['text length'] = yelp['text'].apply(len)

# Data Anaysis
sns.set_style('white')

# g = sns.FacetGrid(yelp, col='stars')
# g.map(plt.hist, 'text length')
# plt.show()

# sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')
# plt.show()

# sns.countplot(x='stars', data=yelp, palette='rainbow')
# plt.show()

# stars = yelp.groupby('stars').mean()
# print(stars)
# print(stars.corr())

# sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)
# plt.show()

# Preprocess
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]
X = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()
X = cv.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Prediction
predictions = nb.predict(X_test)
print(predictions[:10])
print(y_test.shape)
print(y_test[:10])
# for i in range(10):
#     print(predictions[i], y_test[i])

# Evaluation
print("Confustion Matrix:\n", confusion_matrix(y_test, predictions))
print()
print('Classification Report =\n', classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')


# Using Text Processing
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)

# Evaluation
print("Uisng Text Processing: CountVectorizer --> TfidfTransformer --> MultinominalNB")
print("Confustion Matrix:\n", confusion_matrix(y_test, predictions))
print()
print('Classification Report =\n', classification_report(y_test, predictions))
print('------------------------------------------------------------------------\n')
