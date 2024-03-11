import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def converter(cluster):
    if cluster=='Yes':
        return 0
    else:
        return 1
    
# Data Load
df = pd.read_csv('./Data/College_Data.csv')
print(df.head())
# print(df.info())
# print(df.describe())
# print(df.columns)
# 'Unnamed: 0', 'Private', 'Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F.Undergrad', 'P.Undergrad', 'Outstate', 'Room.Board',
# 'Books', 'Personal', 'PhD', 'Terminal', 'S.F.Ratio', 'perc.alumni', 'Expend', 'Grad.Rate'
print('------------------------------------------------------------------------\n')

# Data Analysis
# sns.set_style('whitegrid')
# sns.lmplot(data=df, x='Room.Board', y='Grad.Rate', hue='Private', palette='coolwarm', aspect=1, fit_reg=False)
# plt.show()

# sns.set_style('whitegrid')
# sns.lmplot(data=df, x='Outstate', y='F.Undergrad', hue='Private', palette='coolwarm', aspect=1, fit_reg=False)
# plt.show()

# sns.set_style('darkgrid')
# g = sns.FacetGrid(data=df, hue='Private', palette='coolwarm', aspect=2, height=6)
# g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
# plt.show()

# sns.set_style('darkgrid')
# g = sns.FacetGrid(data=df, hue='Private', palette='coolwarm', aspect=2, height=6)
# g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
# plt.show()

print(df[df['Grad.Rate'] > 100])
df['Grad.Rate']['Cazenovia College'] = 100
print(df[df['Grad.Rate'] > 100])
print('------------------------------------------------------------------------\n')

# Preprocess
kmeans = KMeans(n_clusters=2)

# Train
df = df.drop('Unnamed: 0', axis=1)
kmeans.fit(df.drop('Private', axis=1))
print(kmeans.cluster_centers_)
print(kmeans.labels_[:5])

# Evaluate
df['Cluster'] = df['Private'].apply(converter)
print(df.head())

print("Confustion Matrix:\n", confusion_matrix(df['Cluster'], kmeans.labels_))
print()
print('Classification Report =\n', classification_report(df['Cluster'], kmeans.labels_))
print('------------------------------------------------------------------------\n')

