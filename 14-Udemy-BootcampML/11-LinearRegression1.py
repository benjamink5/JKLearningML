import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load Data
USAhousing = pd.read_csv("./Data/USA_Housing.csv")

# Display Info
# print(USAhousing.head())
# print(USAhousing.info())
# print(USAhousing.describe())
print(USAhousing.columns)
# ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
# 'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']

# Data Analysis 
# sns.pairplot(USAhousing)
# sns.displot(USAhousing['Price'])
# sns.heatmap(USAhousing.corr())

# plt.show()

# Prepare Train and Test data
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Training model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Evaluate the model
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# Predict 
predictions = lm.predict(X_test)
# plt.scatter(y_test, predictions)
# sns.displot((y_test - predictions), bins=50)
# plt.show()

print("Train Score:", lm.score(X_train, y_train))
print("Test Score:", lm.score(X_test, y_test))

# Metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))