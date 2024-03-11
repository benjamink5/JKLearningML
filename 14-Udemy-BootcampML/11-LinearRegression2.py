'''
Problem Definition
Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online 
but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, 
then they can go home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website. 
They've hired you on contract to help them figure it out! Let's get started!
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Get the Data

customers = pd.read_csv('./Data/Ecommerce_Customers.csv')
# print(customers.head())
# print(customers.describe())
# print(customers.info())
#print(customers.columns)
# Columns: ['Email', 'Address', 'Avatar', 'Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership', 'Yearly Amount Spent']

# Data Visualization and Analysis
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)

# sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)

# sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)

# sns.pairplot(customers)

# sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

# plt.show()

# Training and Testing Data
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Training Model
lm = LinearRegression()
lm.fit(X_train, y_train)
print('Coefficents:', lm.coef_)

# Predict Test Data
predictions = lm.predict(X_test)

# plt.scatter(y_test, predictions)
# plt.show()

print("Train Score:", lm.score(X_train, y_train))
print("Test Score:", lm.score(X_test, y_test))

# Evaluating the Model
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('R^2:', metrics.explained_variance_score(y_test, predictions))

# Residuals.
sns.displot((y_test - predictions), bins=50)
plt.xlabel('Y Test')
plt.ylabel('Predctied Y')
plt.show()

# Conclusion
coefficients = pd.DataFrame(lm.coef_, X.columns)
coefficients.columns = ['Coefficient']
print(coefficients)

'''
Interpreting the coefficients:
- Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
- Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
- Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
- Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.
'''

'''
Do you think the company should focus more on their mobile app or on their website?
This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, 
or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on at the company, 
you would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!
'''