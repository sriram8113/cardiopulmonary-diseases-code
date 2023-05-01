#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


df = pd.read_csv("Cleaned_heart_cholestrol_data.csv")
df.head(10)

df.drop(columns = ['index', 'Unnamed: 0'], inplace = True)
df.head()


df1 = df[['age', 'trestbps']]
df2 = df[['chol', 'thalach']]


from sklearn.preprocessing import MinMaxScaler

# define min max scaler
scaler = MinMaxScaler()
# transform data
df1 = pd.DataFrame(scaler.fit_transform(df1), columns = df1.columns)
df2 = pd.DataFrame(scaler.fit_transform(df2), columns = df2.columns)
print(df1)

df1
df2


from sklearn.linear_model import LinearRegression


X = df1[['age']]
y = df1['trestbps']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train
X_test.head(10)
y_train
y_test.head(10)


# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients and intercept of the model
print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)

slope = model.coef_[0]
intercept = model.intercept_

# Print the equation of the line
print(f"Resting Blood Pressure = {slope:.2f} * age + {intercept:.2f}")


model.score(X, y)
y_pred = model.predict(X_test)


from sklearn.metrics import r2_score
accuracy = r2_score(y_test, y_pred)
print('Accuracy:', accuracy)

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))


plt.scatter(X, y, alpha=0.5)
plt.plot(X_test['age'], y_pred, 'r', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.title( 'Age Vs Resting Blood Pressure')
plt.legend()
plt.show()

sns.jointplot(x="age", y="trestbps", data=df1, kind="reg")
plt.show()

sns.regplot(x="age", y="trestbps", data=df1);
plt.title( 'Age Vs Resting Blood Pressure')
plt.show()

sns.lmplot(x="age", y="trestbps", data=df1);
plt.title( 'Age Vs Resting Blood Pressure')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot

# Assuming y_test and y_pred are the actual and predicted values, respectively
residuals = y_test - y_pred

# Residual plot
sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual plot')
plt.show()

# Q-Q plot
probplot(residuals, plot=plt)
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles')
plt.title('Q-Q plot')
plt.show()

# Scale-location plot
fig, ax = plt.subplots()
sns.regplot(x=y_pred, y=np.sqrt(np.abs(residuals)), lowess=True, color="g", ax=ax)
ax.set(xlabel='Predicted values', ylabel='Sqrt(|Residuals|)', title='Scale-Location plot')
plt.show()


import statsmodels.api as sm

#define response variable
y = df1['trestbps']

#define predictor variables
x = df1[['age']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())


# Select the two columns of interest
X1 = df2[['chol']]
y1 = df2['thalach']


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)


# Fit the linear regression model
model1 = LinearRegression()
model1.fit(X1, y1)

# Print the coefficients and intercept of the model
print("Coefficient: ", model1.coef_)
print("Intercept: ", model1.intercept_)

slope1 = model1.coef_[0]
intercept1 = model1.intercept_

# Print the equation of the line
print(f"Maximun heart rate achieved = {slope1:.2f} * cholestrol + {intercept1:.2f}")

y1_pred = model1.predict(X1_test)

accuracy1 = r2_score(y1_test, y1_pred)
print('Accuracy:', accuracy1)

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y1_test, y1_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y1_test, y1_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y1_test, y1_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y1_test, y1_pred), 2)) 
print("R2 score =", round(sm.r2_score(y1_test, y1_pred), 2))

import matplotlib.pyplot as plt

plt.scatter(X1, y1, alpha=0.5)
plt.plot(X1_test['chol'], y1_pred, 'g', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.title( 'Age Vs Resting Blood Pressure')
plt.legend()
plt.show()

sns.jointplot(x="chol", y="thalach", data=df2, kind="reg")

sns.regplot(x='chol', y="thalach", data=df2);
plt.title( 'Chol Vs Max heart rate achieved')

sns.lmplot(x='chol', y="thalach", data=df2);
plt.title( 'Chol Vs Max heart rate achieved')

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot

# Assuming y_test and y_pred are the actual and predicted values, respectively
residuals1 = y1_test - y1_pred

# Residual plot
sns.residplot(x=y1_pred, y=residuals1, lowess=True, color="g")
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual plot')
plt.show()

# Q-Q plot
probplot(residuals1, plot=plt)
plt.xlabel('Theoretical quantiles')
plt.ylabel('Sample quantiles')
plt.title('Q-Q plot')
plt.show()

# Scale-location plot
fig, ax = plt.subplots()
sns.regplot(x=y1_pred, y=np.sqrt(np.abs(residuals1)), lowess=True, color="g", ax=ax)
ax.set(xlabel='Predicted values', ylabel='Sqrt(|Residuals|)', title='Scale-Location plot')
plt.show()

import statsmodels.api as sm

#define response variable
y = df2['thalach']

#define predictor variables
x = df2[['chol']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model = sm.OLS(y, x).fit()

#view model summary
print(model.summary())
