#Add / import required packages

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import eli5
from eli5.sklearn import PermutationImportance

# Data Reading 

df = pd.read_csv('Cleaned_heart_cholestrol_data.csv')
print(df.head())


print(df.target.value_counts())


# dropping waste columns 
df.drop(columns = ['index', 'Unnamed: 0'], inplace = True)
df.head()


df.target.value_counts()

#copying into new varaiable 
df2 = df.copy()


df1 = df.copy()

#scaling the numerical data uisng stnadard scaler 

standardScaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

# conevrting categorical variables using one hot encoding 
df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)

df
#splitting data into training and testing 

labels = df['target']
features = df.drop(['target'], axis = 1)

features_train , features_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.2, random_state=42)


pd.DataFrame(labels_train).target.value_counts()

pd.DataFrame(labels_test).target.value_counts()

# Naive bayes model Implememnttaion 
from sklearn.naive_bayes import GaussianNB

nb1 = GaussianNB() # Instantiation
nb1.fit(features_train, labels_train) # fitting into model

labels_predicted = nb1.predict(features_test) # making predictions 
print(labels_predicted)

# checking accuracy of training and testing sets 
fit_accuracy = nb1.score(features_train, labels_train)
test_accuracy = nb1.score(features_test, labels_test)
    
print(f"Train accuracy: {fit_accuracy:0.2%}")
print(f"Test accuracy: {test_accuracy:0.2%}")


labels_predicted = nb1.predict(features_test)
plt.subplots(figsize=(10,5))

#plotting confusion matrix 

conf_mat = confusion_matrix(labels_test, labels_predicted)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Naive Bayes: Confusion Matrix')
plt.show()

print(conf_mat)

# Finding important features 

perm = PermutationImportance(nb1, random_state=1).fit(features_test, labels_test)
eli5.show_weights(perm, feature_names = features_test.columns.tolist())

# classification report
print(classification_report(labels_test, labels_predicted))


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# confusion matrix 
CM = pd.crosstab(labels_test, labels_predicted)
CM

TN=CM.iloc[0,0]
FP=CM.iloc[0,1]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]

# False negative rates 
fnr2 = FN*100/(FN+TP)
print("False Negative rate : {}". format(fnr2))


from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB

from yellowbrick.classifier import ClassificationReport
from yellowbrick.datasets import load_occupancy

# Visualising classification report 

visualizer = ClassificationReport(nb1, support=True)

visualizer.fit(features_train, labels_train)        # Fit the visualizer and the model
visualizer.score(features_test, labels_test)        # Evaluate the model on the test data
visualizer.show()


################################################ Only on continuous numerical data

# reading data 
df2 = pd.read_csv('Cleaned_heart_cholestrol_data.csv')

# considering only continous numerical data 
columns = ['age','trestbps','chol','thalach','oldpeak','target']
data_naive = df2[columns]


data_naive

#sclaing data using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data_naive[columns_to_scale] = scaler.fit_transform(data_naive[columns_to_scale])

data_naive

# splitting data into test train splits 

labels2 = data_naive['target']
features2 = data_naive.drop(['target'], axis = 1)


features2_train , features2_test, labels2_train, labels2_test = train_test_split(features2, labels2, test_size= 0.2, random_state=42)
features2_train.head(10)
features2_test.head(10)
labels2_train.head(10)
labels2_test.head(10)

#Implementation of the model 
from sklearn.naive_bayes import GaussianNB

nb2 = GaussianNB() #instantiation 
nb2.fit(features2_train, labels2_train) # fitting into model

labels_predicted_nb2 = nb2.predict(features2_test) # predictions 
print(labels_predicted_nb2)

# Accurcay of training and testing 
fit_accuracy1 = nb2.score(features2_train, labels2_train)
test_accuracy1 = nb2.score(features2_test, labels2_test)
    
print(f"Train accuracy: {fit_accuracy1:0.2%}")
print(f"Test accuracy: {test_accuracy1:0.2%}")

# classificiation report
print(classification_report(labels2_test, labels_predicted_nb2))

labels_predicted_nb2 = nb2.predict(features2_test)
plt.subplots(figsize=(10,5))

#plotting confusion matrix 
conf_mat2 = confusion_matrix(labels2_test, labels_predicted_nb2)
sns.heatmap(conf_mat2, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Gaussian: Confusion Matrix')
plt.show()

conf_mat2 

# finding important features 
perm1 = PermutationImportance(nb2, random_state=1).fit(features2_test, labels2_test)
eli5.show_weights(perm1, feature_names = features2_test.columns.tolist())

perm1

# visualising classification report 
visualizer = ClassificationReport(nb2, support=True)

visualizer.fit(features2_train, labels2_train)        # Fit the visualizer and the model
visualizer.score(features2_test, labels2_test)        # Evaluate the model on the test data
visualizer.show()

# confusion matrix 
bn_matrix_R = confusion_matrix(labels2_test, labels_predicted_nb2)
print("\nThe confusion matrix is:")
print(bn_matrix_R)


