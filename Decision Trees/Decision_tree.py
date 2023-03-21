# import required packages 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#Reading the data 
df = pd.read_csv('Cleaned_heart_cholestrol_data.csv')

print(df.head(5))

# sleecting only numerical continous data 
X = df[['age', 'trestbps','chol','thalach', "oldpeak"]]

print(X)

print(X.isnull().sum())

print(X.describe())


print(X.info())

# normalising the data 
names = X.columns

from sklearn import preprocessing

d = preprocessing.normalize(X)
X = pd.DataFrame(d, columns=names)
X
y = df['target']

# test train splitting 

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)

print(X_train)

print(y_train)

################################### Grid search to find best parameters 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# specify the hyperparameters you want to tune
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(5,12),
    'min_samples_split': range(1,12),
    'min_samples_leaf': range(1, 12)
}

# create a decision tree classifier
dtc = DecisionTreeClassifier()

# create a grid search object
grid_search = GridSearchCV(dtc, parameters, cv=5)

# fit the grid search object to your data
grid_search.fit(X, y)

# print the best hyperparameters and corresponding accuracy score
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)


###########################################################################

#Model Implementation 

decision_tree = DecisionTreeClassifier(random_state=0, max_depth = 5, min_samples_split=2,min_samples_leaf = 3)
decision_tree = decision_tree.fit(X_train, y_train)

# Accuracy of testing and training sets 

fit_accuracy = decision_tree.score(X_train, y_train)
test_accuracy = decision_tree.score(X_test, y_test)
    
print("Train accuracy: {:.2%}".format(fit_accuracy))
print("Test accuracy: {:.2%}".format(test_accuracy))   
print(decision_tree)

#plotting tree

tree.plot_tree(decision_tree)
plt.show()

text_representation = tree.export_text(decision_tree)
print(text_representation)

fig = plt.figure(figsize=(100,100))
_ = tree.plot_tree(decision_tree, 
                   feature_names=X.columns, 
                   class_names= str(df.target),
                   filled=True)
fig.savefig("decision_tree11.png")

import graphviz
# DOT data
dot_data = tree.export_graphviz(decision_tree, out_file=None, 
                                feature_names=X.columns,  
                                class_names=str(df.target),
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph
graph.render("decision_tree_graphivz11")

#Making Prediction 
decision_tree_prediction = decision_tree.predict(X_test)

#Confusion Matrix 

bn_matrix_R = confusion_matrix(y_test, decision_tree_prediction)
print("\nThe confusion matrix is:")
print(bn_matrix_R)

# Finding Important Features 

FeatureImpR=decision_tree.feature_importances_   
indicesR = np.argsort(FeatureImpR)[::-1]
indicesR
feature_namesR = X.columns
print ("feature name: ", feature_namesR[indicesR])

## print out the important features.....
for f in range(X_train.shape[1]):
    if FeatureImpR[indicesR[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indicesR[f], FeatureImpR[indicesR[f]]))
        print ("feature name: ", feature_namesR[indicesR[f]])

labels_predicted = decision_tree.predict(X_test)
plt.subplots(figsize=(10,5))

#Plotting confusion matrix 

conf_mat = confusion_matrix(y_test, labels_predicted)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Decision Tree: Confusion Matrix')
plt.show()

print(conf_mat)
print(classification_report(y_test, labels_predicted))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB

from yellowbrick.classifier import ClassificationReport
from yellowbrick.datasets import load_occupancy

# visualising classification report 

visualizer = ClassificationReport(decision_tree, support=True)

visualizer.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show()

#Finding weights of the features 
perm = PermutationImportance(decision_tree, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())

html = eli5.show_weights(perm, feature_names=X_test.columns.tolist()).data
with open('feature_importance3.html', 'wb') as f:
    f.write(html.encode('utf-8'))

########################################################## Similar code with different data and different parameters ################################################

df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)
       
standardScaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

print(df.head(10))
df.drop(columns = ['index', 'Unnamed: 0'], inplace = True)
df.head()


y1 = df['target']
X1 = df.drop(['target'], axis = 1)

X1_train , X1_test , y1_train , y1_test = train_test_split(X1 , y1 , test_size = 0.2 , random_state = 42)

print(X1_train)

print(y1_train)

#####################################################################################


# specify the hyperparameters you want to tune
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(5,12),
    'min_samples_split': range(1,12),
    'min_samples_leaf': range(1, 12)
}

# create a decision tree classifier
dtc = DecisionTreeClassifier()

# create a grid search object
grid_search = GridSearchCV(dtc, parameters, cv=5)

# fit the grid search object to your data
grid_search.fit(X1, y1)

# print the best hyperparameters and corresponding accuracy score
print("Best parameters: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)

########################################################################################

decision_tree1 = DecisionTreeClassifier(random_state=0, max_depth = 7, min_samples_split=6, criterion = 'entropy', min_samples_leaf = 4)
decision_tree1 = decision_tree1.fit(X1_train, y1_train)

fit_accuracy1 = decision_tree1.score(X1_train, y1_train)
test_accuracy1 = decision_tree1.score(X1_test, y1_test)
print("Train accuracy: {:.2%}".format(fit_accuracy1))
print("Test accuracy: {:.2%}".format(test_accuracy1))   


from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get the mean accuracy score
scores = cross_val_score(decision_tree1, X1, y1, cv=kf)
print("Cross-validation scores: ", scores)
print("Mean score: ", np.mean(scores))


fig = plt.figure(figsize=(100,100))
_ = tree.plot_tree(decision_tree1, 
                   feature_names=X1.columns, 
                   class_names= str(df.target),
                   filled=True)
fig.savefig("decision_tree12.png")

import graphviz
# DOT data
dot_data = tree.export_graphviz(decision_tree1, out_file=None, 
                                feature_names=X1.columns,  
                                class_names=str(df.target),
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph
graph.render("decision_tree_graphivz12")

decision_tree_prediction1 = decision_tree1.predict(X1_test)

bn_matrix_R1 = confusion_matrix(y1_test, decision_tree_prediction1)
print("\nThe confusion matrix is:")
print(bn_matrix_R1)

FeatureImpR1=decision_tree1.feature_importances_   
indicesR1 = np.argsort(FeatureImpR1)[::-1]
indicesR1
feature_namesR1 = X1.columns
print ("feature name: ", feature_namesR1[indicesR1])

## print out the important features.....
for f1 in range(X1_train.shape[1]):
    if FeatureImpR1[indicesR1[f1]] > 0:
        print("%d. feature %d (%f)" % (f1 + 1, indicesR1[f1], FeatureImpR1[indicesR1[f1]]))
        print ("feature name: ", feature_namesR1[indicesR1[f1]])

labels_predicted1 = decision_tree1.predict(X1_test)
plt.subplots(figsize=(10,5))

conf_mat1 = confusion_matrix(y1_test, labels_predicted1)
sns.heatmap(conf_mat1, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Decision Tree: Confusion Matrix')

print(conf_mat1)
print(classification_report(y1_test, labels_predicted1))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB

from yellowbrick.classifier import ClassificationReport
from yellowbrick.datasets import load_occupancy
visualizer1 = ClassificationReport(decision_tree1, support=True)

visualizer1.fit(X1_train, y1_train)        # Fit the visualizer and the model
visualizer1.score(X1_test, y1_test)        # Evaluate the model on the test data
visualizer1.show()

perm1 = PermutationImportance(decision_tree1, random_state=1).fit(X1_test, y1_test)
eli5.show_weights(perm1, feature_names = X1_test.columns.tolist())

html1 = eli5.show_weights(perm1, feature_names=X1_test.columns.tolist()).data
with open('feature_importance4.html', 'wb') as f:
    f.write(html1.encode('utf-8'))

######################################################################################################
decision_tree2 = DecisionTreeClassifier(random_state=0, max_depth = 5, min_samples_split=2,min_samples_leaf = 3, criterion = 'entropy')
decision_tree2 = decision_tree2.fit(X_train, y_train)

fit_accuracy2 = decision_tree2.score(X_train, y_train)
test_accuracy2 = decision_tree2.score(X_test, y_test)
print("Train accuracy: {:.2%}".format(fit_accuracy2))
print("Test accuracy: {:.2%}".format(test_accuracy2))   


from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get the mean accuracy score
scores = cross_val_score(decision_tree2, X, y, cv=kf)
print("Cross-validation scores: ", scores)
print("Mean score: ", np.mean(scores))


fig = plt.figure(figsize=(100,100))
_ = tree.plot_tree(decision_tree2, 
                   feature_names=X.columns, 
                   class_names= str(df.target),
                   filled=True)
fig.savefig("decision_tree13.png")

import graphviz
# DOT data
dot_data1 = tree.export_graphviz(decision_tree2, out_file=None, 
                                feature_names=X.columns,  
                                class_names=str(df.target),
                                filled=True)

# Draw graph
graph1 = graphviz.Source(dot_data1, format="png") 
graph1
graph1.render("decision_tree_graphivz13")

decision_tree_prediction2 = decision_tree2.predict(X_test)

bn_matrix_R2 = confusion_matrix(y_test, decision_tree_prediction2)
print("\nThe confusion matrix is:")
print(bn_matrix_R2)

FeatureImpR2=decision_tree2.feature_importances_   
indicesR2 = np.argsort(FeatureImpR2)[::-1]
indicesR2
feature_namesR2 = X.columns
print ("feature name: ", feature_namesR2[indicesR2])

## print out the important features.....
for f2 in range(X_train.shape[1]):
    if FeatureImpR2[indicesR2[f2]] > 0:
        print("%d. feature %d (%f)" % (f2 + 1, indicesR2[f2], FeatureImpR2[indicesR2[f2]]))
        print ("feature name: ", feature_namesR2[indicesR2[f2]])

labels_predicted2 = decision_tree2.predict(X_test)
plt.subplots(figsize=(10,5))

conf_mat2 = confusion_matrix(y_test, labels_predicted2)
sns.heatmap(conf_mat2, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Decision Tree: Confusion Matrix')


print(conf_mat2)
print(classification_report(y_test, labels_predicted2))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB

from yellowbrick.classifier import ClassificationReport
from yellowbrick.datasets import load_occupancy
visualizer2 = ClassificationReport(decision_tree2, support=True)

visualizer2.fit(X_train, y_train)        # Fit the visualizer and the model
visualizer2.score(X_test, y_test)        # Evaluate the model on the test data
visualizer2.show()

perm2 = PermutationImportance(decision_tree2, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm2, feature_names = X_test.columns.tolist())

html2 = eli5.show_weights(perm2, feature_names=X_test.columns.tolist()).data
with open('feature_importance5.html', 'wb') as f:
    f.write(html2.encode('utf-8'))
