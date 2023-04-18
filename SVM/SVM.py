#importing required packages and libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import roc_curve, auc
from sklearn import svm

#reading the dataset
df = pd.read_csv('Cleaned_heart_cholestrol_data.csv')
df.head()

df.target.value_counts()

#dropping unnecessary columnms 
df.drop(columns = ['index', 'Unnamed: 0'], inplace = True)
df.head()

#selcting continuous numerical data 
columns = ['age','trestbps','chol','thalach','oldpeak','target']
data_svm = df[columns]

#scaling the data 
standardScaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
data_svm[columns_to_scale] = standardScaler.fit_transform(data_svm[columns_to_scale])
data_svm


#separating data and target varaiable
labels = data_svm['target']
features = data_svm.drop(['target'], axis = 1)

#test train splitting the data 
features_train , features_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.2, random_state=42)

#choosing 3 different c values 
Cs = [0.1, 1, 10]
for C in Cs:
    #model instantiation
    clflinear = svm.SVC(kernel='linear', C=C, probability=True)
    
    #fitting into model
    clflinear.fit(features_train, labels_train)
    
    #plotting confusion matrix
    plot_confusion_matrix(clflinear, features_test, labels_test, display_labels=['No Disease', 'Disease'])
    plt.title(f'SVM with linear kernel (C={C})')
    plt.show()
    print('\n')
    
    #predicting the test data with the model
    labels_predicted_linear = clflinear.predict(features_test) 
    labels_predicted_linear
    
    #classification report printing
    print(classification_report(labels_test, labels_predicted_linear))
    
    #plotting roc curve 
    fpr, tpr, thresholds = roc_curve(labels_test, clflinear.predict_proba(features_test)[:,1])
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    #taking 2 features and plotting the model using scatterplot 
    y_pred = clflinear.predict(features_test)
    plt.figure()
    plt.scatter(features_test.iloc[:, 0], features_test.iloc[:, 1], c=y_pred, cmap=plt.cm.Paired)
    plt.xlabel('age')
    plt.ylabel('trestbps')
    plt.title(f'linear kernel')
    plt.show()
    
    #Feature importance of the model 
    importances = clflinear.coef_

    # Print feature importances
    for i in range(importances.shape[1]):
        print("Feature ", data_svm.columns[i], " : ", importances[0][i])


data_svm.columns[0]


Cs = [0.1, 1, 10]
for C in Cs:
    clfrbf = svm.SVC(kernel='rbf', C=C, probability=True)
    clfrbf.fit(features_train, labels_train)
    plot_confusion_matrix(clfrbf, features_test, labels_test, display_labels=['No Disease', 'Disease'])
    plt.title(f'SVM with RBF kernel (C={C})')
    plt.show()
    print('\n')
    labels_predicted_rbf = clfrbf.predict(features_test) 
    labels_predicted_rbf
    print(classification_report(labels_test, labels_predicted_rbf))
    
    fpr, tpr, thresholds = roc_curve(labels_test, clfrbf.predict_proba(features_test)[:,1])
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    y_pred = clfrbf.predict(features_test)
    plt.figure()
    plt.scatter(features_test.iloc[:, 0], features_test.iloc[:, 1], c=y_pred, cmap=plt.cm.Paired)
    plt.xlabel('age')
    plt.ylabel('trestbps')
    plt.title(f'linear kernel')
    plt.show()
    
    # Estimate feature importances using permutation feature importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(clfrbf, features, data_svm.target, n_repeats=10, random_state=0)
    importances = result.importances_mean

    # Print feature importances
    for i in range(importances.shape[0]):
        print("Feature ",data_svm.columns[i], " : ", importances[i])


Cs = [0.1, 1, 10]
for C in Cs:
    clfpoly = svm.SVC(kernel='poly', degree=3, C=C, probability=True)
    clfpoly.fit(features_train, labels_train)
    plot_confusion_matrix(clfpoly, features_test, labels_test, display_labels=['No Disease', 'Disease'])
    plt.title(f'SVM with polynomial kernel (degree=3, C={C})')
    plt.show()
    print('\n')
    labels_predicted_poly = clfpoly.predict(features_test) 
    labels_predicted_poly
    print(classification_report(labels_test, labels_predicted_poly))
    
    fpr, tpr, thresholds = roc_curve(labels_test, clfpoly.predict_proba(features_test)[:,1])
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    y_pred = clfpoly.predict(features_test)
    plt.figure()
    plt.scatter(features_test.iloc[:, 0], features_test.iloc[:, 1], c=y_pred, cmap=plt.cm.Paired)
    plt.xlabel('age')
    plt.ylabel('trestbps')
    plt.title(f'linear kernel')
    plt.show()
    
    # Estimate feature importances using permutation feature importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(clfpoly, features, data_svm.target, n_repeats=10, random_state=0)
    importances = result.importances_mean

    # Print feature importances
    for i in range(importances.shape[0]):
        print("Feature ",data_svm.columns[i], " : ", importances[i])


Cs = [0.1, 1, 10]
for C in Cs:
    clfsig = svm.SVC(kernel='sigmoid', C=C, probability=True)
    clfsig.fit(features_train, labels_train)
    plot_confusion_matrix(clfsig, features_test, labels_test, display_labels=['No Disease', 'Disease'])
    plt.title(f'SVM with Sigmoid kernel , (C={C})')
    plt.show()
    
    print('\n')
    labels_predicted_sig = clfsig.predict(features_test) 
    labels_predicted_sig
    print(classification_report(labels_test, labels_predicted_sig))
    
    fpr, tpr, thresholds = roc_curve(labels_test, clfsig.predict_proba(features_test)[:,1])
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    y_pred = clfsig.predict(features_test)
    plt.figure()
    plt.scatter(features_test.iloc[:, 0], features_test.iloc[:, 1], c=y_pred, cmap=plt.cm.Paired)
    plt.xlabel('age')
    plt.ylabel('trestbps')
    plt.title(f'linear kernel')
    plt.show()
    
    # Estimate feature importances using permutation feature importance
    from sklearn.inspection import permutation_importance
    result = permutation_importance(clfsig, features, data_svm.target, n_repeats=10, random_state=0)
    importances = result.importances_mean

    # Print feature importances
    for i in range(importances.shape[0]):
        print("Feature ", data_svm.columns[i], " : ", importances[i])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

labels = data_svm['target']
X = pd.DataFrame( data_svm[['age', 'trestbps']])
y = data_svm.target

# fit the model
C = 5.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C),
          svm.SVC(kernel='sigmoid', degree=3, C=C))

models = (clf.fit(X, y) for clf in models)

# plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ('SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          'SVC with sigmoid kernel')

fig, sub = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for clf, title, ax in zip(models, titles, sub.flatten()):
    # plot decision boundary
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # plot the scatter plot of data points
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Age')
    ax.set_ylabel('Trestbps')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

###############################################################################
#pca is done for the visualisation of the model 

from sklearn.decomposition import PCA

#components as 2 

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(features)

pcadata = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


pcadata

X = pcadata.copy()
y = labels


###################################### C =0.1 #################################
# fit the model
C = 0.1  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C),
          svm.SVC(kernel='sigmoid', degree=3, C=C))

models = (clf.fit(X, y) for clf in models)

# plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ('SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          'SVC with sigmoid kernel')

fig, sub = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for clf, title, ax in zip(models, titles, sub.flatten()):
    # plot decision boundary
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # plot the scatter plot of data points
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

###################################### C =1 ###################################
# fit the model
C = 1  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C),
          svm.SVC(kernel='sigmoid', degree=3, C=C))

models = (clf.fit(X, y) for clf in models)

# plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ('SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          'SVC with sigmoid kernel')

fig, sub = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for clf, title, ax in zip(models, titles, sub.flatten()):
    # plot decision boundary
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # plot the scatter plot of data points
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


###################################### C =10 ##################################
# fit the model
C = 10  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C),
          svm.SVC(kernel='sigmoid', degree=3, C=C))

models = (clf.fit(X, y) for clf in models)

# plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ('SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          'SVC with sigmoid kernel')

fig, sub = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for clf, title, ax in zip(models, titles, sub.flatten()):
    # plot decision boundary
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # plot the scatter plot of data points
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

