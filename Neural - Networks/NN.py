#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Reading the data

df = pd.read_csv("Cleaned_heart_cholestrol_data.csv")
df.head(10)


df.drop(columns = ['index', 'Unnamed: 0'], inplace = True)
df.head()


df['cp'] = df.cp.replace({"typical_angina" : 1, 
                          "atypical_angina": 2, 
                          "non-anginal pain": 3,
                          "asymtomatic": 4})
df['exang'] = df.exang.replace({"Yes": 1, "No": 2})
df['fbs'] = df.fbs.replace({"True":1,"False":0})
df['slope'] = df.slope.replace({"upsloping":1, "flat":2,"downsloping":3})
df['thal'] = df.thal.replace({"thal_fixed_defect":6, "thal_reversable_defect":7, "thal_normal":3})
df['sex'] = df.sex.replace({"Male":1, "Female":0})
df['restecg'] = df.restecg.replace({"ecg_normal":0, 
                          "ecg_having ST-T wave abnormality ":1, 
                          "ecg_showing probable or definite left ventricular hypertrophy":2})

df.info()

df['fbs'] = df['fbs'].astype(int)

df.info()
df = df.apply(pd.to_numeric)
df.dtypes
df.head(10)

X = df.drop('target',axis=1).values
y = df['target'].values

from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)

X_train

X_test
y_train

y_test


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train

X_test

#pip install keras

#pip install tensorflow

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers
from tensorflow.keras.layers import Activation

def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())

history = binary_model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=500, batch_size=10)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Model Losss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# generate classification report using predictions for binary model
from sklearn.metrics import classification_report, accuracy_score
# generate classification report using predictions for binary model 
binary_pred = np.round(binary_model.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(y_test, binary_pred))
print(classification_report(y_test, binary_pred))

conf_mat = confusion_matrix(y_test, binary_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Neural Networks: Confusion Matrix')
plt.show()


from tensorflow.keras.callbacks import EarlyStopping


# # Early stopping 

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)



history1 = binary_model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=500, batch_size=10, callbacks=[early_stop])


# Model Losss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# # Adding dropout layers 

def create_binary_model1():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model1 = create_binary_model1()

print(binary_model1.summary())


history2 = binary_model1.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=500, batch_size=10)


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# Model Losss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()



# generate classification report using predictions for binary model
from sklearn.metrics import classification_report, accuracy_score
# generate classification report using predictions for binary model 
binary_pred1 = np.round(binary_model1.predict(X_test)).astype(int)

print('Results for Binary Model')
print(accuracy_score(y_test, binary_pred1))
print(classification_report(y_test, binary_pred1))




conf_mat = confusion_matrix(y_test, binary_pred1)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Neural Networks: Confusion Matrix')
plt.show()



history3 = binary_model1.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=500, batch_size=10, callbacks=[early_stop])



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Model accuracy
plt.plot(history3.history['accuracy'])
plt.plot(history3.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()



# Model Losss
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


binary_model.save_weights

# assume `model` is your neural network model
weights = binary_model.get_weights()

# print the shape of the weight matrices
for i, w in enumerate(weights):
    print(f"Weight matrix {i}: {w.shape}")

# assume `model` is your neural network model
weights = binary_model1.get_weights()

# print the shape of the weight matrices
for i, w in enumerate(weights):
    print(f"Weight matrix {i}: {w.shape}")

num_layers = len(binary_model.layers)

print("Number of layers:", num_layers)

num_layers1 = len(binary_model1.layers)

print("Number of layers:", num_layers1)

