# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:18:08 2020

@author: Vishwas Basotra
"""
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# importing dataset
iris_df = pd.read_csv('dataset/iris.csv')

# label encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris_df.iloc[:,-1] = le.fit_transform(iris_df.iloc[:,-1])

# visualizing data
## correlation btw features
iris_df.iloc[:,:-1].corrwith(iris_df.iloc[:,-1]).plot(kind = 'bar',
                  figsize=(20,10),
                  color = 'steelblue',
                  fontsize = 15,
                  rot = 45,
                  grid = True
                  )
plt.title('Correlation between Independent and Dependent Features', fontsize=20)
plt.ylabel('Correlation')
plt.xlabel('Features')
plt.show()

## correlation matrix
plt.figure(figsize=(20,10))
sns.set(font_scale=1.5)
sns.heatmap(iris_df.corr(),
            linewidth=0.5,
            annot=True,
            fmt='g',
            cmap="RdBu_r"
            )
plt.xticks(rotation=45)
plt.title('Correlation Matrix', fontsize=25)
plt.show()

# separating dependent and independent features
X = iris_df.iloc[:,:-1].values
y = iris_df.iloc[:,-1].values

# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# feature scaling the independent features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN model
## initializing model
model = tf.keras.models.Sequential()

## adding layers to the model
model.add(tf.keras.layers.Dense(units=4,input_shape=[4,],activation='relu'))
model.add(tf.keras.layers.Dense(units=3,activation='softmax'))

## compliling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

## training the model
early_stop = tf.keras.callbacks.EarlyStopping(patience=10)
model.fit(x = X_train,
          y = y_train,
          batch_size=10,
          epochs=300,
          validation_data=(X_test,y_test),
          callbacks=[early_stop]
          )

metrics = pd.DataFrame(model.history.history)
print(metrics.head())
print(metrics.tail())

## plotting loss vs validation loss
plt.figure(figsize=(20,10))
metrics[['loss','val_loss']].plot()
plt.title('loss Vs val_loss',fontsize=15)
plt.xlabel('loss',fontsize=10)
plt.ylabel('val_loss',fontsize=10)
plt.show()

## plotting accuracy vs val_accuracy
plt.figure(figsize=(20,10))
metrics[['accuracy','val_accuracy']].plot()
plt.title('accuracy Vs val_accuracy',fontsize=15)
plt.xlabel('accuracy',fontsize=10)
plt.ylabel('val_accuracy',fontsize=10)
plt.show()

# evaluating the model
evaluate = model.evaluate(X_test,y_test, verbose=0)
print('Final loss: {:.2f}%'.format(evaluate[0]*100))
print('Final Accuracy: {:.2f}%'.format(evaluate[1]*100))
      
# model deployment
## training all of my data
epochs = len(metrics)
X = sc.fit_transform(X)

## ANN model
### initializing model
model = tf.keras.models.Sequential()

### adding layers to the model
model.add(tf.keras.layers.Dense(units=4,input_shape=[4,],activation='relu'))
model.add(tf.keras.layers.Dense(units=3,activation='softmax'))

### compliling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

### training the model
early_stop = tf.keras.callbacks.EarlyStopping(patience=10)
model.fit(x = X,
          y = y,
          batch_size=10,
          epochs=300,
          callbacks=[early_stop]
          )

### saving the model
model.save('final_iris_model.h5')

### saving the scaler
import joblib
joblib.dump(sc,'iris_scaler.pkl')

## predictions:
flower_model = tf.keras.models.load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

flower_example = {"sepal_length":5.1,
                  "sepal_width":3.5,
                  "petal_length":1.4,
                  "petal_width":0.2
                  }

def return_prediction(model,scaler,sample_json):
    s_len = sample_json["sepal_length"]
    s_width = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_width = sample_json["petal_length"]
    
    flower = [[s_len,s_width,p_len,p_width]]
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    flower = scaler.transform(flower)
    
    class_ind = model.predict_classes(flower)[0]
    
    return classes[class_ind]

return_prediction(flower_model,flower_scaler,flower_example)

# code for deployment
flower_model = tf.keras.models.load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')
def return_prediction(model,scaler,sample_json):
    s_len = sample_json["sepal_length"]
    s_width = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_width = sample_json["petal_length"]
    
    flower = [[s_len,s_width,p_len,p_width]]
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    flower = scaler.transform(flower)
    
    class_ind = model.predict_classes(flower)[0]
    
    return classes[class_ind]