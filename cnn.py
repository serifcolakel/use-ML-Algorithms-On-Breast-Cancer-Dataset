import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, roc_auc_score, recall_score, \
    f1_score, precision_score, classification_report

cancerData = datasets.load_breast_cancer()
X = pd.DataFrame(data = cancerData.data, columns=cancerData.feature_names )
X.head()
y = cancerData.target
X.shape
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1,stratify=y)
X_train.shape
y_test.shape

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(512,30,1)
X_test = X_test.reshape(57,30,1)

model = Sequential()

model.add(Conv1D(filters=16,kernel_size=2,activation='relu',input_shape=(30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(32,2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=100,verbose=1,validation_data=(X_test,y_test))

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
if cv2.waitKey(1) & 0xFF == ord("q"):
  pass

print("acc : ", accuracy_score(y_test, y_pred))
print("roc_auc_score : ", roc_auc_score(y_test, y_pred))
print("recall_score : ", recall_score(y_test, y_pred))
print("f1_score : ", f1_score(y_test, y_pred))
print("precision_score : ", precision_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

DN = confusion_matrix(y_test, y_pred)[0][0]
YP = confusion_matrix(y_test, y_pred)[0][1]
YN = confusion_matrix(y_test, y_pred)[1][0]
DP = confusion_matrix(y_test, y_pred)[1][1]
dogruluk = (DP + DN)/(DP+DN+YP+YN)
belirginlik = DN / (DN + YP)
hassasiyet = DP / (DP + YP)
duyarlılık = DP / (DP + YN)
F1SCORE = 2 * ((hassasiyet*duyarlılık)/(hassasiyet+duyarlılık))
ROC = DP / (DP+YN)
AUC = YP / (YP+DN)
print("acc :", dogruluk)
print("Belirginlik : ", belirginlik)
print("Hassasiyet : ", DP / (DP + YP))
print("duyarlılık : ", DP / (DP + YN))
print("F1-SCORE : ", F1SCORE)
print("ROC-AUC", ROC)


def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

plotLearningCurve(history,100)


