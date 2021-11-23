import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, plot_confusion_matrix

cancer = load_breast_cancer()
# print(cancer.keys()) # keyler
# print(cancer["DESCR"]) # açıklama
#Tablo halinde verileri gösterelim
df_feat = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
# print(df_feat)
df_target = pd.DataFrame(cancer["target"], columns=["cancer"])
# print(df_target)

# print(cancer["target_names"]) # ['malignant' 'benign']

# print(df_feat.head())
# print(df_target.head())

X_train, X_test, y_train, y_test = train_test_split(df_feat, df_target, test_size=0.33, random_state=42)
svc = SVC()
svc.fit(X_train, y_train)
pred = svc.predict(X_test)

print(confusion_matrix(y_test, pred))
plot_confusion_matrix(svc, X_test, y_test)
plt.title("SVM Classification Confusion Matrix")
plt.show()
DN = confusion_matrix(y_test, pred)[0][0]
YP = confusion_matrix(y_test, pred)[0][1]
YN = confusion_matrix(y_test, pred)[1][0]
DP = confusion_matrix(y_test, pred)[1][1]
print("acc :", (DP + DN)/(DP+DN+YP+YN))
print("acc : ", accuracy_score(y_test, pred))
print("roc_auc_score : ", roc_auc_score(y_test, pred))
print("recall_score : ", recall_score(y_test, pred))
print("f1_score : ", f1_score(y_test, pred))
print("precision_score : ", precision_score(y_test, pred))

print(classification_report(y_test, pred))