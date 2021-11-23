import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, plot_confusion_matrix

cancer = load_breast_cancer()
cancer_data = np.array(cancer.data)
# print(cancer_data.shape)
df_cancer = pd.DataFrame(cancer_data, columns=cancer.feature_names)
print(df_cancer)

X_train, X_test, y_train, y_test = train_test_split(cancer_data, cancer.target, stratify=cancer.target)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print("acc : ", accuracy_score(y_test, pred))
print("roc_auc_score : ", roc_auc_score(y_test, pred))
print("recall_score : ", recall_score(y_test, pred))
print("f1_score : ", f1_score(y_test, pred))
print("precision_score : ", precision_score(y_test, pred))

print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))
DN = confusion_matrix(y_test, pred)[0][0]
YP = confusion_matrix(y_test, pred)[0][1]
YN = confusion_matrix(y_test, pred)[1][0]
DP = confusion_matrix(y_test, pred)[1][1]
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
plot_confusion_matrix(knn, X_test, y_test)
plt.title("KNN Classification Confusion Matrix")
plt.show()