import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, plot_confusion_matrix

from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()

df = pd.DataFrame(
    np.c_[breast_cancer.data, breast_cancer.target],
    columns = [list(breast_cancer.feature_names)+ ['target']]
                 )
# dataset split X and y
df_feat = pd.DataFrame(breast_cancer["data"], columns=breast_cancer["feature_names"])
df_target = pd.DataFrame(breast_cancer["target"], columns=["cancer"])
X_train, X_test, y_train, y_test = train_test_split(df_feat, df_target, test_size = 0.2, random_state = 0)
logisticRegressor =LogisticRegression()
logisticRegressor.fit(X_train, y_train)
pred = logisticRegressor.predict(X_test)


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

plot_confusion_matrix(logisticRegressor, X_test, y_test)
plt.title("Logistic Regression Classification Confusion Matrix")
plt.show()
