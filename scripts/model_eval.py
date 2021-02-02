import pandas as pd 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


data = pd.read_csv('../data/test.csv')
y_pred = data['LABELS']
y = pd.read_csv('../data/test_labels.csv')

print('\nROC AUC score', roc_auc_score(y,y_pred))
print('\nAccuracy score:', accuracy_score(y,y_pred))
print('\nConfusion matrix:')
print(confusion_matrix(y,y_pred))