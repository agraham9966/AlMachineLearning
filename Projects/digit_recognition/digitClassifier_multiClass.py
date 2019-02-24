from digitClassifier_util import * 
from sklearn.linear_model import SGDClassifier 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve 
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
import sys, os 
import numpy as np
import pandas as pd 

##Stochastic Gradient Descent 

def plot_prec_recall_v_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], 'b--', label = "Precision")
    plt.plot(thresholds, recalls[:-1], 'g-', label = "Recall") 
    plt.xlabel("Threshold") 
    plt.legend(loc="center left") 
    plt.ylim([0,1]) 
	
def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.show()
	
def SGDclassifier_eval(model, threshold, X_train, y_train): #for class-based or categorical classifier results 
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix 
    
    y_train_pred = cross_val_predict(model, X_train, y_train, cv=3, method='decision_function') # returns predictions made on each test fold 
    y_train_pred = (y_train_pred > threshold) 
    conf_mat = confusion_matrix(y_train, y_train_pred) ##top-left, top-right, bot-left, bot-right = true negatives, false positives, false negatives, true-positives 
    
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN) 
	
    F1_HM = (FN + FP) / 2
    F1 = TP / (TP + F1_HM)  
    
    print('precision = {} , recall = {}, F1 = {}'.format(precision, recall, F1))
	
    return 

#Split data for training / testing 
X_train, X_test, y_train, y_test = prep_train_data(sys.argv[1]) # takes training.csv as argument
sgd_clf = SGDClassifier(random_state=42) 
sgd_clf.fit(X_train, y_train) 
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))

scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))


## error analysis 

y_train_predict = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_predict)
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show() 
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show() 


