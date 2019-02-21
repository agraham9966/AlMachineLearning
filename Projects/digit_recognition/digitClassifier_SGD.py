from digitClassifier_util import * 
from sklearn.linear_model import SGDClassifier 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_auc_score
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

#Classifying the number 7 
#Split data for training / testing 
X_train, X_test, y_train, y_test = prep_train_data(sys.argv[1]) # takes training.csv as argument 
#set num 7 for bool training data 
y_train_7 = (y_train == 7) 
#set model and fit classifier 
sgd_clf = SGDClassifier(random_state=42) 
sgd_clf.fit(X_train, y_train_7)  
#get scores 
##################run this portion to determine an optimal threshold - then run classifier eval to get overall scores from new threshold 
y_scores_SGD = cross_val_predict(sgd_clf, X_train, y_train_7, cv=3, method='decision_function') #returns predictions before thresholding - wheras default method uses set threshold of zero 
#precisions, recalls, thresholds = precision_recall_curve(y_train_7, y_train_scores) 
#plot_prec_recall_v_threshold(precisions, recalls, thresholds) 
#plt.plot(precisions, recalls)
#plt.show() ## shows optimal threshold that can be chosen for SGD Classifier 

SGDclassifier_eval(sgd_clf, -70000, X_train, y_train_7) 
#output (threshold = -70000): precision = 0.9068175052098839 , recall = 0.8633786848072562, F1 = 0.8845651226949325 


##############
from sklearn.ensemble import RandomForestClassifier 

forest_clf = RandomForestClassifier(random_state=42) 
y_prob_forest = cross_val_predict(forest_clf, X_train, y_train_7, cv=3, method='predict_proba') 
y_scores_forest = y_prob_forest[:, 1] #score = proba of a positive class 

#####The ROC Curve ##########Compare with RandomForest 
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_7, y_scores_forest) 
fpr_SGD, tpr_SGD, thresholds_SGD = roc_curve(y_train_7, y_scores_SGD) 

plt.plot(fpr_SGD, tpr_SGD, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right") 
plt.show()




