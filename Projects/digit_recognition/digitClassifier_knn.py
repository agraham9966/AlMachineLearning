from digitClassifier_util import * 
from sklearn.linear_model import SGDClassifier 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve 
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
import sys, os 
import numpy as np
import pandas as pd 
from scipy.ndimage.interpolation import shift

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

def plot_digit_image(train_data, index = None): 
    '''
    dataset consists of each row as a 1d array - to be reshaped into 2d array for visualization 
    '''
    if index != None: 
        some_digit = train_data[index].reshape(28,28)
    else: 
        some_digit = train_data.reshape(28,28)
	
    plt.imshow(some_digit, cmap = matplotlib.cm.binary, interpolation = 'nearest') 
    plt.axis("off") 
    plt.show()
	
    return


#Split data for training / testing 
#https://github.com/ageron/handson-ml/blob/master/03_classification.ipynb
X_train, X_test, y_train, y_test = prep_train_data(sys.argv[1]) # takes training.csv as argument

def shift_digit(digit_array, dx, dy, new=0):
    return shift(digit_array.reshape(28, 28), [dy, dx], cval=new).reshape(784)


shifted = shift_digit(X_train[100], 5, 1, new=100)

plot_digit_image(X_train, 100)
plot_digit_image(shifted)

#plot_digit_image(shift_digit(X_train[100], 5, 1, new=100), 100)





# knn_clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
# knn_clf.fit(X_train, y_train)
# y_pred = knn_clf.predict(X_train) 
# print(y_pred)
# accuracy = accuracy_score(y_train, y_pred)
# recall = recall_score(y_train, y_pred)

# print('precision={}, accuracy={}, recall={}'.format(precision, accuracy, recall))