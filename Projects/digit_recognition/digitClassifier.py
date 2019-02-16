from six.moves import urllib
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict 
from sklearn.metrics import confusion_matrix 
from sklearn.base import clone 
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np

#try different solutions from here
#https://github.com/ageron/handson-ml/issues/7
#the try code below would return operation timed out as the web is down
#try to import data
    # Alternative method to load MNIST, if mldata.org is down
from scipy.io import loadmat
def data_down():
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
    return mnist_path

#run this if it is the first time or you need to update
#mnist_path = data_down()

#since we already download the dataset, we could directly use them
mnist_path = "./mnist-original.mat"
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
    }
X, y = mnist['data'], mnist['target']
print(X.shape) 
####################################################################################################################################
#some_digit = X[36000] 
# some_digit_image = some_digit.reshape(28, 28) 
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest') 
# plt.axis("off") 
# plt.show() 
# print(y[36050]) 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] #data already split 
shuffle_index = np.random.permutation(60000) 
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

##simplify the problem - let's see if we can classify a single digit: #5 
y_train_5 = (y_train == 5) # true for all 5's, false for all else 
# let's try stochastic gradient descent (SGD) 
some_digit = X[36000] 
sgd_clf = SGDClassifier(random_state=42) 
sgd_clf.fit(X_train, y_train_5) 
print(sgd_clf.predict([some_digit])) 

#####performance measures - Stratified Cross Validation 
# skfolds = StratifiedKFold(n_splits=3, random_state=42) 

# for train_index, test_index in skfolds.split(X_train, y_train_5): 
    # clone_clf = clone(sgd_clf) 
    # X_train_folds = X_train[train_index]
    # y_train_folds = y_train_5[train_index]
    # X_test_fold = X_train[test_index] 
    # y_test_fold = y_train_5[test_index] 

    # clone_clf.fit(X_train_folds, y_train_folds) 
    # y_pred = clone_clf.predict(X_test_fold) 
    # n_correct = sum(y_pred == y_test_fold) 
    # print(n_correct / len(y_pred))

# ##cross_val_score 
# cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy') 

#########using confusion matrix to eval. performance of classifier 
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
#cross_val_predict performs k-folds- but returns predictions made on each test fold.
print(confusion_matrix(y_train_5, y_train_pred)) 
#first row represents non-5 images (53000 correctly classed as non-5s, )
##  [true-negative, false-pos
#    false-negative, true-pos]
 
#   [non-5s corr classified, non-5s misclassified as 5
#    5s classed as non 5s, 5s classed as 5s]



