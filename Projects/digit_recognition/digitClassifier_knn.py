from six.moves import urllib
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
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
print(X.shape) # #images, image shape (sqrt(X.shape[1]))
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

print(X_train.shape) 






