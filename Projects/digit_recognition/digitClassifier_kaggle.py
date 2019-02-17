import matplotlib.pyplot as plt 
import matplotlib
import sys, os 
import numpy as np
import pandas as pd 



def prep_train_data(train_data): 
    train_data = pd.read_csv(train_data) 
    data_train_num = train_data.iloc[:, 1:]
    data_train_labels = train_data.iloc[:, 0:1]

    return data_train_num, data_train_labels
	
def show_digit_image(train_data, index): 
    '''
    dataset consists of each row as a 1d array - to be reshaped into 2d array for visualization 
    '''
    train_data = pd.read_csv(train_data) 
    data_train_num = train_data.iloc[:, 1:]
    
    some_digit = data_train_num.iloc[index, :].values.reshape(28,28)
	
    plt.imshow(some_digit, cmap = matplotlib.cm.binary, interpolation = 'nearest') 
    plt.axis("off") 
    plt.show()
	
    label = train_data.iloc[index, 0]
    print "image label = {}".format(label) 
	
    return
	
	
train_data, train_labels = prep_train_data(sys.argv[1]) 
