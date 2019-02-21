import matplotlib.pyplot as plt 
import matplotlib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 


def prep_train_data(train_data): 
    train_data_df = pd.read_csv(train_data)
    y_train = np.array(train_data_df['label'])
    X_train = np.array(train_data_df.drop(['label'], axis = 1))
    #X_train = X_train.reshape(42000, 28, 28, 1)   #reshape n-features into 2d array 
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 101)
  
    return X_train, X_test, y_train, y_test 
  
 
def prep_train_data_nosplit(train_data): 
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
    print("image label = {}".format(label)) 
	
    return
	

	



