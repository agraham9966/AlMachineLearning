##N-Fold Cross Validation - used when your dataset is too small to split into training and validation datasets
##do the process N times and average the accuracy 
##10-fold cross validation - make the validation set the first 10% of the data and calculate accuracy, precision, recall and F1 score. 
##We then so this with the 2nd 10% of the data and repeat until the end. 
##then we average the 10 accuracies. 

import codecademylib3_seaborn
import pandas as pd

streeteasy = pd.read_csv("https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

print(df.head()) ##shows first few rows of dataframe 

##training v. test set - In general, putting 80% of your data in the training set and 20% of your data 
##in the test set is a good place to start.

import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split 

# import train_test_split

streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape) ##array.shape returns [rows, columns]


mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# Input code here:

print(model.coef_)

##Residual analysis 
#Residual e is the diff between the actual y and the predicted y 
#coefficient of determination R-2 is 1 - (u/v). u is residual sum of squares ((y - y-predict)**2).sum(), v is total sum of squares (TSS) 
##TSS tells how much variation there is in the y variable 
train set = 0.77
test set = 0.8


