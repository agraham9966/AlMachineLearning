##honey production 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import sys
import os 

dir1 = sys.argv[1] 
#mylist = listdir(dir1) 

#path, filename = os.path.split(dir1)
#regrab = dir1 + '/*.vrt'
#outfilename = dir1 + '/Mosaic.vrt'

df = pd.read_csv(dir1)

print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index() ##reset index ensures index order for each row remains 

X = prod_per_year[['year']]
X = X.values.reshape(-1, 1) ##what does this do? Puts each value in a list format it seems 

y = prod_per_year[['totalprod']]
y = y.values.reshape(-1, 1) 

plt.scatter(X, y) 
#plt.show()

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_[0]) ##prints the slope 


y_predict = regr.predict(X)
print(y_predict)

plt.plot(X, y_predict, '-') 
plt.scatter(X, y) ##shows honey decline 
#plt.show()

X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1, 1) ##restructure the numbers so they resemble a column of numbers 
future_predict = regr.predict(X_future)
plt.plot(X_future, future_predict) ##shows the predicted production up to 2050 using the model 
plt.show()

