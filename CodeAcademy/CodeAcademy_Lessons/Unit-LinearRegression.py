##list comprehension example : https://www.pythonforbeginners.com/basics/list-comprehensions-in-python
import codecademylib3_seaborn
import matplotlib.pyplot as plt
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

#slope:
m = 10
#intercept:
b = 50

plt.plot(months, revenue, "o")

plt.show()
  
y = [m * i + b for i in months] ##multiply m by each element in the list and create new list 

a += b is essentially the same as a = a + b, except that:

+ always returns a newly allocated object, but += should (but doesn't have to) modify the object in-place if it's mutable (e.g. list or dict, but int and str are immutable).
In a = a + b, a is evaluated twice.
http://docs.python.org/reference/simple_stm

import codecademylib3_seaborn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

temperature = np.array(range(60, 100, 2))
temperature = temperature.reshape(-1, 1)
sales = [65, 58, 46, 45, 44, 42, 40, 40, 36, 38, 38, 28, 30, 22, 27, 25, 25, 20, 15, 5]

plt.plot(temperature, sales, 'o')
plt.show()

line_fitter = LinearRegression() ##creates model for linear regression 
line_fitter.fit(temperature, sales) ##fits model to x, y

sales_predict = line_fitter.predict(temperature) ##predicts new y based on x 
plt.plot(temperature, sales_predict, '-')
plt.show()