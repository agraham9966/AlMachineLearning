import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets 
from sklearn.cluster import KMeans

digits = datasets.load_digits() 
print(digits) ##shows everything 
print(digits.DESCR) ##shows info 
print(digits.data) ##shows what data looks like , each list contains 64 values , 0 is white, 16 is black (0-16 range) 
print(digits.target) ## what each datapoint/image was tagged as 

plt.gray() 
plt.matshow(digits.images[1700])
plt.show()

print(digits.target[1700])##shows image and shows what it is actually labelled as , in this case the number 5

model = KMeans(n_clusters=10, random_state = 38)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

###new_samples is an array of 4 new handwritten digits to try and classify 
new_samples = np.array([
[0.00,3.03,5.33,5.33,5.02,3.04,0.45,0.00,3.49,7.62,6.39,5.40,6.85,7.62,5.71,0.00,6.47,6.00,0.23,0.00,0.00,5.03,7.54,1.90,6.85,3.96,0.00,0.00,0.00,0.61,6.93,5.03,6.45,6.55,0.76,0.00,0.00,0.00,6.32,5.33,2.13,7.39,7.31,4.11,2.51,4.34,7.62,3.58,0.00,1.51,5.18,7.62,7.62,7.62,4.79,0.08,0.00,0.00,0.00,0.30,0.76,0.68,0.00,0.00],
[0.00,0.00,0.00,0.53,1.44,0.68,0.00,0.00,0.30,5.10,6.33,7.62,7.60,6.39,0.00,0.00,0.31,5.10,5.26,3.42,6.31,6.78,0.00,0.00,0.00,0.00,0.00,5.31,7.62,6.86,3.57,0.61,0.00,0.00,0.00,5.63,6.10,6.32,7.62,6.85,0.00,0.00,0.00,0.00,0.00,0.00,4.40,7.62,2.21,5.34,5.33,5.33,5.48,6.40,7.62,5.48,2.21,5.34,5.33,5.33,5.34,4.95,3.41,0.22],
[0.00,3.34,4.56,4.57,2.20,0.00,0.00,0.00,0.30,7.23,6.85,7.00,6.85,0.00,0.00,0.00,1.44,7.62,2.73,4.34,7.47,0.53,0.00,0.00,0.99,7.39,7.32,6.32,7.62,1.67,0.00,0.00,0.00,1.44,4.57,6.47,7.54,4.40,0.00,0.00,0.00,0.00,0.00,0.00,5.33,6.09,0.00,0.00,0.00,0.00,0.00,0.00,4.50,6.02,0.00,0.00,0.00,0.00,0.00,0.00,0.53,0.84,0.00,0.00],
[0.00,0.00,0.00,0.15,1.22,1.14,0.00,0.00,0.00,2.89,4.80,7.01,7.62,7.55,1.07,0.00,0.00,5.03,6.32,5.03,3.42,7.62,2.51,0.00,0.00,0.00,0.00,0.00,0.23,7.62,3.27,0.00,0.00,0.00,0.00,0.91,4.94,7.62,3.73,0.00,0.00,0.22,5.55,7.61,7.62,7.15,3.65,0.46,0.00,0.38,5.86,6.48,6.86,7.23,7.62,1.83,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')

