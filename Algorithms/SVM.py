from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Makes concentric circles
points, labels = make_circles(n_samples=300, factor=.2, noise=.05, random_state = 1)

#Makes training set and validation set.
training_data, validation_data, training_labels, validation_labels = train_test_split(points, labels, train_size = 0.8, test_size = 0.2, random_state = 100)

classifier = SVC(kernel = "linear", random_state = 1)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
print(training_data[0])

new_training = [[2 ** 0.5 * pt[0] * pt[1], pt[0] ** 2, pt[1] ** 2] for pt in training_data]
new_validation = [[2 ** 0.5 * pt[0] * pt[1], pt[0] ** 2, pt[1] ** 2] for pt in validation_data]

classifier.fit(new_training, training_labels)
print(classifier.score(new_validation, validation_labels))

import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()
https://www.youtube.com/watch?v=d-5KFI8A_ZA
def SVM_Pitcher_SZone_Score(player): 
  player['type'] = player['type'].map({'S':1, 'B':0})
  player = player.dropna(subset = ['plate_x', 'plate_z', 'type'])
#print(aaron_judge.head())
  plt.scatter(x = player['plate_x'], y = player['plate_z'], c =   player['type'],cmap = plt.cm.coolwarm, alpha = 0.25)
  training_set, validation_set = train_test_split(player, random_state = 1)
  classifier = SVC(kernel = "rbf", gamma = 3, C = 1)
  classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
  draw_boundary(ax, classifier)
  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3)
  plt.show()
  score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
  return score

score = SVM_Pitcher_SZone_Score(david_ortiz)
print(score)

