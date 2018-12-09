##Distance Formula 

star_wars = [125, 1977, 11000000]
raiders = [115, 1981, 18000000]
mean_girls = [97, 2004, 17000000]

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

print(distance(star_wars, raiders))
print(distance(star_wars, mean_girls))

release_dates = [1897.0, 1998.0, 2000.0, 1948.0, 1962.0, 1950.0, 1975.0, 1960.0, 2017.0, 1937.0, 1968.0, 1996.0, 1944.0, 1891.0, 1995.0, 1948.0, 2011.0, 1965.0, 1891.0, 1978.0]

def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = []
  
  for value in lst:
    normalized_num = (value - minimum) / (maximum - minimum)
    normalized.append(normalized_num)
  
  return normalized

print(min_max_normalize(release_dates))

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, k): 
  distances = [] 
  for title in dataset: 
    distance_to_point =distance(dataset[title], unknown)
    distances.append([distance_to_point, title])
    distances.sort()
  neighbors = distances[0:k]
  return neighbors

print(classify([.4, .2, .9], movie_dataset, 5))

#################

from movies import movie_dataset, movie_labels

print("Troll 2" in movie_dataset) ##checks if title is in dataset 
my_movie = [200000, 95, 1990]

normalized_my_movie = normalize_point(my_movie)

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  num_good = 0
  num_bad = 0
  for neighbor in neighbors:
    title = neighbor[1]
    if labels[title]:
      num_bad += 1
    elif labels[title] == 1:
      num_good += 1
  if num_good > num_bad:
    return 1
  else:
    return 0
classify(normalized_my_movie, movie_dataset, movie_labels, 5)


def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
  num_correct = 0.0
  for title in validation_set:
    guess = classify(validation_set[title], training_set, training_labels, k)
    if guess == validation_labels[title]:
      num_correct += 1
  return num_correct / len(validation_set)


print(find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, 3))
    
def predict(unknown, dataset, movie_ratings, k): ##predicts based on the average movie rating for k-nn
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  total = 0
  for n in neighbors: 
    title = n[1]
    total += movie_ratings[title]
  average = total / k
  return average 

print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))  


def predict(unknown, dataset, movie_ratings, k):##gives weighted average instead 
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]
  numerator = 0
  denominator = 0
  for neighbor in neighbors:
    rating = movie_ratings[neighbor[1]]
    distance_to_neighbor = neighbor[0]
    numerator += rating / distance_to_neighbor
    denominator += 1 / distance_to_neighbor
  return numerator / denominator

print(predict([0.016, 0.300, 1.022], movie_dataset, movie_ratings, 5))

###########use SKlearn with weighted avg regression 
from movies import movie_dataset, movie_ratings
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors = 5, weights = "distance")
regressor.fit(movie_dataset, movie_ratings)

unknown_points = [
  [0.016, 0.3, 1.022],
  [0.0004092981, 0.283, 1.0112],
  [0.00687649, 0.235, 1.0112]
]

guesses = regressor.predict(unknown_points)
print(guesses)
 