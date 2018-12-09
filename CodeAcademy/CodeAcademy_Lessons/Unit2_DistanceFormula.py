##Distance Formula - Defining Distance 

#3 Different ways to define distance between points: 
#      1. Euclidean Distance 
#      2. Manhattan Distance 
#      3. Hamming Distance 

pt1 = [5, 8] ##2d point 
pt1 = [4, 8, 15, 16, 23] ##5d point 
distance([1, 2, 3], [5, 8, 9]) ##distance between 2 3d points--- they must be same dimension 

pt1 = [4, 5] 
pt2 = [5, 6]

##Euclidean Distance - To find this calculate the squared distance between two points in each dimension,. 
##If we add up the squared distances and take the square root, that is euclidean distance. 


def euclidean_distance(pt1, pt2): 
  distance = 0 
  for i in range(len(pt1)): ##len returns number of elements in a list 
    distance = distance + (pt1[i]- pt2[i])**2
  return distance 
  
##manhattan distance - takes the sum of the absolute difference between points 
def manhattan_distance(pt1, pt2): 
  distance = 0 
  for i in range(len(pt1)): 
    distance = distance + abs(pt1[i] - pt2[i])
  return distance 
  
# Hamming Distance is another slightly different variation on the distance formula. 
# Instead of finding the difference of each dimension, Hamming distance only cares 
# about whether the dimensions are exactly equal. When finding the Hamming distance 
# between two points, add one for every dimension that has different values.

# Hamming distance is used in spell checking algorithms. For example, the Hamming 
# distance between the word "there" and the typo "thete" is one. Each letter is a 
# dimension, and each dimension has the same value except for one.

def hamming_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    if pt1[i] != pt2[i]:
      distance += 1
  return distance

print(hamming_distance([1, 2], [1, 100]))
print(hamming_distance([5, 4, 9], [1, 7, 9]))


