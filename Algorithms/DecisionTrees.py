from collections import Counter
from sklearn import tree

##gini impurity calculation - to be calculated on the leaves of the dec. tree 

#labels = ["unacc", "unacc", "acc", "acc", "good", "good"]
#labels = ["unacc","unacc","unacc", "good", "vgood", "vgood"]
#labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc"]

# unsplit_labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "good", "good", "vgood", "vgood", "vgood"]

# split_labels_1 = [
  # ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "vgood"], 
  # [ "good", "good"], 
  # ["vgood", "vgood"]
# ]

# split_labels_2 = [
  # ["unacc", "unacc", "unacc", "unacc","unacc", "unacc", "good", "good", "good", "good"], 
  # ["vgood", "vgood", "vgood"]
# ]


cars = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def gini(dataset): 
    impurity = 1
    label_counts = Counter(dataset) 
    for label in label_counts: 
        probability_of_label = (label_counts[label] / len(dataset))**2 
        impurity -= probability_of_label

    return impurity


def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels) 
    for subset in split_labels: 
        info_gain -= (len(subset) / len(starting_labels)) * gini(subset) ## weighted multiplication of each leaf to account for impurity values when leaves have diffent lengths/# labels 

    return info_gain		

	
def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets
	
def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain
	
def build_tree(data, labels):
  best_feature, best_gain = find_best_split(data, labels)
  if best_gain == 0:
    return Counter(labels)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    branch = build_tree(data_subsets[i], label_subsets[i])
    branches.append(branch)
  return branches
  
# tree = build_tree(car_data, car_labels)
# print_tree(tree) 	
	
print(Counter(car_labels))
#split_data, split_labels = split(cars, car_labels, 3)
# print(information_gain(car_labels, split_labels)) ##gives info gain for one feature at index 3

# for feature in range(len(cars[0])):  ##loops through dataset and prints the info_gain from each feature 
    # split_data, split_labels = split(cars, car_labels, feature) 
    # print(information_gain(car_labels, split_labels))


