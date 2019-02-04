import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt 
import sys, os 


flags = pd.read_csv(sys.argv[1], header=0)
print(flags.columns)
labels = flags[['Language']]
##let's try and predict flag only from color 
data = flags[['Red', 'Green', 'Blue',
 'Gold', 'White', 'Black', 'Orange', "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"
 ]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels)
tree = DecisionTreeClassifier(random_state=1)
clf = tree.fit(train_data, train_labels)
print(tree.score(test_data, test_labels)) # not a great initial score but better than random guessing 

export_graphviz(
    clf, 
    out_file='iris.dot'
)


# prune_depth = range(1, 21)
# scores = []

# for i in prune_depth: 
#     tree = DecisionTreeClassifier(random_state=1, max_depth=i)
#     tree.fit(train_data, train_labels)
#     scores.append(tree.score(test_data, test_labels))

# plt.plot(prune_depth, scores)
# plt.show()