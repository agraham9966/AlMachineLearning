import pandas as pd 
import sys 
import os 
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

survey = pd.read_csv(sys.argv[1]) 
# print(survey) ##answers how many rows there are
#print(survey['q0007_0007'].value_counts()) ##shows histogram of answers for this question
#print(survey.columns.values) ##shows column names 

	   
cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003",
 "q0007_0004", "q0007_0005", "q0007_0006",
 "q0007_0007", "q0007_0008", "q0007_0009",
 "q0007_0010", "q0007_0011"]
	   
for col in cols_to_map: 
    survey[col] = survey[col].map({
	"Never, and not open to it": 0, 
	"Never, but open to it": 1, 
	"Rarely": 2, 
	"Sometimes": 3, 
	"Often": 4
	})
	
# plt.scatter(survey['q0007_0001'], survey['q0007_0002'], alpha = 0.1, marker = "o")
# plt.show()

rows_to_cluster = survey.dropna(subset = ["q0007_0001", "q0007_0002", "q0007_0003",
 "q0007_0004", "q0007_0005", "q0007_0006",
 "q0007_0007", "q0007_0008", "q0007_0009",
 "q0007_0010", "q0007_0011"])

classifier = KMeans(n_clusters = 2)
classifier.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004", "q0007_0005", "q0007_0008", "q0007_0009"]]) 
print(classifier.cluster_centers_) 
print(classifier.labels_)

cluster_zero_indices = []
cluster_one_indices = []

for i in range(len(classifier.labels_)): ##append indices of each class type (0, 1) to a list 
    if classifier.labels_[i] == 0: 
        cluster_zero_indices.append(i)
    if classifier.labels_[i] == 1: 
        cluster_one_indices.append(i)
#print(cluster_zero_indices)

cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]
print(cluster_zero_df.head())