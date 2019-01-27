from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_iris 
import pandas as pd 
import numpy as np 

np.random.seed(0) 
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(df.head())
#adding a new col for the species name 
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) 
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

#view features 
features = df.columns[:4]

#convert species name in to digits 
y = pd.factorize(train['species'])[0]

clf = RandomForestClassifier(n_jobs=2, random_state=0) 
clf.fit(train[features], y) 