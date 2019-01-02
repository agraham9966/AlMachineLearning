import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

all_tweets = pd.read_json("random_tweets.json", lines=True)

#print(len(all_tweets))
print(all_tweets.columns)
#print(all_tweets.loc[0]['user']['screen_name'])
#print(all_tweets.loc[0]['text'])

#print(np.median(all_tweets['retweet_count'])) ##median of 13
#print(all_tweets['retweet_count'].value_counts())

all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] >= 13, 1, 0)
#print(all_tweets['is_viral'].value_counts())

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)# setting axis=1 creates new column rather than new row. 
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

labels = all_tweets['is_viral']
data = all_tweets[['followers_count', 'friends_count']]
scaled_data = scale(data, axis=0) # scales columns and not the rows with axis = 0
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 1)

##set up classifier
# clf = KNeighborsClassifier(n_neighbors = 5) 
# ##get predicted labels
# clf.fit(train_data, train_labels)
# y_prediction = clf.predict(test_data)
# print(y_prediction, clf.score(test_data, test_labels))


# scores = []
# k_list = range(1, 201)
# for k in k_list: 
    # clf = KNeighborsClassifier(n_neighbors = k) 
    # clf.fit(train_data, train_labels) 
    # scores.append(clf.score(test_data, test_labels))
	
# plt.plot(k_list, scores)
# plt.show()
