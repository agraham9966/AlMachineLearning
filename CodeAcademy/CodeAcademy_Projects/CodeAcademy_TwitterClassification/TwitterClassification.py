import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale

all_tweets = pd.read_json("random_tweets.json", lines=True)

#print(len(all_tweets))
#print(all_tweets.columns)
#print(all_tweets.loc[0]['user']['screen_name'])
#print(all_tweets.loc[0]['text'])

#print(np.median(all_tweets['retweet_count'])) ##median of 13
#print(all_tweets['retweet_count'].value_counts())

all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] >= 13, 1, 0)
#print(all_tweets['is_viral'].value_counts())

all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

labels = all_tweets['is_viral']
data = all_tweets[['tweet_length', 'followers_count', 'friends_count']]
scaled_data = scale(data, axis=0) # scales columns and not the rows with axis = 0
