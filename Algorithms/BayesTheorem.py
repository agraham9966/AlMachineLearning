##bayes theorem 

#conditional probability 
##Conditional probability is the probability that two events happen. It's easiest to calculate conditional probability when the two events are independent.
##Bayes - takes prior knowledge into consideration before computing statistics, probability


P(A&B) = P(A) * P(B) ## probability that both events are occuring, if they are independent of each other (have no effect on eachother)
##OR, probability that both A and B will happen 

##rolling 2 6's with double dice 
P(6&6) = P(6) * P(6) = 1/6 * 1/6 = 1/36

prob that patient had disease and correctly diagnosed 
P_disease_and_correct = (1-0.99) * (1/100000)

# In statistics, if we have two events (A and B), we write 
# the probability that event A will happen, given that event 
# B already happened as P(A|B). In our example, we want to find
# P(rare disease | positive result). In other words, we want to 
# find the probability that the patient has the disease given the 
# test came back positive.

We can calculate P(A|B) using Bayes' Theorem, which states:

P(A|B) = (P(B|A) * P(A)) / P(B)

P(A|B) = the probability that event A will happen given that B already happened 

A Naive Bayes classifier is a supervised machine learning algorithm that leverages 
Bayes' Theorem to make predictions and classifications. Recall Bayes' Theorem:
P(A|B) = (P(B|A) * P(A)) / P(B)
# This equation is finding the probability of A given B. This can be turned into a
 # classifier if we replace B with a data point and A with a class. For example, 
 # let's say we're trying to classify an email as either spam or not spam. We could 
 # calculate P(spam | email) and P(not spam | email). Whichever probability is higher
 # will be the classifier's prediction. Naive Bayes classifiers are often used for text 
 # classification.

# So why is this a supervised machine learning algorithm? In order to compute the probabilities
 # used in Bayes' theorem, we need previous data points. For example, in the spam example, we'll 
 # need to compute P(spam). This can be found by looking at a tagged dataset of emails and finding
 # the ratio of spam to non-spam emails.
 
 from reviews import neg_list, pos_list
from sklearn.feature_extraction.text import CountVectorizer

review = "This crib was amazing"

counter = CountVectorizer()
counter.fit(neg_list + pos_list)
print(counter.vocabulary_)

review_counts = counter.transform([review])
print(review_counts.toarray())

training_counts = counter.transform(neg_list + pos_list)

from reviews import counter, training_counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review = "This crib was goddam amazing, wonderful and completely satisfying."
review_counts = counter.transform([review])

classifier = MultinomialNB()
training_labels = [0] * 1000 + [1] * 1000
classifier.fit(training_counts, training_labels)
print(classifier.predict(review_counts))
print(classifier.predict_proba(review_counts))