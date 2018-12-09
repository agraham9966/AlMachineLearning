labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
  #True Positives
  if labels[i] == 1 and guesses[i] == 1:
    true_positives += 1
  #True Negatives
  if labels[i] == 0 and guesses[i] == 0:
    true_negatives += 1
  #False Positives
  if labels[i] == 0 and guesses[i] == 1:
    false_positives += 1
  #False Negatives
  if labels[i] == 1 and guesses[i] == 0:
    false_negatives += 1
    
accuracy = (true_positives + true_negatives) / len(guesses)
print(accuracy)

recall = true_positives / (true_positives + false_negatives)
print(recall)

precision = true_positives / (true_positives + false_positives)
print(precision)

f_1 = 2*(precision*recall)/ (precision + recall) ##calculates harmonic mean rather than arithmetic mean 
print(f_1)
##a low f_1 close to 0 will indicate a poor classifier. closer to 1 a good one. 


# Classifying a single point can result in a true positive (truth = 1, guess = 1), a true negative (truth = 0, guess = 0), 
#a false positive (truth = 0, guess = 1), or a false negative (truth = 1, guess = 0).
# Accuracy measures how many classifications your algorithm got correct out of every classification it made.
# Recall measures the percentage of the relevant items your classifier was able to successfully find.
# Precision measures the percentage of items your classifier found that were actually relevant.
# Precision and recall are tied to each other. As one goes up, the other will go down.
# F1 score is a combination of precision and recall.
# F1 score will be low if either precision or recall is low.
# The decision to use precision, recall, or F1 score ultimately comes down to the context of your classification. 
#Maybe you don't care if your classifier has a lot of false positives. If that's the case, precision doesn't matter as much.

# As long as you have an understanding of what question you're trying to answer, you should be able to determine which 
#statistic is most relevant to you.