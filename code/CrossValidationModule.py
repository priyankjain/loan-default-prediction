###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Module to perform cross validation
###############################################################################

import numpy as np

def crossValidate(classifier, X, y, K=5):
	'''Performs cross validation for the given classifier and training set
	Uses simple K fold cross validation
	Parameters: classifier (type: sklearn classifier object):\
	The classifier to use for cross validation, classifier is un-trained
	X (type:np.array) Features to be split into test and training set
	y (type:np.array) labels to be splt into test and training set
	Returns: (mean, std): Mean and standard deviation of accuracies
	over the K classifiers
	'''
	accuracies = list()
	for k in range(0, K):
		X_train = [x for i, x in enumerate(X) if i%K != k]
		X_test = [x for i, x in enumerate(X) if i%K == k]
		y_train = [z for i,z in enumerate(y) if i%K != k]
		y_test = [z for i,z in enumerate(y) if i%K == k]
		classifier.fit(X_train, y_train)
		y_predicted = classifier.predict(X_test)
		num_wrong_predictions = sum([abs(a-b) for a, b in zip(y_test, y_predicted)])
		total_predictions = len(y_test)
		accuracies.append((total_predictions-num_wrong_predictions)/total_predictions)
	accuracies = np.array(accuracies)
	return (accuracies.mean(), accuracies.std())