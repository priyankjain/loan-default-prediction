###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Module to perform cross validation
###############################################################################

import numpy as np

def crossValidate(classifier, X_train, y_train, X_test, y_test):
	'''Performs cross validation for the given classifier and training set
	Uses simple K fold cross validation
	Parameters: classifier (type: sklearn classifier object):\
	The classifier to use for cross validation, classifier is un-trained
	X (type:np.array) Features to be split into test and training set
	y (type:np.array) labels to be splt into test and training set
	Returns: (mean, std): Mean and standard deviation of accuracies
	over the K classifiers
	'''
	classifier.fit(X_train, y_train)
	y_predicted = classifier.predict(X_test)
	y_test = [x[0] for x in y_test]
	print(y_train)
	print(y_predicted)
	print(y_test)
	num_wrong_predictions = sum([1 if a != b else 0 for a, b in zip(y_test, y_predicted) ])
	total_predictions = len(y_test)
	accuracy = total_predictions-num_wrong_predictions
	accuracy /= total_predictions
	return accuracy