###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Module to perform cross validation
###############################################################################

import numpy as np
import PlottingModule as plotter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

count = 0
def crossValidate(classifier, X_train, y_train, X_test, y_test, classifier_type):
	global count
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
	y_probabilities = None
	if classifier_type == 'Binary-SVM' or classifier_type == 'Gradient-Boosting':
		y_probabilities = classifier.predict_proba(X_test)
		y_probabilities = y_probabilities[:,1]
	elif classifier_type == 'One-Class-SVM':
		y_probabilities = classifier.decision_function(X_test)
	if any(isinstance(el, np.ndarray) for el in y_test):
		y_test = [x[0] for x in y_test]
	fpr, tpr, thresholds = roc_curve(y_test, y_probabilities,\
		pos_label = 1)
	roc_auc = auc(fpr, tpr)
	#plotter.plotAndSaveLineChart(str(count), \
	#	fpr, tpr, 'False Positive Rate', 'True Positive Rate')
	count += 1
	num_wrong_predictions = sum([1 if a != b else 0 for a, b in zip(y_test, y_predicted) ])
	total_predictions = len(y_test)
	accuracy = total_predictions-num_wrong_predictions
	accuracy /= total_predictions
	return classifier, accuracy, roc_auc