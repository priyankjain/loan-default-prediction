###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Class which would select the appropriate number of features,
# using specified strategy
###############################################################################
from ClassifierModule import classify
from sklearn import feature_selection

def removeZeroVarianceFeatures(X):
	'''Function to remove features which use zero variance
	Parameters: X (np.array): Features for the dataset
	Return: X (np.array): Modified array of features
	with no zero-variance features
	'''
	varianceSelector = feature_selection.VarianceThreshold()
	X = varianceSelector.fit_transform(X)
	return X

def getAccuraciesForPercentiles(percentiles, **kwargs):
	'''Calculates accuracy by running the pipeling for top K features, once for 
	each element in the percentiles array
	Parameters: 
	percentile (array of float): Percentile of features to select
	Returns: array of (mean, std): Accuracy obtained by running the pipeline against the specified parameters
	and using K-fold cross-validation
	'''
	score_means = list()
	score_stds = list()
	for percentile in percentiles:
		mean, std = classify(percentile = percentile, **kwargs)
		score_means.append(mean)
		score_stds.append(std)
		print("Percentile: {}, Accuracy: {}".format(percentile, mean))
	return score_means, score_stds