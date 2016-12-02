###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Actually runs the classification pipeline
###############################################################################
from sklearn.pipeline import Pipeline
from CrossValidationModule import crossValidate

def classify(normalizer, featureSelector, model, percentile, X, y):
	'''Calculates accuracy by running the pipeling for top K features
	Parameters: normalizer: Normalizer to use from sklearn.preprocessing
	featureSelector: Feature selector to use from sklearn.feature_selection
	model: The model to use for classification
	percentile (float): Percentile of features to select
	X (np.array): Samples
	y (np.array): labels
	K (int): K for K-fold cross-validation
	Returns: (mean, std): Accuracy obtained by running the pipeline against the specified parameters
	and using K-fold cross-validation
	'''
	clf = Pipeline([('featureSelector', featureSelector), ('normalizer', normalizer) , ('model', model)])
	clf.set_params(featureSelector__percentile=percentile)
	return crossValidate(clf, X, y)