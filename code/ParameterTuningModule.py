###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: The main script, which runs the pipeline
###############################################################################
from ClassifierModule import classify
from sklearn import svm

def getAccuraciesForCs(Clist, **kwargs):
	'''Calculates accuracies by running the pipeline for different values of C
	Parameters: 
	C (array of float): C corresponding to the bias in the SVC
	Returns: array of (mean, std): Accuracy obtained by running the pipeline against
	increasing values of C as specified in Clist
	'''
	score_means = list()
	score_stds = list()
	for C in Clist:
		mean, std = classify(model=svm.SVC(C=C), **kwargs)
		score_means.append(mean)
		score_stds.append(std)
		print("C: {}, Accuracy: {}".format(C, mean))
	return score_means, score_stds