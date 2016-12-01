###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: The main script which runs all the experiments
###############################################################################
import os
import config
from sklearn import feature_selection
from FilterModule import Filter
from DataReaderModule import DataReader
from sklearn import preprocessing
from sklearn import svm
from sklearn import ensemble
import PlottingModule as plotter
from CrossValidationModule import crossValidate
from config import *
import numpy as np
import csv
from sklearn.metrics import roc_curve, auc

class Experiments(object):
	def __init__(self, classifierType, percentiles_range, kernels_range, gamma_range, C_range, skipFilter=True):
		if not skipFilter:
			fltr = Filter(['id'], 'loss')
			fltr.generateCleanFile(DATA_FOLDER, RAW_TRAINING_FILE, CLEAN_TRAINING_FILE)
			fltr.generateCleanFile(DATA_FOLDER, RAW_TESTING_FILE, CLEAN_TESTING_FILE)
		self.dataSource = DataReader(DATA_FOLDER, CLEAN_TRAINING_FILE, CLEAN_TESTING_FILE)
		self.classifierType = classifierType
		if self.classifierType == 'Binary-SVM' or self.classifierType == 'Gradient-Boosting':
			self.X_train, self.y_train = self.dataSource.getTrainData(5000, 50)
		elif self.classifierType == 'One-Class-SVM':
			self.X_train, self.y_train = self.dataSource.getTrainData(5000, 0)
		standardNormalizer = preprocessing.StandardScaler()
		self.X_train = standardNormalizer.fit_transform(self.X_train)
		mi_featureSelector = feature_selection.mutual_info_classif
		self.normalizer = standardNormalizer
		self.featureSelector = mi_featureSelector
		self.percentiles_range = percentiles_range
		self.kernels_range = kernels_range
		self.gamma_range = gamma_range
		self.C_range = C_range
		if self.classifierType == 'Binary-SVM':
			self.model = svm.SVC
			self.defaultKernel = 'rbf'
			self.defaultGamma = 'auto'
			self.defaultC = 1
		elif self.classifierType == 'One-Class-SVM':
			self.model = svm.OneClassSVM
			self.defaultKernel = 'rbf'
			self.defaultGamma = 'auto'
			self.defaultC = 0.5
		elif self.classifierType == 'Gradient-Boosting':
			self.model = ensemble.GradientBoostingClassifier
			self.defaultKernel = 100
			self.defaultGamma = 3
			self.defaultC = 0.1
		self.X_test, self.y_test = self.dataSource.getTestData(1000)
		self.X_test = self.normalizer.transform(self.X_test)
		self.K = 5
		self.bestPercentile = None
		self.bestKernel = None
		self.bestGamma = None
		self.bestC = None

	def trainAndTest(self, kernel, percentile, C, gamma):
		print("Processing Kernel: {}, percentile: {}, C: {}, gamma: {}".format(\
								kernel, percentile, C, gamma))
		fselector = feature_selection.SelectPercentile(self.featureSelector,\
								percentile)
		X_train = fselector.fit_transform(self.X_train, self.y_train)
		if self.classifierType == 'Binary-SVM':
			mean, std = crossValidate(self.model(kernel=kernel, C=C, \
				gamma=gamma),\
				X_train, self.y_train, self.K)
		elif self.classifierType == 'One-Class-SVM':
			mean, std = crossValidate(self.model(kernel=kernel, nu=C, \
				gamma=gamma),\
				X_train, self.y_train)
		elif self.classifierType == 'Gradient-Boosting':
			mean, std = crossValidate(self.model(n_estimators = kernel,\
				max_depth = gamma, learning_rate=C), \
				X_train, self.y_train)
		print("Kernel: {}, percentile: {}, C: {}, gamma: {}, mean-accuracy: {}, std-accuracy: {}".format(\
								kernel, percentile, C, gamma, mean, std))
		return mean, std

	def selectBestPercentile(self, kernel=None, C=None, gamma=None):
		means = list()
		stds = list()
		if kernel is None:
			kernel = self.defaultKernel
		if C is None:
			C = self.defaultC
		if gamma is None:
			gamma = self.defaultGamma
		bestMean = None
		for percentile in self.percentiles_range:
			mean, std = self.trainAndTest(kernel, percentile, \
				C, gamma)
			mean = round(mean, 2)
			means.append(mean)
			stds.append(std)
			if bestMean is None or mean > bestMean:
				bestMean = mean
				self.bestPercentile = percentile
		plotter.plotAndSaveErrorBar('Percentile validation curve for {}'.format(self.classifierType), \
			self.percentiles_range, means, stds, 'Percentile', 'Accuracy')

	def selectBestKernel(self, percentile, C=None, gamma=None):
		means = list()
		stds = list()
		if C is None:
			C = self.defaultC
		if gamma is None:
			gamma = self.defaultGamma
		bestMean = None
		for kernel in self.kernels_range:
			mean, std = self.trainAndTest(kernel, percentile, \
				C, gamma)	
			mean = round(mean, 2)
			means.append(mean)
			stds.append(std)
			if bestMean is None or mean > bestMean:
				bestMean = mean
				self.bestKernel = kernel
		plotter.plotAndSaveErrorBar('Kernel validation curve for {}'.format(self.classifierType), \
			range(0, len(self.kernels_range)), means, stds, 'Kernel ({})'.\
			format({k:v for k, v in zip(range(0, len(self.kernels_range)), self.kernels_range)}), 'Accuracy')

	def selectBestGamma(self, percentile, kernel, C=None):
		means = list()
		stds = list()
		if C is None:
			C = self.defaultC
		bestMean = None
		for gamma in self.gamma_range:
			mean, std = self.trainAndTest(kernel, percentile, \
				C, gamma)
			mean = round(mean, 2)
			means.append(mean)
			stds.append(std)
			if bestMean is None or mean > bestMean:
				bestMean = mean
				self.bestGamma = gamma
		plotter.plotAndSaveErrorBar('Gamma validation curve for {}'.format(self.classifierType), \
			self.gamma_range, means, stds, 'Gamma', 'Accuracy')

	def selectBestC(self, percentile, kernel, gamma):
		means = list()
		stds = list()
		bestMean = None
		for C in self.C_range:
			mean, std = self.trainAndTest(kernel, percentile, \
				C, gamma)
			mean = round(mean, 2)
			means.append(mean)
			stds.append(std)
			if bestMean is None or mean > bestMean:
				bestMean = mean
				self.bestC = C
		plotter.plotAndSaveErrorBar('Regularization parameter validation curve for {}'.format(self.classifierType),\
			self.C_range, means, stds, 'C', 'Accuracy')

	def runGreedySearch(self):
		self.selectBestPercentile()
		self.selectBestKernel(self.bestPercentile)
		self.selectBestGamma(self.bestPercentile, self.bestKernel)
		self.selectBestC(self.bestPercentile, self.bestKernel, self.bestGamma)


	def runGridSearch(self):
		CVresults = list()
		self.best_roc_auc = None
		self.bestParams = None
		self.bestClassifier = None
		self.bestfselector = None
		for kernel in self.kernels_range:
			for C in self.C_range:
				for percentile in self.percentiles_range:
					if kernel != 'linear':
						for gamma in self.gamma_range:
							fselector = feature_selection.SelectPercentile(self.featureSelector,\
								percentile)
							X_train = fselector.fit_transform(self.X_train, self.y_train)							
							X_cv = fselector.transform(self.X_cv)
							classifier, accuracy, roc_auc = self.trainAndTest(\
								kernel, percentile, C, gamma, X_train, self.y_train, \
								X_cv, self.y_cv)
							CVresults.append([percentile, kernel, C, gamma, accuracy, roc_auc])
							if self.best_roc_auc is None or roc_auc > self.best_roc_auc:
								self.bestParams = {'percentile': percentile, 'kernel': kernel, \
								'C': C, 'gamma': gamma, 'accuracy': accuracy, 'roc_auc': roc_auc}
								self.best_roc_auc = roc_auc
								self.bestClassifier = classifier
								self.bestfselector = fselector
					else:
						gamma = 'auto'
						fselector = feature_selection.SelectPercentile(self.featureSelector,\
								percentile)
						X_train = fselector.fit_transform(self.X_train, self.y_train)
						X_cv = fselector.transform(self.X_cv)
						classifier, accuracy, roc_auc = self.trainAndTest(\
								kernel, percentile, C, gamma, X_train, self.y_train, \
								X_cv, self.y_cv)
						CVresults.append([percentile, kernel, C, gamma, accuracy, roc_auc])						
						if self.best_roc_auc is None or roc_auc > self.best_roc_auc:
								self.bestParams = {'percentile': percentile, 'kernel': kernel, \
								'C': C, 'gamma': gamma, 'accuracy': accuracy, 'roc_auc': roc_auc}
								self.best_roc_auc = roc_auc
								self.bestClassifier = classifier
								self.bestfselector = fselector
		with open('GridSearch-{}-Results.csv'.format(self.classifierType), 'w') as fout:
			writer = csv.writer(fout)
			writer.writerows(CVresults)

	def runTests(self):		
		X_test = self.bestfselector.transform(self.X_test)
		y_predicted = self.bestClassifier.predict(X_test)
		y_probabilities = None
		if self.classifierType == 'Binary-SVM' or self.classifierType == 'Gradient-Boosting':
			y_probabilities = self.bestClassifier.predict_proba(X_test)
			y_probabilities = y_probabilities[:,1]
		elif classifierType == 'One-Class-SVM':
			y_probabilities = self.bestClassifier.decision_function(X_test)
		#y_test = [x[0] for x in y_test]
		fpr, tpr, thresholds = roc_curve(self.y_test, y_probabilities,\
			pos_label = 1)
		roc_auc = auc(fpr, tpr)
		plotter.plotAndSaveLineChart('ROC Curve for {}'.format(self.classifierType), \
			fpr, tpr, 'False Positive Rate', 'True Positive Rate')
		num_wrong_predictions = sum([1 if a != b else 0 for a, b in zip(self.y_test, y_predicted) ])
		total_predictions = len(self.y_test)
		accuracy = total_predictions-num_wrong_predictions
		accuracy /= total_predictions

	def plotCurves(self):
		percentile = self.bestParams['percentile']
		kernel = self.bestParams['kernel']
		C = self.bestParams['C']
		gamma = self.bestParams['gamma']
		fselector = feature_selection.SelectPercentile(self.featureSelector, percentile)
		X_train = fselector.fit_transform(self.X_train, self.y_train)
		X_test = fselector.transform(self.X_test)
		kernel_accuracies = list()
		for kernel_iter in self.kernels_range:
			print("Plotting curve for Kernel: {}".format(kernel_iter))			
			classifier, accuracy, roc_auc = self.trainAndTest(kernel_iter, percentile, C, gamma, \
				X_train, self.y_train, X_test, self.y_test)
			kernel_accuracies.append(accuracy)
		C_accuracies = list()
		for C_iter in self.C_range:
			print("Plotting curve for C: {}".format(C_iter))
			classifier, accuracy, roc_auc = self.trainAndTest(kernel, percentile, C_iter, gamma,\
				X_train, self.y_train, X_test, self.y_test)
			C_accuracies.append(accuracy)
		if kernel != 'linear':
			gamma_accuracies = list()
			for gamma_iter in self.gamma_range:
				print("Plotting curve for gamma: {}".format(gamma_iter))
				classifier, accuracy, roc_auc = self.trainAndTest(kernel, percentile, C, gamma_iter,\
				X_train, self.y_train, X_test, self.y_test)
				gamma_accuracies.append(accuracy)
		percentile_accuracies = list()
		for percentile_iter in percentiles_range:
			fselector = feature_selection.SelectPercentile(self.featureSelector, percentile_iter)
			X_train = fselector.fit_transform(self.X_train, self.y_train)							
			X_test = fselector.transform(self.X_test)
			print("Plotting curve for percentile: {}".format(percentile_iter))
			classifier, accuracy, roc_auc = self.trainAndTest(kernel, percentile_iter, C, gamma,\
				X_train, self.y_train, X_test, self.y_test)
			percentile_accuracies.append(accuracy)
		print(self.kernels_range)
		print(kernel_accuracies)
		plotter.plotAndSaveLineChart('Kernel validation curve for {}'.format(self.classifierType), \
			self.kernels_range, kernel_accuracies, 'Kernel', 'Accuracy')
		print(self.C_range)
		print(C_accuracies)
		plotter.plotAndSaveLineChart('C validation curve for {}'.format(self.classifierType), \
			self.C_range, C_accuracies, 'C', 'Accuracy')
		print(self.gamma_range)
		print(gamma_accuracies)
		plotter.plotAndSaveLineChart('Gamma validation curve for {}'.format(self.classifierType), \
			self.gamma_range, gamma_accuracies, 'C', 'Accuracy')
		print(self.percentiles_range)
		print(percentile_accuracies)
		plotter.plotAndSaveLineChart('Percentile validation curve for {}'.format(self.classifierType), \
			self.percentiles_range, percentile_accuracies, 'Percentile', 'Accuracy')
		trainSize_range = range(10, self.X_train.shape[0] + 1)
		training_accuracies = list()
		testing_accuracies = list()
		for trainSize in trainSize_range:
			fselector = feature_selection.SelectPercentile(self.featureSelector, percentile)			
			X_train = self.X_train[:trainSize]
			y_train = self.y_train[:trainSize]
			X_train = fselector.fit_transform(X_train, y_train)
			X_test = fselector.transform(self.X_test)
			classifier = None
			print("Plotting curve for {} training examples".format(trainSize))
			classifier, accuracy, roc_auc = self.trainAndTest(kernel, percentile, C, gamma,\
				X_train, y_train, X_test, self.y_test)
			y_predicted = classifier.predict(X_train)
			num_wrong_predictions = sum([1 if a != b else 0 for a, b in zip(y_train, y_predicted) ])
			total_predictions = len(y_train)
			training_accuracy = total_predictions-num_wrong_predictions
			training_accuracy /= total_predictions
			training_accuracies.append(training_accuracy)
			testing_accuracies.append(accuracy)
		print(trainSize_range)
		print(training_accuracies)
		print(testing_accuracies)
		plotter.drawBiasVarianceCurve('Bias-Variance-Curve for {}'.format(self.classifierType),\
			trainSize_range, training_accuracies, testing_accuracies, \
			'Number of training examples', 'Training Error', 'Testing Error')

if __name__ == '__main__':
	os.chdir(HOME_FOLDER)
	C_range = np.logspace(-2, 1.5, 8)
	percentiles_range = (10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100)
	kernels_range = ['linear', 'poly', 'rbf', 'sigmoid']
	gamma_range = np.logspace(-9, -1, 10)
	binarySVMexps = Experiments('Binary-SVM', percentiles_range, kernels_range, gamma_range, C_range)
	#binarySVMexps.runGridSearch()
	binarySVMexps.runGreedySearch()
	#binarySVMexps.runTests()
	#binarySVMexps.plotCurves()
	nu_range = np.arange(0.05, 1.00, 0.1)
	oneClassSVMexps = Experiments('One-Class-SVM', percentiles_range, kernels_range, gamma_range, nu_range)
	oneClassSVMexps.runGreedySearch()
	#oneClassSVMexps.runTests()
	#oneClassSVMexps.plotCurves()
	learning_range = np.arange(0.1, 1, 0.1)
	estimators_range = np.arange(100, 200, 10)
	depth_range = np.arange(1, 5, 1)
	gradientBoostingexps = Experiments('Gradient-Boosting', percentiles_range, estimators_range, \
		depth_range, learning_range)
	gradientBoostingexps.runGreedySearch()
	#gradientBoostingexps.runTests()
	#gradientBoostingexps.plotCurves()

