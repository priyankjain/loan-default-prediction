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
			self.X_train, self.y_train, self.X_cv, self.y_cv = self.dataSource.getTrainData(500, 50)
		elif self.classifierType == 'One-Class-SVM':
			self.X_train, self.y_train, self.X_cv, self.y_cv = self.dataSource.getTrainData(500, 0)
		standardNormalizer = preprocessing.StandardScaler()
		self.X_train = standardNormalizer.fit_transform(self.X_train)
		self.X_cv = standardNormalizer.transform(self.X_cv)
		mi_featureSelector = feature_selection.mutual_info_classif
		self.normalizer = standardNormalizer
		self.featureSelector = mi_featureSelector
		self.percentiles_range = percentiles_range
		self.kernels_range = kernels_range
		self.gamma_range = gamma_range
		self.C_range = C_range
		if self.classifierType == 'Binary-SVM':
			self.model = svm.SVC
		elif self.classifierType == 'One-Class-SVM':
			self.model = svm.OneClassSVM
		elif self.classifierType == 'Gradient-Boosting':
			self.model = ensemble.GradientBoostingClassifier

	def runGridSearch(self):
		CVresults = list()
		self.best_roc_auc = None
		self.bestTuple = None
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
							print("Processing Kernel: {}, percentile: {}, C: {}, gamma: {}".format(\
								kernel, percentile, C, gamma))
							if self.classifierType == 'Binary-SVM':
								classifier, accuracy, roc_auc = crossValidate(self.model(kernel=kernel, C=C, \
									probability=True, gamma=gamma),\
									X_train, self.y_train, X_cv, self.y_cv, self.classifierType)
							elif self.classifierType == 'One-Class-SVM':
								classifier, accuracy,roc_auc = crossValidate(self.model(kernel=kernel, nu=C, \
									probability=True, gamma=gamma),\
									X_train, self.y_train, X_cv, self.y_cv, self.classifierType)
							elif self.classifierType == 'Gradient-Boosting':
								classifier, accuracy, roc_auc = crossValidate(self.model(n_estimators = kernel,\
									max_depth = gamma, learning_rate=C), \
									X_train, self.y_train, X_cv, self.y_cv, self.classifierType)
							CVresults.append([percentile, kernel, C, gamma, accuracy, roc_auc])
							print("Kernel: {}, percentile: {}, C: {}, gamma: {}, accuracy: {}, roc_auc: {}".format(\
								kernel, percentile, C, gamma, accuracy, roc_auc))
							if self.best_roc_auc is None or roc_auc > self.best_roc_auc:
								self.bestTuple = (percentile, kernel, C, gamma, accuracy, roc_auc)
								self.best_roc_auc = roc_auc
								self.bestClassifier = classifier
								self.bestfselector = fselector
					else:
						gamma = 'auto'
						fselector = feature_selection.SelectPercentile(self.featureSelector,\
								percentile)
						X_train = fselector.fit_transform(self.X_train, self.y_train)
						X_cv = fselector.transform(self.X_cv)
						print("Processing Kernel: {}, percentile: {}, C: {}, gamma: {}".format(\
								kernel, percentile, C, gamma))
						if self.classifierType == 'Binary-SVM':
							classifier, accuracy, roc_auc = crossValidate(self.model(kernel=kernel, C=C, \
								probability=True, gamma=gamma),\
								X_train, self.y_train, X_cv, self.y_cv, self.classifierType)
						elif self.classifierType == 'One-Class-SVM':
							classifier, accuracy, roc_auc = crossValidate(self.model(kernel=kernel, nu=C, \
								gamma=gamma),\
								X_train, self.y_train, X_cv, self.y_cv, self.classifierType)
						CVresults.append([percentile, kernel, C, gamma, accuracy, roc_auc])
						print("Kernel: {}, percentile: {}, C: {}, gamma: {}, accuracy: {}, roc_auc: {}".format(\
								kernel, percentile, C, gamma, accuracy, roc_auc))
						if self.best_roc_auc is None or roc_auc > self.best_roc_auc:
								self.bestTuple = (percentile, kernel, C, gamma, accuracy, roc_auc)
								self.best_roc_auc = roc_auc
								self.bestClassifier = classifier
								self.bestfselector = fselector
		with open('GridSearch-{}-Results.csv'.format(self.classifierType), 'w') as fout:
			writer = csv.writer(fout)
			writer.writerows(CVresults)

	def runTests(self):
		X_test, y_test = self.dataSource.getTestData(5000)
		print(X_test)
		print(y_test)
		X_test = self.normalizer.transform(X_test)
		X_test = self.bestfselector.transform(X_test)
		y_predicted = self.bestClassifier.predict(X_test)
		y_probabilities = None
		if self.classifierType == 'Binary-SVM' or self.classifierType == 'Gradient-Boosting':
			y_probabilities = self.bestClassifier.predict_proba(X_test)
			y_probabilities = y_probabilities[:,1]
		elif classifierType == 'One-Class-SVM':
			y_probabilities = self.bestClassifier.decision_function(X_test)
		#y_test = [x[0] for x in y_test]
		fpr, tpr, thresholds = roc_curve(y_test, y_probabilities,\
			pos_label = 1)
		roc_auc = auc(fpr, tpr)
		plotter.plotAndSaveLineChart('ROC Curve for {}'.format(self.classifierType), \
			fpr, tpr, 'False Positive Rate', 'True Positive Rate')
		num_wrong_predictions = sum([1 if a != b else 0 for a, b in zip(y_test, y_predicted) ])
		total_predictions = len(y_test)
		accuracy = total_predictions-num_wrong_predictions
		accuracy /= total_predictions

	def runFeatureSelection(self, percentilesh, hyperParameter):
		if self.classifierType == 'Binary-SVM':
			means, stds = fselector.getAccuraciesForPercentiles(percentiles=percentiles,\
			normalizer=self.normalizer, featureSelector=self.featureSelector,\
			model=self.model(C=hyperParameter), X=self.X, y=self.y)
			plotter.plotAndSaveErrorBar('Binary SVC Accuracy vs Percentile',\
			 percentiles, means, stds, 'Percentiles', 'Accuracy')
		elif self.classifierType == 'One-Class-SVM':
			means, stds = fselector.getAccuraciesForPercentiles(percentiles=percentiles,\
			normalizer=self.normalizer, featureSelector=self.featureSelector,\
			model=self.model(nu=hyperParameter), X=self.X, y=self.y)
			plotter.plotAndSaveErrorBar('One Class SVM Accuracy vs Percentile',\
			 percentiles, means, stds, 'Percentiles', 'Accuracy')

	def runParameterTuning(self, hyperParameterList, percentile):
		if self.classifierType == 'Binary-SVM':
			means, stds = tuner.getAccuraciesForCs(Clist=hyperParameterList, model=self.model, percentile=percentile,
				normalizer=self.normalizer, featureSelector=self.featureSelector,\
				X=self.X, y=self.y)
			plotter.plotAndSaveErrorBar('SVC Accuracy vs C',\
			 hyperParameterList, means, stds, 'C', 'Accuracy')
		elif self.classifierType == 'One-Class-SVM':
			means, stds = tuner.getAccuraciesForNus(nulist=hyperParameterList, model=self.model, percentile=percentile,
				normalizer=self.normalizer, featureSelector=self.featureSelector,\
				X=self.X, y=self.y)
			plotter.plotAndSaveErrorBar('One Class SVM Accuracy vs nu',\
			 hyperParameterList, means, stds, 'nu', 'Accuracy')
if __name__ == '__main__':
	os.chdir(HOME_FOLDER)
	C_range = np.logspace(-2, 1.5, 8)
	percentiles_range = (10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100)
	kernels_range = ['linear', 'poly', 'rbf', 'sigmoid']
	gamma_range = np.logspace(-9, 0, 10)
	#C_range = np.logspace(-2, 1.5, 8)
	#gamma_range = np.logspace(-9, -1, 9)
	#percentiles_range = [10]
	#kernels_range = ['linear']	
	#binarySVMexps = Experiments('Binary-SVM', percentiles_range, kernels_range, gamma_range, C_range)
	#binarySVMexps.runGridSearch()
	nu_range = np.arange(0.01, 1.00, 0.1)
	#oneClassSVMexps = Experiments('One-Class-SVM', percentiles_range, kernels_range, gamma_range, nu_range)
	#oneClassSVMexps.runGridSearch()

	learning_range = np.arange(0.9, 1, 0.1)
	estimators_range = np.arange(100, 110, 10)
	depth_range = np.arange(1, 2, 1)
	gradientBoostingexps = Experiments('Gradient-Boosting', percentiles_range, estimators_range, \
		depth_range, learning_range)
	gradientBoostingexps.runGridSearch()
	gradientBoostingexps.runTests()