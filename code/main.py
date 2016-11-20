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
import PlottingModule as plotter
from CrossValidationModule import crossValidate
from config import *
import numpy as np

class Experiments(object):
	def __init__(self, classifierType, percentiles_range, kernels_range, gamma_range, C_range, skipFilter=True):
		if not skipFilter:
			fltr = Filter(['id'], 'loss')
			fltr.generateCleanFile(DATA_FOLDER, RAW_TRAINING_FILE, CLEAN_TRAINING_FILE)
			fltr.generateCleanFile(DATA_FOLDER, RAW_TESTING_FILE, CLEAN_TESTING_FILE)
		dataSource = DataReader(DATA_FOLDER, CLEAN_TRAINING_FILE, CLEAN_TESTING_FILE)
		self.classifierType = classifierType
		if (self.classifierType == 'Binary-SVM'):
			self.X_train, self.y_train, self.X_cv, self.y_cv = dataSource.getTrainData(1000, 50)
		elif(self.classifierType == 'One-Class-SVM'):
			self.X_train, self.y_train, self.X_cv, self.y_cv = dataSource.getTrainData(1000, 0)
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

	def runGridSearch(self):
		CVresults = list()
		for percentile in self.percentiles_range:
			for kernel in self.kernels_range:
				for C in self.C_range:
					if kernel != 'linear':
						for gamma in self.gamma_range:
							fselector = feature_selection.SelectPercentile(self.featureSelector,\
								percentile)
							X_train = fselector.fit_transform(self.X_train, self.y_train)							
							X_cv = fselector.transform(self.X_cv)
							print("Processing Kernel: {}, percentile: {}, C: {}, gamma: {}".format(\
								kernel, percentile, C, gamma))
							if self.classifierType == 'Binary-SVM':
								accuracy = crossValidate(self.model(kernel=kernel, C=C, gamma=gamma),\
									X_train, self.y_train, X_cv, self.y_cv)
							elif self.classifierType == 'One-Class-SVM':
								accuracy = crossValidate(self.model(kernel=kernel, nu=C, gamma=gamma),\
									X_train, self.y_train, X_cv, self.y_cv)
							CVresults.append([percentile, kernel, C, gamma, accuracy])
							print("Kernel: {}, percentile: {}, C: {}, gamma: {}, accuracy: {}".format(\
								kernel, percentile, C, gamma, accuracy))
					else:
						gamma = 'auto'
						fselector = feature_selection.SelectPercentile(self.featureSelector,\
								percentile)
						X_train = fselector.fit_transform(self.X_train, self.y_train)
						X_cv = fselector.transform(self.X_cv)
						print("Processing Kernel: {}, percentile: {}, C: {}, gamma: {}".format(\
								kernel, percentile, C, gamma))
						if self.classifierType == 'Binary-SVM':
							accuracy = crossValidate(self.model(kernel=kernel, C=C, gamma=gamma),\
								X_train, self.y_train, X_cv, self.y_cv)
						elif self.classifierType == 'One-Class-SVM':
							accuracy = crossValidate(self.model(kernel=kernel, nu=C, gamma=gamma),\
								X_train, self.y_train, X_cv, self.y_cv)
						CVresults.append([percentile, kernel, C, gamma, accuracy])
						print("Kernel: {}, percentile: {}, C: {}, gamma: {}, accuracy: {}".format(\
								kernel, percentile, C, gamma, accuracy))

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
	gamma_range = np.logspace(-9, 3, 13)
	#binarySVMexps = Experiments('Binary-SVM', percentiles_range, kernels_range, gamma_range, C_range)
	#binarySVMexps.runGridSearch()
	nu_range = np.arange(0.01, 1.01, 0.01)
	oneClassSVMexps = Experiments('One-Class-SVM', percentiles_range, kernels_range, gamma_range, nu_range)
	oneClassSVMexps.runGridSearch()

	




