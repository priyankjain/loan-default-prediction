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
import FeatureSelectionModule as fselector
import PlottingModule as plotter
import ParameterTuningModule as tuner
from config import *

class Experiments(object):
	def __init__(self):
		fltr = Filter(['id'], 'loss')
		fltr.generateCleanFile(DATA_FOLDER, RAW_FILE, TRAINING_FILE)
		dataSource = DataReader(DATA_FOLDER, TRAINING_FILE)
		self.X, self.y = dataSource.getData(5000, 50)
		self.X = fselector.removeZeroVarianceFeatures(self.X)
		standardNormalizer = preprocessing.StandardScaler().fit(self.X)
		minMaxNormalizer = preprocessing.MinMaxScaler().fit(self.X)
		maxAbsNormalizer = preprocessing.MaxAbsScaler().fit(self.X)
		mi_featureSelector = feature_selection.SelectPercentile(feature_selection.f_classif)
		fscore_featureSelector = feature_selection.SelectPercentile(feature_selection.f_classif)
		self.normalizers = {'Standard': standardNormalizer, \
			'Min Max': minMaxNormalizer, 'Max Abs': maxAbsNormalizer}
		self.featureSelectors={'F score': fscore_featureSelector,'Mutual Information': mi_featureSelector}
		self.normalizer = self.normalizers['Standard']
		self.featureSelector = self.featureSelectors['Mutual Information']
		self.model = svm.SVC(C=1.0)
		self.K = 5

	def runFeatureSelection(self, percentiles):
		means, stds = fselector.getAccuraciesForPercentiles(percentiles=percentiles,\
			normalizer=self.normalizer, featureSelector=self.featureSelector,\
			model=self.model, K=self.K, X=self.X, y=self.y)
		plotter.plotAndSaveErrorBar('SVC Accuracy vs Percentile',\
		 percentiles, means, stds, 'Percentiles', 'Accuracy')

	def runParameterTuning(self, Clist, percentile):
		means, stds = tuner.getAccuraciesForCs(Clist=Clist, percentile=percentile,
			normalizer=self.normalizer, featureSelector=self.featureSelector,\
			K=self.K, X=self.X, y=self.y)
		plotter.plotAndSaveErrorBar('SVC Accuracy vs C',\
		 Clist, means, stds, 'C', 'Accuracy')

if __name__ == '__main__':
	os.chdir(HOME_FOLDER)
	Clist = list()
	C = 0.01
	count = 0
	while C<=30:
		Clist.append(C)
		count += 1
		C = C*3
		if count%2 == 0:
			C = C * 10/9
	percentiles = (10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100)
	exps = Experiments()
	exps.runFeatureSelection(percentiles)
	exps.runParameterTuning(Clist, 20)


	




