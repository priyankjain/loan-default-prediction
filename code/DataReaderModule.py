###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Module which would read in required percentage of positive
# and negative samples from the specified file path
###############################################################################
import csv
import numpy as np 
import math
import os 

class DataReader(object):
	'''Class to handle reading of specified number of positive and negative samples
	from the specified clean source file
	'''
	def __init__(self, dataFolder, dataFileName, testDataFileName):
		'''Parameters: dataFile (string): Path to the clean data file
		'''
		self._dataFile = os.path.join(dataFolder, dataFileName)
		self._allData = None
		self._testDataFile = os.path.join(dataFolder, testDataFileName)

	def getTrainData(self, nSamples, percentNeg):
		'''Parameters: nSamples(int): Number of samples to be read in
		percentNeg(int): Percentage of samples to be negative
		Return: (X, y)
		X (np.array): Features of the samples
		y (np.array): labels of the samples
		'''
		nNeg = math.ceil(nSamples*percentNeg/100)
		nPos = nSamples - nNeg
		nTest = nSamples*0.3
		if self._allData is None:
			self._readData(nPos, nNeg, nTest)
			if self._rows < nSamples:
				raise ValueError('Requested {} samples from DataGenerator'.\
				format(nSamples) + 'but data only has {} samples'.format(self._rows))
			self.X = self._allData[:, 0:self._cols - 1]
			self.y = self._allData[:, - 1]
		return (self.X, self.y, self.X_cv, self.y_cv)

	def _readData(self, nPos, nNeg, nCV):
		'''Reads data from the actual file, with required number of 
		positive and negative samples. Better than using np.loadtxt
		which loads the entire file
		Parameters: nPos (int): Number of positive samples to be read in
		nNeg(int): Number of negative samples to be read in
		Return: None
		'''
		self._allData = list()
		self.X_cv = list()
		self.y_cv = list()
		with open(self._dataFile, 'r') as fin:
				reader = csv.reader(fin)
				count = 0
				posCount = 0
				negCount = 0
				testCount = 0
				for row in reader:
					if count == 0:
						count = 1
						continue
					if row[-1] == '1':
						posCount +=1
						if posCount <= nPos:
							self._allData.append([float(x) for x in row])
					elif row[-1] == '-1':
						negCount += 1
						if negCount <= nNeg:
							self._allData.append([float(x) for x in row])
					if posCount > nPos and negCount > nNeg:
						if testCount >= nCV:
							break
						else:
							self.X_cv.append([float(x) for x in row[:-1]])
							if row[-1] == '-1':
								self.y_cv.append([-1])
							elif row[-1] == '1':
								self.y_cv.append([1])
							testCount += 1
		self._allData = np.array(self._allData)
		self.X_cv = np.array(self.X_cv)
		self.y_cv = np.array(self.y_cv)
		np.random.shuffle(self._allData)
		(self._rows, self._cols) = self._allData.shape
		print('Read in {} training samples ({} positive and {} negative) and {} cross-validation samples from {}: '
				.format(self._rows, nPos, nNeg, nCV, self._dataFile))

	def getTestData(self, nSamples):
		'''Reads data from the testing file
		Parameters: nSamples (int): Number of test samples to read in
		Return: X_test, y_test
		'''
		self.X_test = list()
		self.y_test = list()
		with open(self._testDataFile, 'r') as fin:
			reader = csv.reader(fin)
			count = 1
			for row in reader:
				count += 1
				record = [float(x) for x in row]
				self.X_test.append(record[:-1])
				self.y_test.append(record[-1])
				if count>nSamples:
					break
		self.X_test = np.array(self.X_test)
		self.y_test = np.array(self.y_test)
		print('Read in {} testing samples from {}'
				.format(nSamples, self._testDataFile))
		return self.X_test, self.y_test