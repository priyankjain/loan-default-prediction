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
	def __init__(self, dataFolder, dataFileName):
		'''Parameters: dataFile (string): Path to the clean data file
		'''
		self._dataFile = os.path.join(dataFolder, dataFileName)
		self._allData = None

	def getData(self, nSamples, percentNeg):
		'''Parameters: nSamples(int): Number of samples to be read in
		percentNeg(int): Percentage of samples to be negative
		Return: (X, y)
		X (np.array): Features of the samples
		y (np.array): labels of the samples
		'''
		nNeg = math.ceil(nSamples*percentNeg/100)
		nPos = nSamples - nNeg
		if self._allData is None:
			self._readData(nPos, nNeg)
			if self._rows < nSamples:
				raise ValueError('Requested {} samples from DataGenerator'.\
				format(nSamples) + 'but data only has {} samples'.format(self._rows))
			self.X = self._allData[:, 0:self._cols - 1]
			self.y = self._allData[:, - 1]
		return (self.X, self.y)

	def _readData(self, nPos, nNeg):
		'''Reads data from the actual file, with required number of 
		positive and negative samples. Better than using np.loadtxt
		which loads the entire file
		Parameters: nPos (int): Number of positive samples to be read in
		nNeg(int): Number of negative samples to be read in
		Return: None
		'''
		self._allData = list()
		with open(self._dataFile, 'r') as fin:
				reader = csv.reader(fin)
				count = 0
				posCount = 0
				negCount = 0
				for row in reader:
					if count == 0:
						count = 1
						continue
					if row[-1] == '0':
						posCount +=1
						if posCount <= nPos:
							self._allData.append([float(x) for x in row])
					elif row[-1] == '1':
						negCount += 1
						if negCount <= nNeg:
							self._allData.append([float(x) for x in row])
					if posCount > nPos and negCount > nNeg:
						break;
		self._allData = np.array(self._allData)
		np.random.shuffle(self._allData)
		(self._rows, self._cols) = self._allData.shape
		print('Read in {} samples from {}: {} positive and {} negative'
				.format(self._rows, self._dataFile, nPos, nNeg))