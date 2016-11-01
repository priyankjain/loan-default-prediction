###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Module which constructs clean data file from the raw file
###############################################################################
import csv
import os

class Filter(object):
	'''Class to filter out rows with missing values and required columns
	'''
	def __init__(self, dropColumns, labelColumn, NAString='NA'):
		'''Parameters: dropColumns: List of column names to be dropped from the raw file
		labelColumn:  Column name which corresponds to the label
		NAString: Placeholder for missing values, all rows with this
		placeholder would be dropped
		'''
		self.NAString = NAString
		self.dropColumns = dropColumns
		self.labelColumn = labelColumn

	def generateCleanFile(self, folder, inputFile, outputFile):
		'''Reads in the raw input file and generates the clean file which
		can be used for training
		Parameters: folder: string folder of both input and output files
		inputFile: string name of the input file to be read in 
		outputFile: string name of the output file to be generated
		'''
		inputFilePath = os.path.join(folder, inputFile)
		outputFilePath = os.path.join(folder, outputFile)
		fin = open(inputFilePath, 'r')
		fout = open(outputFilePath, 'w')
		reader = csv.reader(fin)
		writer = csv.writer(fout)
		header = next(reader)
		outputRow = list()
		for col in header:
			if col not in self.dropColumns:
				outputRow.append(col)
		writer.writerow(outputRow)
		for row in reader:
			if self.NAString in row:
				continue
			outputRow = list()
			for k, v in zip(header, row):
				if k in self.dropColumns:
					continue
				if k == self.labelColumn:
					if v == '0':
						v = 0
					else:
						v = 1
				outputRow.append(v)
			writer.writerow(outputRow)
		fin.close()
		fout.close()