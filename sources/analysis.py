###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Initial analysis of data and feature
###############################################################################
import os
from config import *
from DataReaderModule import DataReader
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import feature_selection
from pandas.tools.plotting import scatter_matrix
import pandas as pd
import numpy as np

def visualizeUsingPCA():
		dataSource = DataReader(DATA_FOLDER, CLEAN_TRAINING_FILE, CLEAN_TESTING_FILE)
		X_train, y_train = dataSource.getTrainData(5000, 50)
		standardNormalizer = preprocessing.StandardScaler()
		X_train = standardNormalizer.fit_transform(X_train)		
		fig = plt.figure(1, figsize=(4,3))
		plt.clf()
		ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
		plt.cla()
		plt.title('PCA Visualization with 3 components')
		pca = decomposition.PCA(n_components=3)
		pca.fit(X_train)
		X = pca.transform(X_train)
		y = y_train		
		for c, s, name, label in [('b', 'o', 'No default', 1), ('r','^','Default', -1)]:
			ax.scatter(X[y==label,0], X[y==label,1], X[y==label,2], c=c, marker=s, cmap=plt.cm.spectral, label=label)
		scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle='None', c='b', marker = 'o')
		scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle='None', c='r', marker = '^')
		ax.legend([scatter1_proxy, scatter2_proxy], ['No default', 'Default'])
		ax.set_xlabel('X1')
		ax.set_ylabel('X2')
		ax.set_zlabel('X3')
		plt.savefig(os.path.join(CHARTS_FOLDER, '3D-PCA'))
		plt.clf()
		plt.cla()
		plt.close()
		fig, ax = plt.subplots()
		plt.title('Visualization using PCA\n with 2 components')
		pca = decomposition.PCA(n_components=2)
		pca.fit(X_train)
		X = pca.transform(X_train)
		y = y_train
		ax.scatter(X[y==1, 0], X[y==1, 1], color='b', marker='o', label='No default')
		ax.scatter(X[y==-1, 0], X[y==-1, 1], color='r', marker='^', label='Default')
		ax.legend(loc='best')
		plt.xlabel('X1')
		plt.ylabel('X2')
		plt.savefig(os.path.join(CHARTS_FOLDER, '2D-PCA'))
		plt.clf()
		plt.cla()
		plt.close()
		mi_featureSelector = feature_selection.mutual_info_classif
		fselector = feature_selection.SelectKBest(mi_featureSelector,\
								10)
		X_train = fselector.fit_transform(X_train, y_train)
		y_train = np.array([y_train])
		print(X_train.shape)
		print(y_train.shape)
		X_train = np.concatenate((X_train, y_train.T), axis=1)
		colNames = ['f'+str(i) for i in range(1, 11)]
		colNames.append('label')
		df = pd.DataFrame(X_train, columns=colNames)
		scatter_matrix(df, alpha=0.2, figsize=(6,6), diagonal='kde')
		plt.savefig(os.path.join(CHARTS_FOLDER, 'Scatter Matrix'))

if __name__ == '__main__':
	os.chdir(HOME_FOLDER)
	visualizeUsingPCA()