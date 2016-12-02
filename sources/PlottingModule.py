###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Module which takes care of all plotting needs
###############################################################################

import matplotlib.pyplot as plt
import numpy as np
from config import *
import os

def plotAndSaveErrorBar(title, x_data, y_means, y_stds, xlabel, ylabel):
	plt.errorbar(x_data, y_means, np.array(y_stds))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.axis('tight')
	plt.savefig(os.path.join(CHARTS_FOLDER, title))  
	plt.clf()
	plt.cla()
	plt.close()

def plotAndSaveLineChart(title, x_data, y_data, xlabel, ylabel):
	plt.plot(x_data, y_data, linewidth=2.0)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(os.path.join(CHARTS_FOLDER, title))
	plt.clf()
	plt.cla()
	plt.close()

def drawBiasVarianceCurve(title, x_data, y1_data, y2_data, xlabel, y1label, y2label):
	plt.figure()
	plt.title(title)
	plt.plot(x_data, y1_data, 'o-', color='r', label=y1label)
	plt.plot(x_data, y2_data, 'o-', color='g', label=y2label)
	plt.legend(loc='best')
	plt.savefig(os.path.join(CHARTS_FOLDER, title))
	plt.clf()
	plt.cla()
	plt.close()