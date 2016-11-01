###############################################################################
# Author: Priyank Jain (@priyankjain)
# Description: Module which takes care of all plotting needs
###############################################################################

import matplotlib.pyplot as plt
import numpy as np

def plotAndSaveErrorBar(title, x_data, y_means, y_stds, xlabel, ylabel):
	plt.errorbar(x_data, y_means, np.array(y_stds))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.axis('tight')
	plt.savefig(title)  
	plt.clf()
	plt.cla()
	plt.close()