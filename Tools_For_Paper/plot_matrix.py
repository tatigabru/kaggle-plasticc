import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt

from collections import Counter
from functools import reduce
from sklearn.metrics import confusion_matrix
import itertools
from lightgbm import LGBMClassifier
import time
#import seaborn as sns

columns_14_short = ['6', '15', '16', '42', '52', '53', '62', '64', '65', '67', '88', '90', '92', '95']

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    # We change the fontsize of minor ticks label
    #plt.tick_params(axis='both', which='major', labelsize='large')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize='x-large',
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.show()

cnf_matrix = np.array([[ 1394,     0,    10,     0,     0,     0,     0,     2,    41,
            0,     0,     0,     0,     0],
       [    0,  4298,     0,   105,    40,     0,   129,    14,     0,
           49,    12,   205,     0,    26],
       [    7,     0,  7754,     0,     0,     0,     0,     0,    19,
            0,     0,     0,     2,     0],
       [    0,   837,     0,  4717,   545,     0,  1273,    69,     0,
          288,     5,  1169,     0,   291],
       [    0,    34,     0,   188,   392,     0,   150,    11,     0,
           59,     0,   359,     0,     7],
       [   20,     0,     0,     0,     0,   244,     0,     0,     0,
            0,     0,     0,     0,     0],
       [    0,    85,     0,   512,   231,     0,  2009,   116,     0,
          423,     0,   247,     0,    56],
       [    0,     1,     0,     2,     0,     0,    28,   954,     0,
            6,     0,    11,     0,     0],
       [   23,     0,    11,     0,     0,     0,     0,     0,  6995,
            0,     0,     0,     0,     0],
       [    0,    52,     0,    52,    84,     0,   177,    17,     0,
          887,     0,   204,     0,    13],
       [    0,    14,     0,     3,     0,     0,     0,     0,     0,
            0,  2602,    10,     0,     0],
       [    0,   960,     0,   844,  1211,     0,   496,    50,     0,
          864,     2, 11061,     0,   109],
       [    5,     0,    41,     0,     0,     0,     0,     0,     3,
            0,     0,     0,  1819,     0],
       [    0,    28,     0,    61,     5,     0,    25,     0,     0,
            8,    20,    13,     0,  1221]], dtype=int)

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
fig_szie = 10
fig = plt.figure(figsize=(fig_szie, fig_szie))
plot_confusion_matrix(cnf_matrix, classes=columns_14_short, normalize=True)
fig.savefig('confusion_matrix.png', pad_inches=0.1, bbox_inches='tight', dpi=600)