# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:25:09 2018

feets features

"""
import numpy as np
import pandas as pd
import os
import gc # garbage collector
import time
import tqdm
import logging
import multiprocessing
from scipy import stats
import matplotlib.pyplot as plt

import feets
import feets.preprocess

# pandas display options
pd.set_option('display.max_columns', None) 

# special packages for features

#import cesium.featurize as featurize

DATA_DIR = 'C:/Users/New/Documents/Challenges/LSST/'
FEAT_DIR = 'C:/Users/New/Documents/Challenges/LSST/features/'

#FEAT_DIR = '/home/rsa-key-20181130/lsst/features/'
#DATA_DIR = '/home/rsa-key-20181130/lsst/'
"""
calculate new FEETS features, save them in numpy array
"""

def feets_features_flux(df, feature_list, object_id): 
    """
    extract features for total flux
    
        df           - lightcurves databas
        feature_list - features to extract
        object_id    - object
    
    """
    
    lc_all = df[df.object_id == object_id][['mjd', 'flux', 'flux_err']]
    
    # all bands
    atime, amag, amag2, aerror, aerror2 = feets.preprocess.align(
        lc_all.mjd.values, lc_all.mjd.values, lc_all.flux.values, lc_all.flux.values, \
        lc_all.flux_err.values, lc_all.flux_err.values)
    # light curve
    lc = [lc_all.mjd.values, lc_all.flux.values, lc_all.flux_err.values,
          lc_all.flux.values, atime, amag, amag2,
          aerror, aerror2]
    
    fs = feets.FeatureSpace(only = feature_list)
    features, values = fs.extract(*lc)
    
    return features, values


def feets_features_band(df, feature_list, object_id, passband = 0, passband2 = 2): 
    """
    extract features per band
    
    """
    
    lc_b = df[(df.object_id == object_id)&(df.passband == passband)][['mjd', 'flux', 'flux_err']]
    lc_r = df[(df.object_id == object_id)&(df.passband == passband2)][['mjd', 'flux', 'flux_err']]
    
    # synchronize the data, blue-red
    atime, amag, amag2, aerror, aerror2 = feets.preprocess.align(
        lc_b.mjd.values, lc_r.mjd.values, lc_b.flux.values, lc_r.flux.values, \
        lc_b.flux_err.values, lc_r.flux_err.values)
    # light curve
    lc = [lc_b.mjd.values, lc_b.flux.values, lc_b.flux_err.values,
          lc_r.flux.values, atime, amag, amag2,
          aerror, aerror2]
    
    fs = feets.FeatureSpace(only = feature_list)
    features, values = fs.extract(*lc)
    
    return features, values

feature_list = [
 'Amplitude',
 'Autocor_length', 
 'Beyond1Std', 
 'CAR_sigma', 'CAR_tau', 'CAR_mean', 
 'Con', 
 'Eta_e',
 'MedianAbsDev',
 'MedianBRP',
 'PairSlopeTrend',
 'PercentAmplitude',
 'PeriodLS',
 'Period_fit',
 'Psi_CS',
 'Psi_eta', 
 'Q31',
 'Rcs',
 'SlottedA_length',
 'StetsonK',
 'StetsonK_AC',
 'StructureFunction_index_21',
 'StructureFunction_index_31',
 'StructureFunction_index_32'
 ]


"""
Load Data

"""

train = pd.read_csv(DATA_DIR + 'training_set.csv')
meta_train = pd.read_csv(DATA_DIR + 'training_set_metadata.csv')
objects = meta_train.object_id.values  # numpy array of objects in train
print('objects', objects, objects.shape)

"""
Calculate features from total flux for all objects

"""
"""
object_id = objects[0] 
features, values = feets_features_band(train, feature_list, object_id, passband = 0, passband2 = 2)

for object_id in objects[1:]:
    features_, feat_values_ = feets_features_band(train, feature_list, object_id, passband = 0, passband2 = 2)
    values = np.column_stack([values, feat_values_]) #numpy array of features values

features0 = [feat + '_0' for feat in features]
     
print(values, values.shape, values[:, 0])
feets_df = pd.DataFrame(data=values.T, index=None, columns=features0)
feets_df['object_id'] = objects
print(feets_df.head())
feets_df.to_csv(FEAT_DIR + 'feets_train_0.csv', header=True, index=False)
"""

object_id = objects[0] 
features, values = feets_features_flux(train, feature_list, objects[0])

for object_id in objects[1:3]:
    features_, feat_values_ = feets_features_flux(train, feature_list, object_id)
    values = np.column_stack([values, feat_values_]) #numpy array of features values
    #feat_df[object_id] = feat_values_

print(values, values.shape, values[:, 0])
feets_df = pd.DataFrame(data=values.T, index=None, columns=features)
feets_df['object_id'] = objects
print(feets_df.head())
feets_df.to_csv(FEAT_DIR + 'feets_train.csv', header=True, index=False)


"""
#print('skew', lc_all.flux.skew(), 'mean', lc_all.flux.mean())