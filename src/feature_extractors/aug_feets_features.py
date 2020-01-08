# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:25:09 2018

Calculate FEETS features, save them in numpy array

"""
import gc  # garbage collector
import logging
import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import stats

import feets
import feets.preprocess
from src.config import DATA_DIR, FEAT_DIR

sys.path.append('/home/user/plasticc/kaggle-plasticc/src')

# pandas display options
pd.set_option('display.max_columns', None) 


def feets_features_flux(df, feature_list, object_id): 
    """
    Extract features for total flux
    
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
    Extract features per band
    
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


def worker(part):
    """
    Compute FEETs features for augmented train, worker for multiprocessing
    
    """
    test = pd.read_csv(DATA_DIR + '/augmented_train/augmented_' + str(part) + '.csv')
    objects = test.augmentation_id.unique()    
    print(objects)
    
    feature_list = ['Amplitude',
                    'Autocor_length', 
                    'Beyond1Std', 
                    'CAR_sigma', 'CAR_tau', 'CAR_mean', 
                    'Con', 
                    'Eta_e',
                    'Gskew',
                    'MedianAbsDev',
                    'MedianBRP',
                    'PairSlopeTrend',
                    'PercentAmplitude',
                    'PercentDifferenceFluxPercentile',
                    'PeriodLS',
                    'Period_fit',
                    'Psi_CS',
                    'Psi_eta', 
                    'Q31',
                    'Rcs',
                    'Skew',
                    'SmallKurtosis',
                    'SlottedA_length',
                    'StetsonK',
                    'StetsonK_AC',              
                  ]         
            
    object_id = objects[0] 
    features, values = feets_features_flux(test, feature_list, objects[0])
    print(features, values)

    for object_id in objects[1:]:
        features_, feat_values_ = feets_features_flux(test, feature_list, object_id)
        values = np.column_stack([values, feat_values_]) #numpy array of features values
        np.savetxt(FEAT_DIR +'values' + str(part) + '.csv', values, delimiter=",")
        #print(object_id)        
    print(values, values.shape, values[:, 0])
    feets_df = pd.DataFrame(data=values.T, index=None, columns=features)
    feets_df['object_id'] = objects
    print(feets_df.head())
    feets_df.to_csv(FEAT_DIR + 'feets_selected_augmented_train' + str(part) + '.csv', header=True, index=False)
    
    print('part done:', part)     
    
    return feets_df


def main():
    start = time.time()

    parts = [i for i in range(24)]

    n_proc = 3 #<number of physical cores on your machine>
    use_parallel = True

    if use_parallel: 
        print('start worker')        
        pool = Pool(processes=n_proc, maxtasksperchild=1)
        feets_df_ = pool.map(worker, parts, chunksize=1)
        pool.close()                   
        

if __name__ == '__main__':
    main()
