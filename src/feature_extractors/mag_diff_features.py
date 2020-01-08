# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 18:00:48 2018

Calculate differences of absolute magnitudes

"""
import gc
import io
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import curve_fit

from src.config import DATA_DIR, FEAT_DIR

sys.path.append('/home/user/plasticc/kaggle-plasticc/src')


def calc_ratios_mag(params):
    """Calculate and clip values of magnitudes differences in diff passbands"""
    
    object_id = params     
    mag = mags[mags.object_id == object_id]
    fit = fits[fits.object_id == object_id]
    
    # calculate observed colors ugrizy
    g_r = mag.det_magn_min_1.values[0] - mag.det_magn_min_2.values[0]
    if np.abs(g_r) > 10: 
        g_r = 0  
    g_i = mag.det_magn_min_1.values[0] - mag.det_magn_min_3.values[0]       
    if np.abs(g_i) > 10: 
        g_i = 0
    r_z = mag.det_magn_min_2.values[0] - mag.det_magn_min_4.values[0]
    if np.abs(r_z) > 10: 
        r_z = 0
    r_i = mag.det_magn_min_2.values[0] - mag.det_magn_min_3.values[0]
    if np.abs(r_i) > 10: 
        r_i = 0
    g_z = mag.det_magn_min_1.values[0] - mag.det_magn_min_4.values[0]
    if np.abs(g_z) > 10: 
        g_z = 0
    u_r = mag.det_magn_min_0.values[0] - mag.det_magn_min_2.values[0]
    if np.abs(u_r) > 10: 
        u_r = 0
    i_z = mag.det_magn_min_3.values[0] - mag.det_magn_min_4.values[0]     
    if np.abs(i_z) > 10: 
        i_z = 0
       
    # fitted observed colors ugrizy
    fg_r = fit.magn_fit_1.values[0] - fit.magn_fit_2.values[0] 
    fg_i = fit.magn_fit_1.values[0] - fit.magn_fit_3.values[0]         
    fr_z = fit.magn_fit_2.values[0] - fit.magn_fit_4.values[0]
    fr_i = fit.magn_fit_2.values[0] - fit.magn_fit_3.values[0]
    fg_z = fit.magn_fit_1.values[0] - fit.magn_fit_4.values[0]
    fu_r = fit.magn_fit_0.values[0] - fit.magn_fit_2.values[0]
    fi_z = fit.magn_fit_3.values[0] - fit.magn_fit_4.values[0]
    if np.abs(fg_r) > 10: 
        fg_r = 0
    if np.abs(fg_i) > 10: 
        fg_i = 0
    if np.abs(fr_z) > 10: 
        fr_z = 0
    if np.abs(fr_i) > 10: 
        fr_i = 0 
    if np.abs(fg_z) > 10: 
        fg_z = 0
    if np.abs(fu_r) > 10: 
        fu_r = 0 
    if np.abs(fi_z) > 10: 
        fi_z = 0    
    print('fit colors: ', fg_r, fg_i, fr_z, fr_i, fg_z, fu_r, fi_z)
        
    return object_id, g_r, g_i, r_z, r_i, g_z, u_r, i_z, \
           fg_r, fg_i, fr_z, fr_i, fg_z, fu_r, fi_z


def test_calc_ratios_mag():
"""Helper, tests feature extractor on a few objects"""    
    objects = meta[meta.target == 90].object_id.values
    print(objects)

    for obj in objects[10:15]:
        column_names = ['object_id', 'g_r', 'g_i', 'r_z', 'r_i', 'g_z', 'u_r', 'i_z', \
                'fg_r', 'fg_i', 'fr_z', 'fr_i', 'fg_z', 'fu_r', 'fi_z']
        features = calc_ratios_mag(obj)
        print(features)

    
def track_job(job, update_interval=30):
    task = pool._cache[job._job]
    while task._number_left > 0:
        print("Tasks remaining = {0}".format(task._number_left * task._chunksize))
        time.sleep(update_interval)
 
    
def calc_mag_ratios_features(objects):
    """
    Create dataframe of features
    for train objects
    """
    params = [object_id for object_id in objects]
    print('start running tasks...')
    res = pool.map_async(calc_ratios_mag, params, chunksize=400)
    track_job(res)
    features_for_all_objects = res.get()
    
    column_names = ['object_id', 'g_r', 'g_i', 'r_z', 'r_i', 'g_z', 'u_r', 'i_z', \
            'fg_r', 'fg_i', 'fr_z', 'fr_i', 'fg_z', 'fu_r', 'fi_z']
    
    return pd.DataFrame(data=features_for_all_objects, columns=column_names)


def main():
    #load magnitude features
    mags = pd.read_csv(FEAT_DIR + 'augmented_features/augmented_det_mag_features.csv')
    print('training_set_tanya_mag loaded')

    fits = pd.read_csv(FEAT_DIR + 'augmented_features/aug_from_fit_features.csv')
    print('training set from fits loaded')

    meta = pd.read_csv(FEAT_DIR + 'augmented_features/augmented_selected_features_v6.csv')
    print('training set meta')

    mags.fillna(0, inplace = True)
    print(mags.head())

    # extract ratios in parallel
    pool = Pool(processes = 3)
    print('start processing test set')
    objects = fits.object_id.values
    print(objects, len(objects))
    m_feat_df = calc_mag_ratios_features(objects)
    m_feat_df.to_csv('aug_color_features.csv', index=False)
    pool.close()


if __name__ == '__main__':
    main()
    
