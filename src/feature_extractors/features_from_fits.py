     # -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:48:21 2018

Calculate features from curve fits

"""

import sys
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgb
from src.config import DATA_DIR, SAVE_DIR
from torch.autograd import grad

sys.path.append('/home/user/plasticc/kaggle-plasticc/src')

sns.set_style('whitegrid')

MODEL_NAME = 'lgbm_61_ac_selected_'


def fit_function(x, fm, tau_rise, tau_fall, t0, f0):    
    """
    fitting function
    """
    
    fall = np.exp(-(x-t0)/tau_fall)
    rise = 1 + np.exp(-(x-t0)/tau_rise)
            
    return fm*fall/rise + f0


def tmax(tau_rise, tau_fall, t0):
    
    tmax = t0 + tau_rise*np.log(tau_fall/tau_rise - 1)
    if tmax == np.nan:
        tmax = t0
        
    return tmax


def m15_m10(fit, meta, object_id, mean, std, passband):
    """m15 and m-10 parameters """
    
    t0_n = 't0_%d'%passband
    fm_n = 'fm_%d'%passband
    
    # get curves params
    distmod  = meta[meta.object_id == object_id].distmod.values[0]
    tau_rise = fit[fit.object_id == object_id].tau_rise.values[0]
    tau_fall = fit[fit.object_id == object_id].tau_fall.values[0]
    t0       = fit[fit.object_id == object_id][t0_n].values[0]
    f0       = fit[fit.object_id == object_id].f0.values[0]
    fm       = fit[fit.object_id == object_id][fm_n].values[0]
        
    # rescale fm and f0 back
    fm = mean + std * fm
    f0 = mean + std * f0
        
    #print('fit params: tau_rise, tau_fall, t0, f0, fm', tau_rise, tau_fall, t0, f0, fm)
        
    # time of max from curve fit
    t_max = tmax(tau_rise, tau_fall, t0)
    #print('t_max', t_max)
    
    Mag_fm = -2.5*np.log10(fm) - distmod # you need at tmax better
    fmax = fit_function(t_max, fm, tau_rise, tau_fall, t0, f0)
    Mmax = -2.5*np.log10(fmax) - distmod
    #print('fmax, Mmax:', fmax, Mmax)
    
    # rescale 15 and 10 days to current scale
    t_15 = 15/(mjd_max- mjd_min) # rescale 15 days to the curve fit scale
    t_10 = 10/(mjd_max- mjd_min) # rescale 10 days to the curve fit scale
    
    f15 = fit_function(t_max + t_15, fm, tau_rise, tau_fall, t0, f0)
    M15 = -2.5*np.log10(f15) - distmod
    f10 = fit_function(t_max - t_10, fm, tau_rise, tau_fall, t0, f0)
    M10 = -2.5*np.log10(f10) - distmod
    #print('f15, f10 and M15, M10:', f15, f10, M15, M10)
    
    m15 = Mmax - M15    
    m10 = Mmax - M10    
    #print('m15, m10:', m15, m10)
    
    time = np.linspace(0, 1, 500) #/(mjd_max- mjd_min)
    t1 = time + t0
    t2 = t0 - time
    epsilon = fmax/50
    
    # peak width at half max
    t_fall50, t_fall20, t_fall80, t_rise50, t_rise20, t_rise80 = 0, 0, 0, 0, 0, 0
    for t in t1: 
        if (np.abs(fit_function(t, fm, tau_rise, tau_fall, t0, f0) - fmax/2))  < epsilon:
            t_fall50 = t            
        if (np.abs(fit_function(t, fm, tau_rise, tau_fall, t0, f0) - fmax/5))  < epsilon:
            t_fall20 = t
        if (np.abs(fit_function(t, fm, tau_rise, tau_fall, t0, f0) - 0.8*fmax))< epsilon:
            t_fall80 = t
            
    for t in t2:
        if (np.abs(fit_function(t, fm, tau_rise, tau_fall, t0, f0) - fmax/2))  < epsilon:
            t_rise50 = t            
        if (np.abs(fit_function(t, fm, tau_rise, tau_fall, t0, f0) - fmax/5))  < epsilon:
            t_rise20 = t
        if (np.abs(fit_function(t, fm, tau_rise, tau_fall, t0, f0) - 0.8*fmax))< epsilon:
            t_rise80 = t        
                   
    t_50 = t_fall50 - t_rise50    
    # peak width at 20% max             
    t_20 = t_fall20 - t_rise20    
    # peak width at 80% max            
    t_80 = t_fall80 - t_rise80
    #print('features:', Mmax, m15, m10, t_50, t_20, t_80)
    
    return fm, Mag_fm, Mmax, m15, m10, t_50, t_20, t_80
    
  
def m_features(train, fit, objects, passband = 2):
    """
    created dataframe of features
    for train objects
    """
       
    object_ids, fm_rescaled, mag_fm, m_max, m_15, m_10, t_50, t_20, t_80  = \
    [], [], [], [], [], [], [], [], []
    
    for obj in objects:
        
        mean, std = train[train['object_id'] == obj]['flux'].mean(), train[train['object_id'] == obj][
        'flux'].std()
        
        object_ids.append(obj)
        
        fm, Mfm, Mmax, m15, m10, t50, t20, t80 = m15_m10(fit, meta, obj, mean, std, passband)
        
        fm_rescaled.append(fm)
        mag_fm.append(Mfm)
        m_max.append(Mmax)        
        m_15.append(m15)
        m_10.append(m10)
        t_50.append(t50)
        t_20.append(t20)
        t_80.append(t80)
    #create a dictionary                                
    m_feat = {}    
    
    m_feat['object_id'] = object_ids
    
    m_feat['fm_rescaled_%d' %passband] = fm_rescaled
    m_feat['magn_fm_%d' %passband] =  mag_fm
    m_feat['magn_fit_%d' %passband] = m_max
    m_feat['m_15_%d' %passband] = m_15
    m_feat['m_10_%d' %passband] = m_10
    m_feat['t_50_%d' %passband] = t_50
    m_feat['t_20_%d' %passband] = t_20
    m_feat['t_80_%d' %passband] = t_80
            
    print(m_feat)    
    
    return  m_feat   
    

def main():
    # curve fits
    fit  = pd.read_csv(DATA_DIR  + 'features/train_exp_ratio_fitted.csv')   
    train  = pd.read_csv(DATA_DIR  + 'training_set.csv') 
    meta = pd.read_csv(DATA_DIR  + 'training_set_metadata.csv') 
    meta.distmod.fillna(0, inplace = True)
    mjd_min = 59580.0338
    mjd_max = 60674.363

    objects = fit.object_id.values
    print(objects)
    object_id = objects[10]
    print(object_id)

    m, sd = train[train['object_id'] == object_id]['flux'].mean(), train[train['object_id'] == object_id][
        'flux'].std()

    fm, Mfm, Mmax, m15, m10, t_50, t_20, t_80 = m15_m10(fit, meta, object_id, m, sd, passband = 2)

    df_pb = train[(train['object_id'] == object_id) &(train['passband'] == 2)]


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot((df_pb['mjd'] - mjd_min) / (mjd_max - mjd_min), df_pb['flux'], 'ok')
    ax.set_xlabel('ModifiedJulian Time, days')

    # read params from fit
    tau_rise = fit[fit.object_id == object_id].tau_rise.values[0]
    tau_fall = fit[fit.object_id == object_id].tau_fall.values[0]
    t0       = fit[fit.object_id == object_id].t0_2.values[0]
    f0       = fit[fit.object_id == object_id].f0.values[0]
    fm       = fit[fit.object_id == object_id].fm_2.values[0]
    print('fit params: tau_rise, tau_fall, t0, f0, fm', tau_rise, tau_fall, t0, f0, fm)
        
    time = np.linspace(0, 1, 500)#/(mjd_max- mjd_min)

    f = fit_function(time, m + sd * fm, tau_rise, tau_fall, t0, m + sd * f0) 
    #plot curve fit
    ax.autoscale(False)
    ax.plot(time, f, 'b')
    # plt.xlabel('ModifiedJulian Time, days')
    plt.show()

    #Save train features
    #m_feat = m_features(train, fit, objects[10:11], passband = 2                    
    for pb in range(6):
        m_feat = m_features(train, fit, objects, passband = pb)  
        pd.DataFrame.from_dict(m_feat).to_csv('train_fit_features_%d.csv'%pb, index=False)


if __name__ == '__main__':
    main()