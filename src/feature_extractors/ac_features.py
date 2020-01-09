# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:12:13 2018

Calculate Handcrafted Features

from autocorrelation

"""
from __future__ import division, print_function

import gc  # garbage collector
import logging
import multiprocessing
import os
import sys
import time as t
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import lightgbm as lgb
from src.config import DATA_DIR, FEAT_DIR

# pandas display options
pd.set_option("display.max_columns", None)

sys.path.append("/home/user/plasticc/kaggle-plasticc/src")


def autocorr(obj_df):
    x = obj_df.sort_values(by="mjd", ascending=True)["flux"]
    result = np.correlate(x, x, mode="full")

    return result[result.size // 2 :]


def linear_interp(mjd, flux):
    """Resample data with interpolation"""

    f = interp1d(mjd, flux)
    time = np.linspace(min(mjd), max(mjd), num=100, endpoint=True)
    fit = f(time)
    return time, fit


def linear_interp_noise(obj_df):
    """
    Helper,
    takes df per object per passband
    returns linear interpolated fit with noise
    """

    flux = obj_df["flux"].values
    flux_err_mean = obj_df["flux_err"].mean()
    mjd = obj_df["mjd"].values
    mjd = mjd - np.min(mjd)
    f = interp1d(mjd, flux)
    time = np.linspace(min(mjd), max(mjd), num=100, endpoint=True)
    noise = (np.random.rand(100) - 0.5) * flux_err_mean
    fit = f(time) + noise

    return time, fit


def func(x, a, b, c):
    """Helper, fit exponent"""
    return a * np.exp(-b * x) + c


def fit_loss(ac, fit):
    """Loss of the fit """
    r = ac - fit
    error = sum(r ** 2) / sum(ac ** 2)
    return error


def calc_ac_fit_band(df, objects, passband=0):
    """
    create dataframe of features
    for train objects
    """

    object_ids, ac_decay, ac_decay_err, ac_loss, ac_amp, ac_amp_err = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for obj in objects:

        obj_df = df[(df.object_id == obj) & (df.passband == passband)]

        time, fit = linear_interp_noise(obj_df)
        result = np.correlate(fit, fit, mode="full")
        ac = result[result.size // 2 :]
        ac_time = np.arange(0, len(ac))

        try:
            popt, pcov = curve_fit(func, ac_time, ac)
            # print(popt[0], popt[1])
            amp = popt[0]
            dec = popt[1]
            perr = np.sqrt(np.diag(pcov))
            # print(perr[0], perr[1])
            amp_err = perr[0]
            dec_err = perr[1]
            loss = fit_loss(ac, func(ac_time, *popt))
        # print(loss)
        except:
            amp = np.nan
            dec = np.nan
            amp_err = np.nan
            dec_err = np.nan
            loss = 1.0
            # print(loss)
            pass

        object_ids.append(obj)

        ac_decay.append(dec)
        ac_decay_err.append(dec_err)
        ac_loss.append(loss)
        ac_amp.append(amp)
        ac_amp_err.append(amp_err)

    # create a dictionary
    acfit = {}

    acfit["object_id"] = object_ids
    acfit["ac_decay_%d" % passband] = ac_decay
    acfit["ac_decay_err_%d" % passband] = ac_decay_err
    acfit["ac_loss_%d" % passband] = ac_loss
    acfit["ac_amp_%d" % passband] = ac_amp
    acfit["ac_amp_err_%d" % passband] = ac_amp_err

    return acfit


def calc_ac_fit(df, objects):
    """
    created dataframe of features
    for train objects
    """

    object_ids, ac_decay, ac_decay_err, ac_loss, ac_amp, ac_amp_err = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for obj in objects:

        obj_df = df[(df.object_id == obj)]

        time, fit = linear_interp_noise(obj_df)
        result = np.correlate(fit, fit, mode="full")
        ac = result[result.size // 2 :]
        ac_time = np.arange(0, len(ac))

        try:
            popt, pcov = curve_fit(func, ac_time, ac)
            # print(popt[0], popt[1])
            amp = popt[0]
            dec = popt[1]
            perr = np.sqrt(np.diag(pcov))
            # print(perr[0], perr[1])
            amp_err = perr[0]
            dec_err = perr[1]
            loss = fit_loss(ac, func(ac_time, *popt))
            # print(loss)
        except:
            amp = np.nan
            dec = np.nan
            amp_err = np.nan
            dec_err = np.nan
            loss = 1.0
            # print(loss)
            pass

        object_ids.append(obj)
        ac_decay.append(dec)
        ac_decay_err.append(dec_err)
        ac_loss.append(loss)
        ac_amp.append(amp)
        ac_amp_err.append(amp_err)

    # create a dictionary
    acfit = {}

    acfit["object_id"] = object_ids
    acfit["ac_decay"] = ac_decay
    acfit["ac_decay_err"] = ac_decay_err
    acfit["ac_loss"] = ac_loss
    acfit["ac_amp"] = ac_amp
    acfit["ac_amp_err"] = ac_amp_err

    return acfit


def worker(part):

    test = pd.read_csv(DATA_DIR + "/test_set/test_set_part" + str(part) + ".csv")
    objects = test.object_id.unique()
    print(objects)

    for pb in range(1, 6):
        acfits = calc_ac_fit_band(df, objects[:10], passband=pb)
        pd.DataFrame.from_dict(acfits).to_csv(
            FEAT_DIR + "ac_fits_" + str(pb) + "_test" + str(part) + ".csv",
            header=True,
            index=None,
        )
        print("passband done: ", pb, "part:", part)
    # ac_fits = calc_ac_fit(df, objects)
    # pd.DataFrame.from_dict(ac_fits).to_csv(FEAT_DIR + 'ac_fits_test' + str(part) + '.csv', header=True, index=None)
    # print('part done:', part)
    return acfits


def main():
    df = pd.read_csv(DATA_DIR + "training_set.csv")
    meta_train = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
    objects = meta_train.object_id.values  # numpy array of objects in train
    print("objects", objects, objects.shape)

    # for pb in range(1,5):
    #    ac_fits = ac_fit_band(df, objects, passband = pb)
    #    pd.DataFrame.from_dict(ac_fits).to_csv('ac_fits_%d.csv'%pb)

    tic = t.clock()
    ac_fits = calc_ac_fit_band(df, objects[:10], passband=pb)
    toc = t.clock()
    print(toc - tic)

    pd.DataFrame.from_dict(ac_fits).to_csv("ac_fits_agg.csv")

    # set params
    parts = [i for i in range(24)]
    n_proc = 2  # <number of physical cores on your machine>
    use_parallel = True

    if use_parallel:
        print("start worker")
        pool = Pool(processes=n_proc, maxtasksperchild=1)
        feets_df_ = pool.map(worker, parts, chunksize=1)
        pool.close()


if __name__ == "__main__":
    main()
