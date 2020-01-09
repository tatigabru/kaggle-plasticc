import gc
import io
import math
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, SAVE_DIR

sys.path.append("/home/user/plasticc/kaggle-plasticc/src")


def fit_function(x, fm, tau_rise, tau_fall, t0, f0):
    """
    Double exponent fitting function (Bazin et.al.)
    """
    fall = np.exp(-(x - t0) / tau_fall)
    rise = 1 + np.exp(-(x - t0) / tau_rise)

    return fm * fall / rise + f0


def tmax(tau_rise, tau_fall, t0):
    """Helper, position of the max from the fit"""
    tmax = t0 + tau_rise * np.log(tau_fall / tau_rise - 1)
    if tmax == np.nan:
        tmax = t0

    return tmax


def calc_features_for_passband(df_fits, passband):
    """Caclulate m15 and m-10 parameters for a single passband"""
    t0_n = "t0_%d" % passband
    fm_n = "fm_%d" % passband

    mean = df_fits["flux_mean"].values[0]
    std = df_fits["flux_std"].values[0]

    # get curves params
    distmod = df_fits.distmod.values[0]
    tau_rise = df_fits.tau_rise.values[0]
    tau_fall = df_fits.tau_fall.values[0]
    t0 = df_fits[t0_n].values[0]
    f0 = df_fits.f0.values[0]
    fm = df_fits[fm_n].values[0]

    # rescale fm and f0 back
    fm = mean + std * fm
    f0 = mean + std * f0

    # time of max from curve fit
    t_max = tmax(tau_rise, tau_fall, t0)
    # print('t_max', t_max)

    Mag_fm = -2.5 * np.log10(fm) - distmod  # you need at tmax better
    fmax = fit_function(t_max, fm, tau_rise, tau_fall, t0, f0)
    Mmax = -2.5 * np.log10(fmax) - distmod
    # print('fmax, Mmax:', fmax, Mmax)

    # rescale 15 and 10 days to current scale
    t_15 = 15 / (mjd_max - mjd_min)  # rescale 15 days to the curve fit scale
    t_10 = 10 / (mjd_max - mjd_min)  # rescale 10 days to the curve fit scale

    f15 = fit_function(t_max + t_15, fm, tau_rise, tau_fall, t0, f0)
    M15 = -2.5 * np.log10(f15) - distmod
    f10 = fit_function(t_max - t_10, fm, tau_rise, tau_fall, t0, f0)
    M10 = -2.5 * np.log10(f10) - distmod
    # print('f15, f10 and M15, M10:', f15, f10, M15, M10)

    m15 = Mmax - M15
    m10 = Mmax - M10
    # print('features:', Mag_fm, Mmax, m15, m10)

    return Mag_fm, Mmax, m15, m10


columns_names = ["object_id"]
for passband in range(2, 3):
    columns_names.append("magn_fm_" + str(passband))
    columns_names.append("magn_fit_" + str(passband))
    columns_names.append("m_15_" + str(passband))
    columns_names.append("m_10_" + str(passband))


def calc_features_for_all_passbands(params):
    """Caclulate features for all passbands"""
    object_data, object_id = params
    features = [object_id]
    for passband in range(2, 3):
        features.extend(list(calc_features_for_passband(object_data, passband)))
    return features


def calc_features(df_fits):
    unique_ids = df_fits["object_id"].unique()

    params = []
    for object_id in unique_ids:
        object_data = df_fits[df_fits["object_id"] == object_id]
        params.append((object_data, object_id))
    features_for_all_objects = pool.map(calc_features_for_all_passbands, params)
    # features_for_all_objects = [calc_features_for_all_passbands((df_lights, object_id)) for object_id in unique_ids]
    full_test = pd.DataFrame(data=features_for_all_objects, columns=columns_names)

    return full_test


def main():

    pool = Pool(processes=2)
    print("start processing test set")
    start = time.time()
    test_feat_file = "test_set_fits_features.csv"

    # curve fits
    fit = pd.read_csv(DATA_DIR + "features/curve_fits_old/test_exp_ratio_fitted.csv")
    stats = pd.read_csv(DATA_DIR + "features/test/test_set_standard_features.csv")
    meta = pd.read_csv(DATA_DIR + "test_set_metadata.csv")
    meta.distmod.fillna(0, inplace=True)

    fits = pd.merge(left=fit, right=stats, on="object_id")
    fits = pd.merge(left=fits, right=meta, on="object_id")
    print(fits.head())

    mjd_min = 59580.0338
    mjd_max = 60674.363

    with open(test_feat_file, "w", encoding="utf-8") as result_file:

        # pandas unique!
        unique_ids = fits["object_id"].unique()
        print(unique_ids)

        cur_obj_features = calc_features(fits)
        # print(cur_obj_features.dtypes)

        buffer = io.StringIO()
        cur_obj_features.to_csv(buffer, header=True, mode="w", index=False)

        buffer.seek(0)
        result_file.write(buffer.getvalue())

        del cur_obj_features
        gc.collect()
    pool.close()


if __name__ == "__main__":
    main()
