import gc
import io
import math
import os
import time
from collections import deque
from multiprocessing import Pool

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


object_id_column = "augmentation_id"


def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def linear_assym(t, t0, A, s1, s2):
    """Helper, a linear fit of the decay"""
    q = A + (t - t0) / s1
    right_filter = t > t0
    q[right_filter] = (A - (t - t0) / s2)[right_filter]
    return q


def calc_features_for_passband(df_lights):
    """Calculate features for a single passband
    Args:
        df_lights: a DataFrame with light curves
    """
    x = df_lights["mjd"].values
    if len(x) <= 3:
        return 0, 0, 0, 0, 1000, 1000
    x_min = np.min(x)
    x = x - x_min
    # print('x_min', x_min)

    y = df_lights["flux"].values
    y = np.clip(y, 1, None)
    y = np.log(y)

    s = np.log(
        1 + df_lights["flux_err"].values / np.clip(df_lights["flux"].values - df_lights["flux_err"].values, 1e-7, None)
    )

    max_ind = np.argmax(y)
    lower_bounds = [x[max_ind] - 100, y[max_ind], 1, 1]
    upper_bounds = [x[max_ind] + 100, y[max_ind] + 2, 500, 2000]

    try:
        popt, pcov = curve_fit(
            linear_assym,
            x,
            y,
            p0=(x[max_ind], y[max_ind], 10, 300),
            sigma=s,
            method="trf",
            maxfev=3000,
            bounds=(lower_bounds, upper_bounds),
            absolute_sigma=True,
        )
    except RuntimeError:
        return 0, 0, 0, 0, 1000, 1000

    err = ((y - linear_assym(x, *popt)) ** 2 / s ** 2).mean()

    mu = (y / s ** 2).sum() / (1 / s ** 2).sum()
    # print('mu', mu)
    const_value = np.full(len(y), mu)
    err0 = ((y - const_value) ** 2 / s ** 2).mean()
    err0 = max(err0, 1e-5)
    err_relative = err / err0

    return [popt[0] + x_min, popt[1], popt[2], popt[3], err, err_relative]


columns_names = ["object_id"]
for passband in range(6):
    columns_names.append("supernova1_A_" + str(passband))
    columns_names.append("supernova1_s1_" + str(passband))
    columns_names.append("supernova1_s2_" + str(passband))
    columns_names.append("supernova1_err_abs_" + str(passband))
    columns_names.append("supernova1_err_rel_" + str(passband))
columns_names.append("supernova1_mjd_diff_0")
columns_names.append("supernova1_mjd_diff_1")
columns_names.append("supernova1_mjd_diff_2")
columns_names.append("supernova1_mjd_diff_4")
columns_names.append("supernova1_mjd_diff_5")


def calc_features_for_all_passbands(params):
    """Calculate features for all passbands"""
    object_data, object_id = params
    object_data.sort_values(by=["mjd"])
    features = [object_id]
    mjds = []
    for passband in range(6):
        pass_lights = object_data[object_data["passband"] == passband]
        pass_features = calc_features_for_passband(pass_lights)
        # print(pass_features)
        features.extend(pass_features[1:])
        mjds.append(pass_features[0])
    # print('mjds', mjds)
    features.append(mjds[0] - mjds[3])
    features.append(mjds[1] - mjds[3])
    features.append(mjds[2] - mjds[3])
    features.append(mjds[4] - mjds[3])
    features.append(mjds[5] - mjds[3])
    return features


def calc_features(df_lights):
    """Calculate features for a dataframe data
    Args:
        df_lights: a DataFrame with light curves
    """
    unique_ids = df_lights[object_id_column].unique()

    params = []
    for object_id in unique_ids:
        object_data = df_lights[df_lights[object_id_column] == object_id]
        params.append((object_data, object_id))
    # features_for_all_objects = pool.map(calc_features_for_all_passbands, params)
    features_for_all_objects = [calc_features_for_all_passbands(param) for param in params]
    full_test = pd.DataFrame(data=features_for_all_objects, columns=columns_names)

    return full_test


def calc_and_save_features(params):
    """Calculate and save features for the dataframe part"""
    start = time.time()
    input_file, output_file = params
    print("start calculate:", input_file, output_file)
    input_df = pd.read_csv(input_file)
    calculated_features = calc_features(input_df)
    calculated_features.to_csv(output_file, index=False)
    print("finish calculate:", input_file, output_file, (time.time() - start) / 60, "minutes")



def main():
    """
    Use mutlithreading to calculate features for all light curves
    """
    pool = Pool(processes=24)

    print("start processing test set")

    params = []
    for chunk_index in range(30):
        input_file = "augmented_" + str(chunk_index) + ".csv"
        output_file = "augmented_" + str(chunk_index) + "_supernova1_features.csv"
        params.append((input_file, output_file))
    pool.map(calc_and_save_features, params)

    pool.close()

    all_features = None
    for chunk_index in range(30):
        output_file = "augmented_" + str(chunk_index) + "_supernova1_features.csv"
        chunk_features = pd.read_csv(output_file)
        if all_features is None:
            all_features = chunk_features
        else:
            all_features = pd.concat((all_features, chunk_features))
    all_features.to_csv("augmented2_supernova1_features.csv", index=False)


if __name__ == "__main__":
    main()
    