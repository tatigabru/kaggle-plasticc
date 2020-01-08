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


def gaussian(t, t0, A, s):
    """Helper, Gaussian function"""
    return A * np.exp(-((t - t0) ** 2) / (2 * s ** 2))


def calc_features_for_passband(df_lights):
    """Calculate features for a single passband
    Args:
        df_lights: a DataFrame with light curves
    """
    x = df_lights["mjd"].values
    if len(x) <= 1:
        return 0, 0, 0, 1000
    x = x - np.min(x)
    y = df_lights["flux"].values
    s = df_lights["flux_err"].values

    max_ind = np.argmax(y)
    ymax = max(y[max_ind], 0)
    lower_bounds = [x[max_ind] - 100, ymax, 0]
    upper_bounds = [x[max_ind] + 100, ymax * 3 + 0.5, 100]
    # print(lower_bounds)
    # print(upper_bounds)
    try:
        popt, pcov = curve_fit(
            gaussian,
            x,
            y,
            p0=(x[max_ind], ymax, 10),
            sigma=s,
            method="trf",
            maxfev=3000,
            bounds=(lower_bounds, upper_bounds),
            absolute_sigma=True,
        )
    except RuntimeError:
        return 0, 0, 0, 1000
    err = ((y - gaussian(x, *popt)) ** 2 / s ** 2).mean()
    A = popt[1]
    return A, -2.5 * math.log(A, 10), popt[2], err


columns_names = ["object_id"]
for passband in range(6):
    columns_names.append("gauss_A_" + str(passband))
    columns_names.append("gauss_A_log_" + str(passband))
    columns_names.append("gauss_s_" + str(passband))
    columns_names.append("gauss_err_" + str(passband))


def calc_features_for_all_passbands(params):
    """Calculate features for all passbands"""
    object_data, object_id = params
    object_data.sort_values(by=["mjd"])
    features = [object_id]
    for passband in range(6):
        pass_lights = object_data[object_data["passband"] == passband]
        features.extend(list(calc_features_for_passband(pass_lights)))
    return features


def calc_features(df_lights: pd.DataFrame) -> pd.DataFrame:
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
        output_file = "augmented_" + str(chunk_index) + "_gauss_features.csv"
        params.append((input_file, output_file))
    pool.map(calc_and_save_features, params)

    pool.close()

    all_features = None
    for chunk_index in range(30):
        output_file = "augmented_" + str(chunk_index) + "_gauss_features.csv"
        chunk_features = pd.read_csv(output_file)
        if all_features is None:
            all_features = chunk_features
        else:
            all_features = pd.concat((all_features, chunk_features))
    all_features.to_csv("augmented2_gauss_features.csv", index=False)


if __name__ == "__main__":
    main()
    
