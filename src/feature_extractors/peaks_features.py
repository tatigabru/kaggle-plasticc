import gc
import io
import os
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.random.seed(42)

gc.enable()

# training_set = pd.read_csv('../Data/training_set.csv')
# training_set_metadata = pd.read_csv('../Data/training_set_metadata.csv')
columns_14 = [
    "class_6",
    "class_15",
    "class_16",
    "class_42",
    "class_52",
    "class_53",
    "class_62",
    "class_64",
    "class_65",
    "class_67",
    "class_88",
    "class_90",
    "class_92",
    "class_95",
]
columns_15 = [
    "class_6",
    "class_15",
    "class_16",
    "class_42",
    "class_52",
    "class_53",
    "class_62",
    "class_64",
    "class_65",
    "class_67",
    "class_88",
    "class_90",
    "class_92",
    "class_95",
    "class_99",
]
# print('total number of objects:', len(training_set_metadata['object_id']))


def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def find_root(x1, v1, x2, v2):
    return (x1 * v2 - x2 * v1) / (v2 - v1)


def calc_peak_width(df_lights, passband):
    p = df_lights[df_lights["passband"] == passband]
    fv = p["flux"].values
    fl = len(fv)
    if fl <= 1:
        return None
    max_flux = np.max(fv)
    if max_flux < 100:
        return None
    thr = max_flux / 2
    # print(passband, max_flux, thr)
    peaks = []
    ind = 0
    while ind < fl:
        while ind < fl and fv[ind] < thr:
            ind += 1
        start = ind
        while ind < fl and fv[ind] >= thr:
            ind += 1
        end = ind
        if end <= start:
            break
        if start == 0:
            # peak_start = p['mjd'].values[start]
            peak_start = df_lights["mjd"].values[0]
        else:
            peak_start = find_root(
                p["mjd"].values[start - 1], fv[start - 1] - thr, p["mjd"].values[start], fv[start] - thr
            )
        if end == fl:
            # peak_end = p['mjd'].values[end - 1]
            peak_end = df_lights["mjd"].values[-1]
        else:
            peak_end = find_root(p["mjd"].values[end - 1], fv[end - 1] - thr, p["mjd"].values[end], fv[end] - thr)
        peaks.append((peak_start, peak_end, p["mjd"].values[start], p["mjd"].values[end - 1]))
    # print(peaks)
    if len(peaks) != 1:
        return None
    return peaks[0]


def add_peak(features, new_peak):
    if new_peak is None:
        return features

    est_new_peak_start, est_new_peak_end, certain_new_peak_start, certain_new_peak_end = new_peak
    if features is None:
        return (1, est_new_peak_start, est_new_peak_end, certain_new_peak_start, certain_new_peak_end)

    # update
    n_peaks, est_peak_start, est_peak_end, certain_peak_start, certain_peak_end = features
    est_peak_start = max(est_peak_start, est_new_peak_start)
    est_peak_end = min(est_peak_end, est_new_peak_end)
    if est_peak_end <= est_peak_start:
        n_peaks = -100
    certain_peak_start = min(certain_peak_start, certain_new_peak_start)
    certain_peak_end = max(certain_peak_end, certain_new_peak_end)
    est_peak_start = min(est_peak_start, certain_peak_start)
    est_peak_end = max(est_peak_end, certain_peak_end)
    n_peaks += 1
    return (n_peaks, est_peak_start, est_peak_end, certain_peak_start, certain_peak_end)


def calc_features_one_object(df_lights):
    features = None
    features = add_peak(features, calc_peak_width(df_lights, 2))
    features = add_peak(features, calc_peak_width(df_lights, 3))
    features = add_peak(features, calc_peak_width(df_lights, 4))
    features = add_peak(features, calc_peak_width(df_lights, 5))
    return features


small_quality = 0.1
best_quality = 1.0


def fuzzy_greater(x, a, b):
    if x <= a:
        return small_quality
    if x >= b:
        return best_quality
    return (best_quality - small_quality) * (x - a) / (b - a) + small_quality


def calc_class6_quality(features):
    if features is None:
        return 0.0, 0.0, -1, -1
    n_peaks, est_peak_start, est_peak_end, certain_peak_start, certain_peak_end = features
    if n_peaks < 0:
        return 0.0, 0.0, -1, -1
    est_width = est_peak_end - est_peak_start
    certain_width = certain_peak_end - certain_peak_start
    quality = fuzzy_greater(n_peaks, 0, 4)
    quality *= fuzzy_greater(est_width, 15, 40)
    return quality, n_peaks, est_width, certain_width


def calc_features_for_df(df_lights):
    """Calculate and save features fot the dataframe data"""
    objects = df_lights["augmentation_id"].unique()
    print("len obj", len(objects))
    print(objects)
    features_for_all_objects = []
    for object_id in objects:
        # if index % 50 == 0 :
        #    print('index', index)
        object_data = df_lights[df_lights["augmentation_id"] == object_id]
        object_data.sort_values(by=["mjd"])
        features = calc_features_one_object(object_data)
        quality, n_peaks, est_width, certain_width = calc_class6_quality(features)
        features_for_all_objects.append([object_id, quality, n_peaks, est_width, certain_width])
    return pd.DataFrame(
        data=features_for_all_objects,
        columns=["object_id", "my_6_quality", "my_6_n_peaks", "my_6_est_width", "my_6_certain_width"],
    )


def calc_and_save_features(params):
    """Calculate and save features"""
    start = time.time()
    input_file, output_file = params
    print("start calculate:", input_file, output_file)
    input_df = pd.read_csv(input_file)
    calculated_features = calc_features_for_df(input_df)
    calculated_features.to_csv(output_file, index=False)
    print("finish calculate:", input_file, output_file, (time.time() - start) / 60, "minutes")


if __name__ == "__main__":
    pool = Pool(processes=24)

    print("start processing test set")

    params = []
    for chunk_index in range(30):
        input_file = "augmented_" + str(chunk_index) + ".csv"
        output_file = "augmented_" + str(chunk_index) + "_my6_features.csv"
        params.append((input_file, output_file))
    pool.map(calc_and_save_features, params)

    pool.close()

    all_features = None
    for chunk_index in range(30):
        output_file = "augmented_" + str(chunk_index) + "_my6_features.csv"
        chunk_features = pd.read_csv(output_file)
        if all_features is None:
            all_features = chunk_features
        else:
            all_features = pd.concat((all_features, chunk_features))
    all_features.to_csv("augmented2_my6_features.csv", index=False)
