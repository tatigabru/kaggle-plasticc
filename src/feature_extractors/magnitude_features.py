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


# object_id_column = 'augmentation_id'
object_id_column = "object_id"


def is_sorted(a):
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def calc_features_for_passband(df_lights, distmod):
    """Calculates magnitude stats per passband for detected == 1 data only"""
    magn = -2.5 * np.log10(df_lights.loc[df_lights["detected"] == 1, "flux"]) - distmod
    magn.fillna(0, inplace=True)
    return magn.min(), magn.mean(), magn.skew(), magn.sum(), magn.std()


columns_names = ["object_id"]
for passband in range(6):
    columns_names.append("det_magn_min_" + str(passband))
    columns_names.append("det_magn_mean_" + str(passband))
    columns_names.append("det_magn_skew_" + str(passband))
    columns_names.append("det_magn_sum_" + str(passband))
    columns_names.append("det_magn_std_" + str(passband))
columns_names.append("det_magn_min")
columns_names.append("det_magn_mean")
columns_names.append("det_magn_skew")
columns_names.append("det_magn_sum")
columns_names.append("det_magn_std")


def calc_features_for_all_passbands(params):
    """Calculates magnitude stats for all passbands"""
    object_data, object_id, distmod = params
    # object_data.sort_values(by=['mjd'])
    features = [object_id]
    for passband in range(6):
        pass_lights = object_data[object_data["passband"] == passband]
        features.extend(list(calc_features_for_passband(pass_lights, distmod)))
    features.extend(list(calc_features_for_passband(object_data, distmod)))
    return features


def calc_features(df_lights, df_metadata):
    unique_ids = df_lights[object_id_column].unique()

    params = []
    for object_id in unique_ids:
        object_data = df_lights[df_lights[object_id_column] == object_id]
        distmod = df_metadata.loc[
            df_metadata[object_id_column] == object_id, "distmod"
        ].values[0]
        if math.isnan(distmod):
            distmod = 0
        params.append((object_data, object_id, distmod))
    # features_for_all_objects = pool.map(calc_features_for_all_passbands, params)
    features_for_all_objects = [
        calc_features_for_all_passbands(param) for param in params
    ]
    full_test = pd.DataFrame(data=features_for_all_objects, columns=columns_names)

    return full_test


def calc_and_save_features(params):
    start = time.time()
    input_file, metadata_file, output_file = params
    print("start calculate:", input_file, output_file)
    input_df = pd.read_csv(input_file)
    df_metadata = pd.read_csv(metadata_file)
    # print(df_metadata.describe())
    calculated_features = calc_features(input_df, df_metadata)
    calculated_features.to_csv(output_file, index=False)
    print(
        "finish calculate:",
        input_file,
        output_file,
        (time.time() - start) / 60,
        "minutes",
    )


def calc_augmented_train_features():
    """Calculates features for augmented train, in parallel"""
    pool = Pool(processes=24)

    print("start processing train set")
    n_chunks = 10

    params = []
    for chunk_index in range(n_chunks):
        input_file = "augmented_" + str(chunk_index) + ".csv"
        metadata_file = "meta_part_" + str(chunk_index) + ".csv"
        output_file = "augmented_" + str(chunk_index) + "_det_mag_features.csv"
        params.append((input_file, metadata_file, output_file))
    pool.map(calc_and_save_features, params)
    pool.close()

    output_file = "augmented_" + str(0) + "_det_mag_features.csv"
    all_features = pd.read_csv(output_file)
    for chunk_index in range(1, n_chunks):
        output_file = "augmented_" + str(chunk_index) + "_det_mag_features.csv"
        chunk_features = pd.read_csv(output_file)
        all_features = pd.concat((all_features, chunk_features))
    all_features.to_csv("augmented_det_mag_features.csv", index=False)


def calc_test_features():
    """Calculates features for augmented train, in parallel"""
    pool = Pool(processes=24)

    print("start processing test set")
    n_chunks = 455
    params = []
    for chunk_index in range(n_chunks):
        input_file = "../Data/test_set_chunk_" + str(chunk_index) + ".csv"
        metadata_file = "../Data/test_set_metadata.csv"
        output_file = "test_set_chunk_" + str(chunk_index) + "_det_mag_features.csv"
        if not os.path.exists(output_file):
            params.append((input_file, metadata_file, output_file))
    pool.map(calc_and_save_features, params)

    pool.close()

    output_file = "test_set_chunk_" + str(0) + "_det_mag_features.csv"
    all_features = pd.read_csv(output_file)
    for chunk_index in range(1, n_chunks):
        output_file = "test_set_chunk_" + str(chunk_index) + "_det_mag_features.csv"
        chunk_features = pd.read_csv(output_file)
        all_features = pd.concat((all_features, chunk_features))
    all_features.to_csv("test_set_mag_features.csv", index=False)


if __name__ == "__main__":
    # calc_and_save_features(('augmented_0.csv', 'meta_part_0.csv', 'augmented_mag_features.csv'))
    calc_augmented_train_features()
    # calc_test_features()
