import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt

from collections import Counter
from functools import reduce
from sklearn.metrics import confusion_matrix
import itertools
from lightgbm import LGBMClassifier
import time

# import seaborn as sns

import eli5
from eli5.sklearn import PermutationImportance

# https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend
def reset_state():
    np.random.seed(42)
    import random

    random.seed(3)
    os.environ["PYTHONHASHSEED"] = "0"


reset_state()
gc.enable()

training_set_metadata = pd.read_csv("../Data/training_set_metadata.csv")
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

training_set_selected = pd.read_csv(
    "../Features_New_Aug_Calc/augmented_basic_features_v2.csv"
)
print("training set loaded")
print("total number of train objects:", len(training_set_selected))

training_set_car = pd.read_csv("../Features_New_Aug_Calc/augmented_car_features_v2.csv")
print("training_set_car loaded")
assert len(training_set_car) == len(training_set_selected)

training_set_fits = pd.read_csv(
    "../Features_New_Aug_Calc/augmented_fits_features_v2.csv"
)
print("training_set_fits loaded")
assert len(training_set_fits) == len(training_set_selected)

training_set_cesium = pd.read_csv(
    "../Features_New_Aug_Calc/augmented_cesium_features_v2.csv"
)
print("augmented_cesium_features_v2 loaded")
assert len(training_set_cesium) == len(training_set_selected)

training_set_tanya_mag = pd.read_csv(
    "../Features_New_Aug_Calc/augmented2_det_mag_features.csv"
)
assert len(training_set_tanya_mag) == len(training_set_selected)
print("training_set_tanya_mag loaded")

training_set_my_6 = pd.read_csv(
    "../Features_New_Aug_Calc/augmented_my6_features_v2.csv"
)
assert len(training_set_my_6) == len(training_set_selected)
print("training set my6 loaded")

training_set_gauss = pd.read_csv(
    "../Features_New_Aug_Calc/augmented_gauss_features_v2.csv"
)
assert len(training_set_gauss) == len(training_set_selected)
print("training set gauss loaded")

training_set_supernova1 = pd.read_csv(
    "../Features_New_Aug_Calc/augmented_supernova1_features_v2.csv"
)
assert len(training_set_supernova1) == len(training_set_selected)
print("training set supernova1 loaded")

training_set_from_fits = pd.read_csv(
    "../Features_New_Aug_Calc/aug_from_fit_features_v2.csv"
)
assert len(training_set_from_fits) == len(training_set_selected)
print("training_set_from_fits loaded")

training_set_colors = pd.read_csv(
    "../Features_colors/augmented_color_features_v2_clipped.csv"
)
assert len(training_set_colors) == len(training_set_selected)
print("augmented_color_features_v2_clipped loaded")

training_set_periods = pd.read_csv("../Features_New_Aug_Calc/augemented_periods_v2.csv")
assert len(training_set_periods) == len(training_set_selected)
print("augemented_periods_v2 loaded")

training_set_metadata = pd.read_csv("../Features_New_Aug_Calc/aug_meta_v2.csv")
training_set_metadata["object_id"] = training_set_metadata["augmentation_id"]
training_set_metadata["aug_fold"] = training_set_metadata["augmentation_id"] % 100
assert len(training_set_metadata) == len(training_set_selected)
print("training set metadata loaded")


# best features for non-ddf
used_columns = [
    "object_id",
    "target",
    "det_mjd_diff",
    "hostgal_photoz",
    "det_magn_min",
    "det_magn_mean",
    "cvec1",
    "fm_0",
    "flux_err_min",
    "exp_ratio_fitting_loss",
    "det_magn_std",
    "gauss_err_5",
    "flux_by_flux_ratio_sq_skew",
    "gauss_s_2",
    "cvec2",
    "cvec5",
    "tau_rise",
    "fm_1",
    "cvec3",
    "fm_5",
    "tau_fall",
    "gauss_s_3",
    "__median_absolute_deviation___5_",
    "__median_absolute_deviation___2_",
    "CAR_tau",
    "fm_2",
    "gauss_err_1",
    "det_magn_mean_2",
    "det_magn_mean_4",
    "det_magn_mean_3",
    "cvec0",
    "detected_mean",
    "gauss_err_2",
    "my_6_certain_width",
    "__skew___1_",
    "__skew___2_",
    "flux_diff2",
    "gauss_s_4",
    "gauss_s_5",
    "my_6_est_width",
    "__stetson_k___3_",
    "cvec4",
    "fm_4",
    "gauss_s_1",
    "__percent_close_to_median___2_",
    "supernova1_s2_3",
    "__qso_log_chi2_qsonu___0_",
    "fm_3",
    "det_magn_min_1",
    "det_magn_min_0",
    "det_magn_min_2",
    "det_magn_min_4",
    "det_magn_min_5",
    "det_magn_min_3",
    "__freq_varrat___1_",
    "gauss_s_0",
    "gauss_err_3",
    "__freq_varrat___4_",
    "flux_skew",
    "gauss_err_4",
    "magn_fit_4",
    "magn_fit_2",
    "magn_fit_1",
    "magn_fm_5",
    "magn_fm_4",
    "magn_fm_1",
    "gauss_err_0",
    "time_score",
    "det_magn_mean_5",
    "g_r",
    "g_i",
    "r_z",
    "r_i",
    "g_z",
    "u_r",
    "i_z",
    "fg_r",
    "fg_i",
    "fr_z",
    "fr_i",
    "fg_z",
    "fu_r",
    "fi_z",
]

tanya_feat = [
    "det_mjd_diff",
    "hostgal_photoz",
    "det_magn_min",
    "det_magn_mean",
    "cvec1",
    "fm_0",
    "flux_err_min",
    "exp_ratio_fitting_loss",
    "det_magn_std",
    "gauss_err_5",
    "flux_by_flux_ratio_sq_skew",
    "gauss_s_2",
    "cvec2",
    "cvec5",
    "tau_rise",
    "fm_1",
    "cvec3",
    "fm_5",
    "tau_fall",
    "gauss_s_3",
    "__median_absolute_deviation___5_",
    "__median_absolute_deviation___2_",
    "CAR_tau",
    "fm_2",
    "gauss_err_1",
    "det_magn_mean_2",
    "det_magn_mean_4",
    "det_magn_mean_3",
    "cvec0",
    "detected_mean",
    "gauss_err_2",
    "my_6_certain_width",
    "det_magn_sum_5",
    "__skew___1_",
    "__skew___2_",
    "flux_diff2",
    "gauss_s_4",
    "gauss_s_5",
    "my_6_est_width",
    "__stetson_k___3_",
    "cvec4",
    "fm_4",
    "gauss_s_1",
    "__percent_close_to_median___2_",
    "supernova1_s2_3",
    "__qso_log_chi2_qsonu___0_",
    "fm_3",
    "det_magn_min_1",
    "det_magn_min_0",
    "det_magn_min_2",
    "det_magn_min_4",
    "det_magn_min_5",
    "det_magn_min_3",
    "__freq_varrat___1_",
    "gauss_s_0",
    "gauss_err_3",
    "__freq_varrat___4_",
    "flux_skew",
    "gauss_err_4",
    "magn_fit_4",
    "magn_fit_2",
    "magn_fit_1",
    "magn_fm_5",
    "magn_fm_4",
    "magn_fm_1",
    "gauss_err_0",
    "time_score",
    "det_magn_mean_5",
    "det_magn_skew_5",
]

used_columns = [
    "CAR_tau",
    "__freq_varrat___0_",
    "__freq_varrat___1_",
    "__freq_varrat___2_",
    #'__freq_varrat___3_',
    "__freq_varrat___4_",
    #'__freq_varrat___5_',
    #'__median_absolute_deviation___0_',
    "__median_absolute_deviation___1_",
    "__median_absolute_deviation___2_",
    #'__median_absolute_deviation___3_',
    #'__median_absolute_deviation___4_',
    "__median_absolute_deviation___5_",
    #'__percent_close_to_median___0_',
    #'__percent_close_to_median___1_',
    "__percent_close_to_median___2_",
    #'__percent_close_to_median___3_',
    #'__percent_close_to_median___4_',
    #'__percent_close_to_median___5_',
    "__qso_log_chi2_qsonu___0_",
    #'__qso_log_chi2_qsonu___1_',
    #'__qso_log_chi2_qsonu___2_',
    #'__qso_log_chi2_qsonu___3_',
    #'__qso_log_chi2_qsonu___4_',
    #'__qso_log_chi2_qsonu___5_',
    #'__skew___0_',
    "__skew___1_",
    "__skew___2_",
    #'__skew___3_',
    "__skew___4_",
    #'__skew___5_',
    #'__stetson_k___0_',
    #'__stetson_k___1_',
    #'__stetson_k___2_',
    "__stetson_k___3_",
    #'__stetson_k___4_',
    #'__stetson_k___5_',
    "cvec0",
    "cvec1",
    "cvec2",
    "cvec3",
    "cvec4",
    "cvec5",
    "det_magn_mean",
    "det_magn_mean_0",
    "det_magn_mean_1",
    "det_magn_mean_2",
    "det_magn_mean_3",
    "det_magn_mean_4",
    "det_magn_mean_5",
    "det_magn_min",
    "det_magn_min_0",
    "det_magn_min_1",
    "det_magn_min_2",
    "det_magn_min_3",
    "det_magn_min_4",
    "det_magn_min_5",
    "det_magn_std",
    "det_mjd_diff",
    "detected_mean",
    "exp_ratio_fitting_loss",
    "fg_i",
    "fg_r",
    "fg_z",
    "fi_z",
    "flux_by_flux_ratio_sq_skew",
    "flux_diff2",
    "flux_err_min",
    "flux_skew",
    "fm_0",
    "fm_1",
    "fm_2",
    "fm_3",
    "fm_4",
    "fm_5",
    "fr_i",
    "fr_z",
    "fu_r",
    "g_i",
    "g_r",
    "g_z",
    "gauss_err_0",
    "gauss_err_1",
    "gauss_err_2",
    "gauss_err_3",
    "gauss_err_4",
    "gauss_err_5",
    "gauss_s_0",
    "gauss_s_1",
    "gauss_s_2",
    "gauss_s_3",
    "gauss_s_4",
    "gauss_s_5",
    "hostgal_photoz",
    "i_z",
    "magn_fit_0",
    "magn_fit_1",
    "magn_fit_2",
    "magn_fit_3",
    "magn_fit_4",
    "magn_fit_5",
    "magn_fm_0",
    "magn_fm_1",
    "magn_fm_2",
    "magn_fm_3",
    "magn_fm_4",
    "magn_fm_5",
    "my_6_certain_width",
    "my_6_est_width",
    "object_id",
    "period",
    "r_i",
    "r_z",
    "supernova1_s2_0",
    "supernova1_s2_1",
    "supernova1_s2_2",
    "supernova1_s2_3",
    "supernova1_s2_4",
    "supernova1_s2_5",
    "target",
    "tau_fall",
    "tau_rise",
    "time_score",
    "u_r",
]

for col in tanya_feat:
    if col not in used_columns:
        print("Aaaa", col)

used_columns.sort()
# for col in used_columns:
#    print(col)

print("used_columns", used_columns)

full_train = training_set_selected
full_train = pd.merge(
    left=full_train, right=training_set_car, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_fits, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_cesium, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_tanya_mag, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_my_6, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_gauss, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_supernova1, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_from_fits, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_colors, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_periods, on="object_id", how="inner"
)
full_train = pd.merge(
    left=full_train, right=training_set_metadata, on="object_id", how="inner"
)
assert len(full_train) == len(training_set_selected)

filter = (training_set_metadata["aug_fold"] == 10) | (training_set_metadata["ddf"] == 0)
non_dff_objects = training_set_metadata.loc[filter, "object_id"].values
print(len(non_dff_objects))
print("before", len(full_train))
full_train = full_train.loc[
    full_train["object_id"].isin(non_dff_objects), :
].reset_index(drop=True)
# print(full_train)
print("after clean", len(full_train))

split_df = training_set_selected.copy()
split_df = pd.merge(left=split_df, right=training_set_metadata, on="object_id")
split_df = split_df.loc[split_df["object_id"].isin(non_dff_objects), :]
split_df = split_df.reset_index(drop=True)
# print(split_df)

all_posible_columns = list(full_train.columns)
print("all_posible_columns len:", len(all_posible_columns))
print("used_columns len:", len(used_columns))
for column in used_columns:
    if column not in all_posible_columns:
        print("Achtung!!!", column)

full_train = full_train[used_columns]

if "target" in full_train:
    y = full_train["target"]
    del full_train["target"]
classes = sorted(y.unique())

# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weight = {c: 1 for c in classes}
for c in [64, 15]:
    class_weight[c] = 2
print("Unique classes : ", classes)

full_train_columns = used_columns.copy()
full_train_columns.remove("object_id")
full_train_columns.remove("target")
print(full_train_columns)
print(len(full_train_columns))

# train_mean = full_train[full_train_columns].mean(axis=0)
# seems to be better
train_mean = 0
full_train[full_train_columns] = full_train[full_train_columns].fillna(train_mean)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


def augmented_split(train, cv_folds, seed=1111):
    np.random.seed(seed)
    aug_id_y = train[["object_id", "target"]].copy()
    aug_id_y["real_object_id"] = aug_id_y["object_id"] // 100
    obj_id_y = aug_id_y[["real_object_id", "target"]].drop_duplicates()
    old_gen = cv_folds.split(obj_id_y["real_object_id"], obj_id_y["target"])

    for i in range(cv_folds.n_splits):
        train_idx, validation_idx = next(old_gen)
        train_idx = np.random.permutation(
            aug_id_y[
                aug_id_y["real_object_id"].isin(
                    obj_id_y.iloc[train_idx]["real_object_id"]
                )
            ].index.values
        )
        validation_idx = np.random.permutation(
            aug_id_y[
                aug_id_y["real_object_id"].isin(
                    obj_id_y.iloc[validation_idx]["real_object_id"]
                )
            ].index.values
        )
        yield train_idx, validation_idx


# check ddf
print("check ddf")
for fold_, (trn_, val_) in enumerate(augmented_split(split_df, folds)):
    valid_objects = full_train.loc[val_, "object_id"].values
    print(
        len(
            training_set_metadata[
                (training_set_metadata["object_id"].isin(valid_objects))
                & (training_set_metadata["ddf"] == 1)
            ]
        )
    )

class_weights_array = np.array(
    [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)


def normalize_weigths(weights):
    total_weight = 0.0
    for ind in range(0, len(weights)):
        total_weight += weights[ind]
    return weights / total_weight


class_weights_array = normalize_weigths(class_weights_array) * len(class_weights_array)
print(class_weights_array)

n_channels = 6
n_classes = 14


def multi_weighted_logloss_chai(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order="F")
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = -np.sum(y_w) / np.sum(class_arr)
    return loss


def lgbm_multi_weighted_logloss(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {
        6: 1,
        15: 2,
        16: 1,
        42: 1,
        52: 1,
        53: 1,
        62: 1,
        64: 2,
        65: 1,
        67: 1,
        88: 1,
        90: 1,
        92: 1,
        95: 1,
    }

    loss = multi_weighted_logloss_chai(y_true, y_preds, classes, class_weights)
    return "wloss", loss, False


def multi_weighted_logloss(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {
        6: 1,
        15: 2,
        16: 1,
        42: 1,
        52: 1,
        53: 1,
        62: 1,
        64: 2,
        65: 1,
        67: 1,
        88: 1,
        90: 1,
        92: 1,
        95: 1,
    }

    loss = multi_weighted_logloss_chai(y_true, y_preds, classes, class_weights)
    return loss


unique_y = np.unique(y)
class_map = dict()
for i, val in enumerate(unique_y):
    class_map[val] = i
print(class_map)

y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
# y_categorical = to_categorical(y_map)

y_count = Counter(y_map)
wtable = np.zeros((len(unique_y),))
for i in range(len(unique_y)):
    wtable[i] = y_count[i] / y_map.shape[0]
print(wtable)


def permutation_scorer(model, x_valid, y_valid):
    y_pred = model.predict_proba(x_valid, num_iteration=model.best_iteration_)
    return -multi_weighted_logloss(y_valid, y_pred)


# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def set_of_real_objects(objects):
    real_objects = [object_id // 100 for object_id in objects]
    return set(real_objects)


def look_importance(model, x_valid, y_valid):
    base_score = permutation_scorer(model, x_valid, y_valid)
    feature_names = x_valid.columns.tolist()
    n_features = len(feature_names)
    print("n_features", n_features)
    result = dict()
    for feat_index in range(0, n_features):
        feat_scores = []
        for iter_ind in range(1, 4):
            x_shuffled = x_valid.values.copy()
            np.random.seed(iter_ind)
            np.random.shuffle(x_shuffled[:, feat_index])
            cur_score = permutation_scorer(model, x_shuffled, y_valid)
            feat_scores.append(cur_score)
        feat_scores = base_score - np.array(feat_scores)
        # print('feature ', feature_names[feat_index], '-->', np.mean(feat_scores), '+-', np.std(feat_scores))
        result[feature_names[feat_index]] = feat_scores
    return pd.DataFrame(result)


def do_train(
    full_train,
    train_columns,
    only_one_fold=False,
    show_confusion_matrix=False,
    save_importances=False,
):
    clfs = []
    oof_preds = np.zeros((len(full_train), n_classes))

    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}
    for c in classes:
        weights[c] *= class_weight[c]

    # params 4 set
    best_params = {
        "device": "cpu",
        "objective": "multiclass",
        "num_class": 14,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        # tune!
        "max_depth": 3,
        "n_estimators": 5000,
        "subsample_freq": 2,
        "subsample_for_bin": 5000,
        "metric_freq": 10,
        "verbosity": -1,
        "metric": "None",
        "xgboost_dart_mode": False,
        "uniform_drop": False,
        "colsample_bytree": 0.5,
        "drop_rate": 0.173,
        "learning_rate": 0.02,
        "max_drop": 5,
        # tune!
        "min_child_samples": 50,
        "min_child_weight": 100.0,
        "min_split_gain": 0.1,
        # tune! should be less than 2^max_depth; Use small num_leaves
        "num_leaves": 5,
        # tune!
        "reg_alpha": 0.11,
        # tune!
        "reg_lambda": 0.01,
        "skip_drop": 0.44,
        "subsample": 0.75,
        "max_bin": 40
        # tune! Use small max_bin (default 255)
    }

    total_importance = None
    for fold_, (trn_, val_) in enumerate(augmented_split(split_df, folds)):
        train_objects = full_train.loc[trn_, "object_id"].values
        valid_objects = full_train.loc[val_, "object_id"].values
        print("train len", len(train_objects))
        print("valid len", len(valid_objects))
        train_objects_set = set_of_real_objects(train_objects)
        valid_objects_set = set_of_real_objects(valid_objects)
        assert train_objects_set.isdisjoint(valid_objects_set)

        x_train, y_train = full_train[train_columns].loc[trn_], y.loc[trn_]
        x_valid, y_valid = full_train[train_columns].loc[val_], y.loc[val_]

        clf = LGBMClassifier(**best_params)
        clf.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            eval_metric=lgbm_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=400,
            sample_weight=y_train.map(weights),
        )
        # Get predicted probabilities for each class
        y_pred = clf.predict_proba(x_valid, num_iteration=clf.best_iteration_)
        print(y_valid.shape)
        print(y_pred.shape)
        current_loss = multi_weighted_logloss(y_valid, y_pred)
        print(current_loss)
        clfs.append(clf)
        # # Get predicted probabilities for each class
        oof_preds[val_, :] = y_pred

        if save_importances:
            imp_df = pd.DataFrame()
            imp_df["feature"] = train_columns
            imp_df["gain"] = clf.feature_importances_
            imp_df.to_csv("importances.csv", index=False)
            plt.figure(figsize=(8, 12))
            sns.barplot(
                x="gain", y="feature", data=imp_df.sort_values("gain", ascending=False)
            )
            plt.tight_layout()
            plt.savefig("importances.png")

        fold_importance = look_importance(clf, x_valid, y_valid)
        if total_importance is None:
            total_importance = fold_importance
        else:
            total_importance = pd.concat((total_importance, fold_importance), axis=0)

        """
        perm = PermutationImportance(clf, scoring=permutation_scorer, random_state=1).fit(x_valid, y_valid)
        expl = eli5.explain_weights(perm, feature_names=x_valid.columns.tolist(), top=None)
        print(eli5.format_as_text(expl))
        print(expl.feature_importances)
        positive_features = []
        for feat_imp in expl.feature_importances.importances:
            if feat_imp.weight > 0:
                positive_features.append(feat_imp.feature)
        print('positive_features')
        print(positive_features.__repr__())
    
        text_file = open("importance.html", "w")
        text_file.write(eli5.format_as_html(expl))
        text_file.close()
        """

        if only_one_fold:
            return current_loss, clfs

    total_importance.to_csv("total_importance.csv", index=False)
    cv_loss = 0.0
    if len(clfs) > 1:
        cv_loss = multi_weighted_logloss(y, oof_preds)
        print("MULTI WEIGHTED LOG LOSS : %.5f " % cv_loss)
        if show_confusion_matrix:
            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_map, np.argmax(oof_preds, axis=-1))
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            plt.figure(figsize=(12, 12))
            plot_confusion_matrix(
                cnf_matrix, classes=columns_14, normalize=True, title="Confusion matrix"
            )
    return cv_loss, clfs


def dump_to_file(str):
    with open("log.txt", "a") as text_file:
        text_file.write(
            str + "\n" + "--------------------------------------------------------\n"
        )


"""
bo = BayesianOptimization(estimate_model, {'learning_rate': (0.01, 0.03),
                                     'n_estimators': (600, 1500),
                                     'max_depth': (3, 8)
                                     })
bo.maximize(init_points=10, n_iter=10)
print(bo.res['max'])
exit(0)
"""

only_one_fold = False
reset_state()
# drop target and object_id
best_columns = used_columns.copy()
best_columns.remove("object_id")
best_columns.remove("target")
best_loss, clfs = do_train(
    full_train,
    best_columns,
    only_one_fold=only_one_fold,
    show_confusion_matrix=False,
    save_importances=False,
)
str_to_dump = "initial loss: " + str(best_loss) + "\n"
str_to_dump += best_columns.__repr__()
dump_to_file(str_to_dump)

print("number of classifiers:", len(clfs))
exit(0)

start = time.time()
chunks = 100000

for i_c, dfs in enumerate(
    zip(
        pd.read_csv(
            "../Features_PT/test_selected_features_v6.csv",
            iterator=True,
            chunksize=chunks,
        ),
        pd.read_csv(
            "../Features_3_Good_Submit/test_set_mag_features.csv",
            iterator=True,
            chunksize=chunks,
        ),
        pd.read_csv(
            "../Features/test_set_my6_features.csv", iterator=True, chunksize=chunks
        ),
        pd.read_csv(
            "../Features/test_set_gauss_features.csv", iterator=True, chunksize=chunks
        ),
        pd.read_csv(
            "../Features/test_set_supernova1_features.csv",
            iterator=True,
            chunksize=chunks,
        ),
        pd.read_csv(
            "../Tanya/calculated/test_from_fit_features.csv",
            iterator=True,
            chunksize=chunks,
        ),
        pd.read_csv(
            "../Features_colors/test_color_features_clipped.csv",
            iterator=True,
            chunksize=chunks,
        ),
    )
):
    print("chunk", i_c)
    full_test = reduce(lambda left, right: pd.merge(left, right, on="object_id"), dfs)
    # this verifies that object_id's are the same in all files
    assert len(full_test) == len(dfs[0])
    full_test[full_train_columns] = full_test[full_train_columns].fillna(train_mean)
    test_set_objects = full_test["object_id"].values

    start_pred = time.time()
    preds = None
    for clf in clfs:
        test_pred = clf.predict_proba(
            full_test[full_train_columns], num_iteration=clf.best_iteration_
        )
        if preds is None:
            preds = test_pred / len(clfs)
        else:
            preds += test_pred / len(clfs)
    print("predict time in sec", time.time() - start_pred)

    mymean = np.mean(preds, axis=1)
    mymedian = np.median(preds, axis=1)
    mymax = np.max(preds, axis=1)
    preds_99 = (
        ((((mymedian) + (((mymean) / 2.0))) / 2.0))
        + (((((1.0) - (((mymax) * (((mymax) * (mymax))))))) / 2.0))
    ) / 2.0

    """
    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])
    preds_99 = 0.14 * preds_99 / np.mean(preds_99)
    """

    # Store predictions
    preds_df = pd.DataFrame(preds, columns=columns_14)
    preds_df["object_id"] = test_set_objects
    preds_df["class_99"] = preds_99

    if i_c == 0:
        preds_df.to_csv("predictions.csv", header=True, mode="w", index=False)
    else:
        preds_df.to_csv("predictions.csv", header=False, mode="a", index=False)

    del full_test, preds_df, preds
    print("%15d done in %5.1f" % (chunks * (i_c + 1), (time.time() - start) / 60))

z = pd.read_csv("predictions.csv")
print(z.groupby("object_id").size().max())
print((z.groupby("object_id").size() > 1).sum())

# normalize all columns
z[columns_15] = z[columns_15].div(z[columns_15].sum(axis=1), axis=0)

print("---------------------")
for column in columns_15:
    print(column, "-->", z[column].mean())

z = z.groupby("object_id").mean()
z.to_csv("predictions_reo.csv", index=True)

test_set_metadata = pd.read_csv("../Data/test_set_metadata.csv")
galactic_objects = test_set_metadata.loc[
    test_set_metadata["hostgal_photoz"] == 0, "object_id"
].values
print(len(galactic_objects), "galactic objects")

# make zero some columns
z = pd.read_csv("predictions_reo.csv")
columns_galactic = [
    "class_6",
    "class_16",
    "class_53",
    "class_65",
    "class_92",
]
columns_intergalactic = [
    "class_15",
    "class_42",
    "class_52",
    "class_62",
    "class_64",
    "class_67",
    "class_88",
    "class_90",
    "class_95",
]
assert len(columns_14) == len(columns_galactic) + len(columns_intergalactic)
galactic_cut = z["object_id"].isin(galactic_objects)
z.loc[galactic_cut, columns_intergalactic] = 0
z.loc[~galactic_cut, columns_galactic] = 0
z.to_csv("predictions_gal.csv", index=False)
