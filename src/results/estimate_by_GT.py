import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

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


def multi_weighted_logloss_without_99(y_true, y_preds):
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


def multi_weighted_logloss_with_99(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
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
        99: 2,
    }

    loss = multi_weighted_logloss_chai(y_true, y_preds, classes, class_weights)
    return loss


print("Load test true labels")
test_df = pd.read_csv("plasticc_test_metadata.csv")
sort_of_99 = {991, 992, 993, 994}
test_df.loc[test_df["true_target"].isin(sort_of_99), "true_target"] = 99
# print(test_df['true_target'].unique())
print("99 samples", (test_df["true_target"] == 99).sum())

print("Load sub")
# sub = pd.read_csv('final_result_average-two-0.815.csv')
# sub = pd.read_csv('cmcp_blend_submission_v38.csv')
# sub = pd.read_csv('kyle_final.csv')
# sub = pd.read_csv('predictions_gal_all_feat_167.csv')
sub = pd.read_csv("rescored_predictions.csv")
test_len = len(test_df)
test_df = test_df.merge(sub, on=["object_id"], how="inner")
assert len(test_df) == test_len
print("Done", test_len)

# normalize all columns
test_df[columns_15] = test_df[columns_15].div(test_df[columns_15].sum(axis=1), axis=0)

print(
    "loss with 99:",
    multi_weighted_logloss_with_99(
        test_df["true_target"].values, test_df[columns_15].values
    ),
)

y = test_df["true_target"]
unique_y = np.unique(y)
class_map = dict()
for i, val in enumerate(unique_y):
    class_map[val] = i
print(class_map)
y_map = np.array([class_map[val] for val in y])
cnf_matrix = confusion_matrix(y_map, np.argmax(test_df[columns_15].values, axis=-1))
print("Matrix calculated")

test_df = test_df[test_df["true_target"] != 99]
print("Without 99 len", len(test_df))
# normalize all columns
test_df[columns_14] = test_df[columns_14].div(test_df[columns_14].sum(axis=1), axis=0)

print(
    "loss without 99:",
    multi_weighted_logloss_without_99(
        test_df["true_target"].values, test_df[columns_14].values
    ),
)

# Compute confusion matrix

y = test_df["true_target"]
unique_y = np.unique(y)
class_map = dict()
for i, val in enumerate(unique_y):
    class_map[val] = i
print(class_map)
y_map = np.array([class_map[val] for val in y])

# cnf_matrix = confusion_matrix(y_map, np.argmax(test_df[columns_14].values, axis=-1))
# print('Matrix calculated')
# print(cnf_matrix.__repr__())

columns_14_short = [
    "6",
    "15",
    "16",
    "42",
    "52",
    "53",
    "62",
    "64",
    "65",
    "67",
    "88",
    "90",
    "92",
    "95",
]
columns_15_short = [
    "6",
    "15",
    "16",
    "42",
    "52",
    "53",
    "62",
    "64",
    "65",
    "67",
    "88",
    "90",
    "92",
    "95",
    "99",
]

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

    # print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # We change the fontsize of minor ticks label
    # plt.tick_params(axis='both', which='major', labelsize='large')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            fontsize="x-large",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    # plt.show()


import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
fig_szie = 10
fig = plt.figure(figsize=(fig_szie, fig_szie))
plot_confusion_matrix(cnf_matrix, classes=columns_15_short, normalize=True)
# fig.savefig('confusion_matrix_our_submit_on_test_with_99.png', pad_inches=0.1, bbox_inches='tight', dpi=600)
