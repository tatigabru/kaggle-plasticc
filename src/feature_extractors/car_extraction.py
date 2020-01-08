import os
import sys
import warnings

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from src.config import SAVE_DIR
from src.feature_extractors.car_extractor import calculate_car


sys.path.append('/home/user/plasticc/kaggle-plasticc/src')

def extract_car_from_group(group):
    g = group.sort_values('mjd')
    t = g['mjd'].values
    f = g['flux'].values
    e = g['flux_err'].values
    sigma, tau = calculate_car(t, f, e)
    return pd.Series({'CAR_sigma': sigma, 'CAR_tau': tau})


if __name__ == '__main__':

    """
    training set
    """
    # train = pd.read_csv('data/training_set.csv')
    # train_meta = pd.read_csv('data/training_set_metadata.csv')
    #
    # res = train.groupby(['object_id', 'passband']).apply(extract_car_from_group)
    #
    # flat = res.unstack('passband')
    # flat.columns = [f'{i}|{j}' if j != '' else f'{i}' for i, j in flat.columns]
    # flat.to_csv(os.path.join(SAVE_DIR, 'train_car_features.csv'), float_format='%.6g')

    """
    test set
    """
    # print('starting cluster...')
    # cluster = LocalCluster(n_workers=6, processes=True, scheduler_port=0,
    #                        diagnostics_port=8787)
    # client = Client(cluster)
    #
    # test = dd.read_parquet('data/test_set/')
    #
    # warnings.filterwarnings('ignore')
    # for n, part in tqdm(enumerate(test.partitions), total=test.npartitions):
    #     if n <= 48:
    #         continue
    #     chunk = dd.from_pandas(part.compute(), npartitions=100)
    #     # chunk = part.compute()
    #     res = chunk.groupby(['object_id', 'passband']).apply(extract_car_from_group).compute()
    #     flat = res.unstack('passband')
    #     flat.columns = [f'{i}|{j}' if j != '' else f'{i}' for i, j in flat.columns]
    #     flat.to_csv(os.path.join(DEST, f'test/part_{n}.csv'), float_format='%.6g')

    """
    augmented set combined
    """
    train = pd.read_csv('data/gp_augmented/gp_augmented_ddf_to_nonddf_class_52.csv')
    flux_mean = train.groupby(['augmentation_id', 'passband'])['flux'].transform('mean')
    flux_std = train.groupby(['augmentation_id', 'passband'])['flux'].transform('std')
    train['flux'] = (train['flux'] - flux_mean) / flux_std
    train['flux_err'] = train['flux_err'] / flux_std

    res = train.groupby(['augmentation_id']).apply(extract_car_from_group)

    # flat = res.unstack('passband')
    # flat.columns = [f'{i}|{j}' if j != '' else f'{i}' for i, j in flat.columns]
    res.to_csv(os.path.join(SAVE_DIR, 'train_car_features_combined.csv'), float_format='%.6g')
