import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

columns_14 = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53', 'class_62', 'class_64',
               'class_65', 'class_67', 'class_88', 'class_90', 'class_92', 'class_95']
columns_15 = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53', 'class_62', 'class_64',
               'class_65', 'class_67', 'class_88', 'class_90', 'class_92', 'class_95', 'class_99']

data = pd.read_csv('predictions_gal_81feat_no_augs.csv', encoding='utf-8')
# normalize first 14 columns
data[columns_14] = data[columns_14].div(data[columns_14].sum(axis=1), axis=0)

def GenUnknown(data):
    return ((((((data["mymedian"]) + (((data["mymean"]) / 2.0)))/2.0)) + (((((1.0) - (((data["mymax"]) * (((data["mymax"]) * (data["mymax"]))))))) / 2.0)))/2.0)

y = pd.DataFrame()
y['mymean'] = data[columns_14].mean(axis=1)
y['mymedian'] = data[columns_14].median(axis=1)
y['mymax'] = data[columns_14].max(axis=1)
data['class_99'] = GenUnknown(y)

# normalize all columns
data[columns_15] = data[columns_15].div(data[columns_15].sum(axis=1), axis=0)

print('save rescored predictions')
data.to_csv('rescored_predictions.csv', index=False)