# Photometric light curves classification with machine learning

The Large Synoptic Survey Telescope will begin its survey in 2022 and produce terabytes of imaging data each night. To work with this massive onset of data, automated algorithms to classify astronomical light curves are crucial. Here, we present a method for automated classification of photometric light curves for a range of astronomical objects. Our approach is based on the gradient boosting of decision trees, feature extraction and selection, and augmentation. The solution was developed in the context of The Photometric LSST Astronomical Time Series Classification Challenge (PLAsTiCC) and achieved one of the top results in the challenge.

__For more details, please refer to the [paper](https://arxiv.org/pdf/1909.05032.pdf).__

If you are using the results and code of this work, please cite it as
```
@article{Gabruseva_2019,
  title={Photometric light curves classification with machine learning},
  author={T. Gabruseva and S. Zlobin and P. wang},
  journal={JAI},
  year={2019}
}
```

## Dataset
The training dataset consisted of simulated astronomical light curves modeled for a
range of transients and periodic objects, see [data]() . The [dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) is available on kaggle platform.

The training dataset had 7848 light curves from 15 classes, and was highly unbalanced.

![eda](pics/eda.png)
Fig. 1 Examples of ”Normal”, ”No Lung Opacity / Not Normal”, ”Lung Opacity” chest X-Ray (CXR) images.

## Metrics
The evaluation metric was provided in the challenge. The models were evaluated with weighted multi-class logarithmic loss. [See evaluation here](https://www.kaggle.com/c/PLAsTiCC-2018/overview/evaluation). The implemented metric calculation is in src/classifier/....py

## Models
In this paper, we use python boosted decision trees implementation, [LightGBM](12),
with 5 folds cross-validation, stratified by classes.
We used different sets of features for the input of LightGBM classifier and selected the optimal features set based on the average 5-folds cross-validation scheme. The hyperparameters used are listed in the paper.

## Features
We calculated a number of various features from thelight curves. The features exptractors used for the paper can be found in src/feature_extractors . The exptracted features calculated for the train and test sets are available on kaggle for download: [features]().

## How to install and run

### Preparing the training data
To download dataset from kaggle one need to have a kaggle account, join the competition and accept the conditions, get the kaggle API token ansd copy it to .kaggle directory. After that you may run 
`bash dataset_download.sh` in the command line. The script for downloading and unpacking data is in dataset_download.sh.

### Prepare environment 
1. Install anaconda
2. You may use the create_env.sh bash file to set up the conda environment

### Reproducing the experiments 
1. Download extracted features from kaggle and place them to the input folder.
2. Train different LightGBM classifiers from src/classifiers/ folder and 
3. Run predict on the test data using the same classifiers


