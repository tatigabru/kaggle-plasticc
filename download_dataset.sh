#!/usr/bin/env bash

pip install --upgrade pip
CUR_DIR=$pwd
DATA_DIR_LOC=dataset

cd ..
mkdir -p $DATA_DIR_LOC
cd $DATA_DIR_LOC

if [ "$(ls -A $(pwd))" ]
then
    echo "$(pwd) not empty!"
else
    echo "$(pwd) is empty!"
    pip install kaggle --upgrade
    kaggle competitions download -c PLAsTiCC-2018
    mkdir input
    unzip PLAsTiCC-2018.zip -d input    
fi

cd $CUR_DIR
echo $(pwd)
