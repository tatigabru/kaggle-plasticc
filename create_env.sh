#!/bin/bash
conda update -n base -c defaults conda
conda create -y -n plasticc python=3.6
conda activate plasticc

conda install -y -n rsna pytorch=0.4.1 cuda90 -c pytorch
pip install --upgrade pip
pip install -r requirements.txt


