#!/bin/bash

conda create -y -n plasticc python=3.6
conda activate plasticc

pip install --upgrade pip
pip install -r requirements.txt


