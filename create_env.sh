#!/bin/bash
conda update -n base -c defaults conda
conda create -y -n plasticc python=3.6
conda activate plasticc

pip install --upgrade pip
pip install -r requirements.txt


