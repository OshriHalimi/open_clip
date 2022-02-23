#!/bin/bash

cat /proc/cpuinfo | grep 'core id'

cd ~/ibm/open_clip

export PYTHONPATH=./src

/opt/conda/envs/open_clip/bin/python src/training/main.py \
--save-frequency \
1 \
--zeroshot-frequency \
1 \
--report-to \
tensorboard \
--train-data="~/ibm/Data/CC3M/Train_GCC-training_output.csv" \
--val-data="~/ibm/Data/CC3M/Validation_GCC-1.1.0-Validation_output.csv" \
--imagenet-val=~/ibm/Data/ImageNet/ILSVRC2012_img_val \
--csv-img-key \
filepath \
--csv-caption-key \
title \
--warmup \
10000 \
--batch-size=256 \
--lr=1e-3 \
--wd=0.1 \
--epochs=30 \
--workers=15 \
--model \
RN50 \
--ngpus_per_node 8 \
--debug