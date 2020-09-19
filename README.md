# Deep-Learning-for-Skeleton-based-gesture-recognition

## Introduction
This repository holds the codebase and models for the paper:
Deep Learning for Skeleton based gesture recognition

## Prerequisites
Python libraries: pytorch, pyyaml, h5py, ffmpeg, SKvideo, opencv, signatory

## Training
To train a new Sig-ST-GCN model, run signature.py or logsignature.py 

then run: 
python main.py recognition -c config/sig_st_gcn/<dataset>/train.yaml [--work_dir <work folder>]

## Supplement
The code about ST-GCN  comes from https://github.com/yysijie/st-gcn/
