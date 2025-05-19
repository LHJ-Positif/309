#!/bin/bash

INFO='consistency_seld'
EPOCHS=50
LR=0.0001
GPU=0
AUDIO_PRIORITY=0.7
SCALES=40

export CUDA_VISIBLE_DEVICES=$GPU

python train.py --epochs=$EPOCHS --lr=$LR --info=$INFO --audio-priority=$AUDIO_PRIORITY --scales=$SCALES