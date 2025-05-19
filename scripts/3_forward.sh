#!/bin/bash

INFO='consistency_seld'
EPOCH=33 #use your best epoch 32
LR=0.0001
AUDIO_PRIORITY=0.7

python forward.py --epoch=$EPOCH --lr=$LR --info=$INFO --audio-priority=$AUDIO_PRIORITY