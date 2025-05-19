#!/usr/bin/python

import torch, os



input = {
    'project_path': '/home/user/workspace/code/309_308_consistency', #,os.getcwd()
    'data_path': '/home/user/workspace/dataset/data_dcase2023_task3', # path to dataset
    'feature_path': '/home/user/workspace/dataset/data_dcase2023_task3/features', # might want to save h5py (and scaler) somewhere else.
    'features': 'IV',
    'fps': 10,
    'fs': 24000, # Hz
    'input_len_sec': 3, # seconds
    'input_step_train': 0.5, #seconds
    'input_step_test': 3, #seconds (no overlap)
    'num_classes': 13,
    'audio_format': 'mic', # 'foa' or 'mic'
    'label_resolution': 10,
}

training_param = {
    'optimizer': torch.optim.Adam,
    #'criterion': nn.CrossEntropyLoss,
    'learning_rate': 0.0001, # default if user does not provide a different lr to the parser
    'epochs': 50, # default if user does not provide a different number to the parser
    'batch_size': 32, # default 32 if user does not provide a different size to the parser
    'frame_len_samples': input['input_len_sec'] * input['fs'], # number of audio samples in input_len_sec,
    'num_video_frames': input['input_len_sec'] * input['fps'], # number of video frames in input_len_sec,
    'visual_encoder_type': 'resnet', # choose between 'resnet' or 'i3d'
    'model_type': 'conformer', # choose between 'conformer' or 'cmaf' or 'cmaf_conformer' or "mamba"
    'num_heads': 8, # numb1er of heads in MHSA-MHCA
    'num_cmaf_layers': 2, #희재 = 4
    'num_conformer_layers': 4,  # cmaf_conformer 전용: concat 후 Conformer 레이어 수
    # 'num_mamba_layers': 1, # film 전용
    'wandb_ok': True
}

spectrog_param = { # used for log mel spec, gcc-phat, or salsa, or IV
    'winlen': 512, # samples
    'hoplen': 150, # samples
    'numcep': 128, # number of cepstrum bins to return
    'n_fft': 512, #fft lenght
    'fmin': 0, #Hz
    'fmax': 12000 #Hz
}

metric = {
    'lad_doa_thresh': 20,
    'average': 'macro'
}

