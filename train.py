#!/usr/bin/python

import numpy as np
import argparse, random, glob, os, h5py, time
from pathlib import Path
from tqdm import tqdm
import core.config as conf
from core.dataloaders import load_data_from_hdf5
from core.consistency_av_seld import ConsistencyAVSELD
import utils.utils as utils
from utils.cls_compute_seld_results import ComputeSELDResults
import utils.cls_feature_class as cls_feature_class
# PyTorch libraries and modules
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from cm import dist_util, logger
import wandb


base_path = conf.input['project_path']

def main():
    # Set up distributed training
    dist_util.setup_dist()
    logger.configure()
    
    train_file = os.path.join(conf.input['feature_path'],
                          'h5py_{}/train_dataset.h5'.format(conf.training_param['visual_encoder_type']))
    scaler_path = os.path.join(conf.input['feature_path'],
                          'h5py_{}/feature_scaler.h5'.format(conf.training_param['visual_encoder_type']))
    test_file = os.path.join(conf.input['feature_path'],
                          'h5py_{}/test_dataset.h5'.format(conf.training_param['visual_encoder_type']))

    ## ---------- Experiment reproducibility --------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## ------------ Set check point directory ----------------
    ckpt_dir = Path(os.path.join(args.ckpt_dir, '%s/%f' % (args.info, args.lr)))
    output_folder = Path(os.path.join(base_path, 'output/%s/%f' % (args.info, args.lr)))
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    output_folder.mkdir(exist_ok=True, parents=True)

    ## --------------- Set device --------------------------
    device = dist_util.dev()
    print(f"Using device: {device}")

    ## ------------- Get feature scaler -----------------
    if args.normalize:
        mean, std = utils.load_feature_scaler(scaler_path)
    else:
        mean = None
        std = None

    ## ------------- Data loaders -----------------
    train_set = load_data_from_hdf5(train_file, normalize=args.normalize, mean=mean, std=std)
    test_set = load_data_from_hdf5(test_file, normalize=args.normalize, mean=mean, std=std)

    # Calculate proper batch size for distributed training
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            print(f"Warning: using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}")
    else:
        batch_size = args.batch_size

    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    ## ---------- Initialize the model -------------
    print("Creating ConsistencyAVSELD model...")
    model = ConsistencyAVSELD(device=device, **vars(args))
    model.to(device)
    
    # Sync model parameters across distributed processes
    if torch.distributed.is_initialized():
        dist_util.sync_params(model.model.parameters())
        dist_util.sync_params(model.target_model.parameters())

    ## ---------- Look for previous check points -------------
    first_epoch = 1
    if args.resume_training:
        ckpt_file = utils.get_latest_ckpt(ckpt_dir)
        if ckpt_file:
            model.load_weights(ckpt_file)
            first_epoch = int(str(ckpt_file)[-8:-5]) + 1
            print(f'Resuming training from epoch {first_epoch}')
        else:
            print(f'No checkpoint found in "{ckpt_dir}"...')

    plot_dir = Path(os.path.join(base_path, 'output/training_plots/%s/%f' % (args.info, args.lr)))
    plot_dir.mkdir(exist_ok=True, parents=True)
    if first_epoch == 1:
        train_loss = []
        test_loss = []
    else:
        train_loss = torch.load(os.path.join(plot_dir, 'train_loss.pt'))
        test_loss = torch.load(os.path.join(plot_dir, 'test_loss.pt'))

    ## ----------- Initialize evaluation metric class -------------
    score_obj = ComputeSELDResults()
    best_val_epoch = -1
    best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999

    ## --------------- TRAIN ------------------------
    if conf.training_param['wandb_ok']:
        # Create a simplified config dictionary for wandb
        wandb_config = {
            'lr': args.lr,
            'batch_size': batch_size,
            'epochs': args.epochs,
            'audio_priority': args.audio_priority,
            'scales': args.scales,
            'model_type': conf.training_param['model_type'],
            'visual_encoder': conf.training_param['visual_encoder_type'],
        }
        
        wandb.init(project="2025_SELD_Consistency", 
                  name=args.info,
                  config=wandb_config)
    
    for epoch in range(first_epoch, args.epochs+1):
        start_time = time.time()
        train_loss.append(model.train_model(dl_train, epoch, ckpt_dir))
        train_time = time.time() - start_time
        print(f'Train epoch time: {train_time:.2f}')

        if epoch % args.validate_every == 0:
            print(f'Epoch {epoch}, test forward pass...')
            start_time = time.time()
            ts = model.test_model(dl_test, output_folder)
            test_loss.append(ts)

            # Calculate the DCASE metrics
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(
                output_folder)

            if conf.training_param['wandb_ok']:
                save_dict = {
                    "valid/epoch_loss": ts, 
                    "valid/ER": val_ER, 
                    "valid/F": val_F,
                    "valid/LE": val_LE, 
                    "valid/LR": val_LR, 
                    "valid/seld_scr": val_seld_scr, 
                    "iter_per_epoch": epoch, 
                }
                wandb.log(save_dict)
                
            if val_seld_scr <= best_seld_scr:
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch, val_ER, val_F, val_LE, val_LR, val_seld_scr

            utils.print_stats(output_folder.parent.absolute(), args.lr, epoch, val_ER, val_F, val_LE, val_LR,
                              val_seld_scr, best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr)

            test_time = time.time() - start_time
            print(f'Test epoch time: {test_time:.2f}')
            # save test_loss list
            torch.save(test_loss, plot_dir / 'test_loss.pt')

        # save train_loss list
        torch.save(train_loss, plot_dir / 'train_loss.pt')


if __name__ == "__main__":
    # Add global_batch_size argument if not already present
    if not hasattr(conf.training_param, 'global_batch_size'):
        conf.training_param['global_batch_size'] = 64
        
    # Training settings
    parser = argparse.ArgumentParser(description='Consistency Training for AV-SELD')
    parser.add_argument('--batch-size', type=int, default=conf.training_param['batch_size'], metavar='N',
                        help='input batch size for training (default: %d)' % conf.training_param['batch_size'])
    parser.add_argument('--global-batch-size', type=int, default=conf.training_param['global_batch_size'], metavar='N',
                        help='global batch size for distributed training')
    parser.add_argument('--epochs', type=int, default=conf.training_param['epochs'], metavar='N',
                        help='number of epochs to train (default: %d)' % conf.training_param['epochs'])
    parser.add_argument('--lr', type=float, default=conf.training_param['learning_rate'], metavar='LR',
                        help='learning rate (default: %f)' % conf.training_param['learning_rate'])
    parser.add_argument('--normalize', default=True, action='store_true',
                        help='set True to normalize dataset with mean and std of train set')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--validate-every', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--ckpt-dir', default=os.path.join(conf.input['project_path'], 'ckpt'),
                        help='path to save models')
    parser.add_argument('--resume-training', default=True, action='store_true',
                        help='resume training from latest checkpoint')
    parser.add_argument('--info', type=str, default='consistency_avmodel', metavar='S',
                        help='Add additional info for storing')
    parser.add_argument('--fixAudioWeights', default=False, metavar='FxAW',
                        help='whether or not to freeze the audio weights')
    # Update the parser argument in train.py (replace only this section)
    parser.add_argument('--audio-priority', type=float, default=0.45, metavar='P',
                        help='priority weight for audio (0.5=equal, 1.0=audio only)')
    parser.add_argument('--scales', type=int, default=40, metavar='S',
                        help='maximum number of scales for consistency training')

    args = parser.parse_args()

    main()