#!/usr/bin/python

import argparse, os
from pathlib import Path
from tqdm import tqdm
import core.config as conf
from core.dataloaders import load_data_from_hdf5
from core.consistency_av_seld import ConsistencyAVSELD
from utils.cls_compute_seld_results import ComputeSELDResults
import utils.cls_feature_class as cls_feature_class
import utils.utils as utils
# PyTorch libraries and modules
import torch
from torch.utils.data import DataLoader

base_path = conf.input['project_path']

def main():
    test_file = os.path.join(conf.input['feature_path'],
                             'h5py_{}/test_dataset.h5'.format(conf.training_param['visual_encoder_type']))
    scaler_path = os.path.join(conf.input['feature_path'],
                             'h5py_{}/feature_scaler.h5'.format(conf.training_param['visual_encoder_type']))

    ## ----------- Set output dir --------------------------
    output_folder = Path(os.path.join(base_path, 'output/%s/%f' % (args.info, args.lr)))
    output_folder_student = Path(os.path.join(base_path, 'output/%s/%f/student' % (args.info, args.lr)))
    output_folder_target = Path(os.path.join(base_path, 'output/%s/%f/target' % (args.info, args.lr)))
    
    output_folder.mkdir(exist_ok=True, parents=True)
    output_folder_student.mkdir(exist_ok=True, parents=True)
    output_folder_target.mkdir(exist_ok=True, parents=True)

    ## ----------- Set device --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ## ------------- Get feature scaler -----------------
    if args.normalize:
        mean, std = utils.load_feature_scaler(scaler_path)
    else:
        mean = None
        std = None

    ## ---------- Data loaders -----------------
    dev_test_cls = cls_feature_class.FeatureClass(train_or_test='test')
    test_seqs_paths_list = dev_test_cls.get_sequences_paths_list()
    d_test = load_data_from_hdf5(test_file, normalize=args.normalize, mean=mean, std=std)
    # keep batch_size=1, shuffle=false, num_workers=1!
    dl_test = DataLoader(d_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    ## ---------- Load consistency model weights -------------
    model = ConsistencyAVSELD(device=device, **vars(args))
    model.load_weights(os.path.join(base_path, 'ckpt/%s/%f/model_%03d.ckpt' % (args.info, args.lr, args.epoch)))

    # Initialize evaluation metric class
    score_obj = ComputeSELDResults()

    ## ----------- FORWARD TEST SET WITH STUDENT MODEL -------------------------------
    print('Evaluating student model...')
    model.av_seld.model = model.model  # Use student model for evaluation
    student_loss = model.av_seld.test_model(dl_test, output_folder_student)
    
    ## ----------- FORWARD TEST SET WITH TARGET MODEL -------------------------------
    print('Evaluating target model...')
    model.av_seld.model = model.target_model  # Use target model for evaluation
    target_loss = model.av_seld.test_model(dl_test, output_folder_target)

    ## ------------ EVALUATION OF STUDENT MODEL ------------------------------------
    use_jackknife = False # set False for faster evaluation
    print('\nStudent Model Results:')
    student_ER, student_F, student_LE, student_LR, student_seld_scr, student_classwise_scr = score_obj.get_SELD_Results(
        output_folder_student, is_jackknife=use_jackknife)
    print('SELD score: {:0.2f}'.format(student_seld_scr))
    print('SED metrics: Error rate: {:0.2f}, F-score: {:0.1f}'.format(
        student_ER, 100 * student_F))
    print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(
        student_LE, 100 * student_LR))

    ## ------------ EVALUATION OF TARGET MODEL ------------------------------------
    print('\nTarget Model Results:')
    target_ER, target_F, target_LE, target_LR, target_seld_scr, target_classwise_scr = score_obj.get_SELD_Results(
        output_folder_target, is_jackknife=use_jackknife)
    print('SELD score: {:0.2f}'.format(target_seld_scr))
    print('SED metrics: Error rate: {:0.2f}, F-score: {:0.1f}'.format(
        target_ER, 100 * target_F))
    print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(
        target_LE, 100 * target_LR))
        
    ## ------------ COMPARISON ------------------------------------
    print('\nComparison (Target - Student):')
    print('SELD score difference: {:0.2f}'.format(target_seld_scr - student_seld_scr))
    print('Error rate difference: {:0.2f}'.format(target_ER - student_ER))
    print('F-score difference: {:0.1f}'.format(100 * (target_F - student_F)))
    print('Localization error difference: {:0.1f}'.format(target_LE - student_LE))
    print('Localization Recall difference: {:0.1f}'.format(100 * (target_LR - student_LR)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test set forward pass, specify arguments')
    parser.add_argument('--epoch', type=int, default=28, metavar='N',
                        help='number of epochs (default: 28)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: %f)' % conf.training_param['learning_rate'])
    parser.add_argument('--normalize', default=True, action='store_true',
                        help='set True to normalize dataset with mean and std of train set')
    parser.add_argument('--info', type=str, default='consistency_avmodel', metavar='S',
                        help='Add additional info for storing')
    parser.add_argument('--model-type', type=str, default=conf.training_param['model_type'], metavar='WS')
    parser.add_argument('--audio-priority', type=float, default=0.45, metavar='P',
                        help='priority weight for audio (0.5=equal, 1.0=audio only)')
    args = parser.parse_args()
    main()