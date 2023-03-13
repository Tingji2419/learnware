# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-04-25 20:49:13
"""
from utils import global_variable as GV
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GV.DEVICES
import argparse
import random
import importlib
import platform
import pickle

import numpy as np

import torch
from torch import nn, randint
from torchvision import models

from Train import train, distill_train
from build_repository import Learnware
from task_generator import Task
import methods
from evaluate import Retrain, get_learnwares_path, search_learnware


def experiment_1(args, target_data_paths):
    # cus model without teacher model
    # directly find epoch=200.cus

    test_acc=[]
    for target_data_path in target_data_paths:
        temp = target_data_path.replace('50.lw','200.lw').\
                                replace('100.lw','200.lw').\
                                replace('150.lw','200.lw')
        with open(temp, 'rb') as fp:
            final_learnware=pickle.load(fp)
        test_acc.append(final_learnware.test_acc_list)
    
    return test_acc



def experiment2(args, target_data_paths):
    args.alpha = 0.1
    args.n_training_epochs=200
    args.distill_loss = 'HintLoss'

    for target_data_path in target_data_paths:
        with open(target_data_path, 'rb') as fp:
            target_data=pickle.load(fp)
        
        selected_learnware_path=search_learnware(args, target_data)

    with open(selected_learnware_path, 'rb') as fp:
        teacher_learnware=pickle.load(fp)


    before_train_acc=target_data.test_acc_list
    after_train_acc=Retrain(args, teacher_learnware, target_data)
    acc_improve=max(after_train_acc)-max(before_train_acc)

    repo_task_list_path = GV.dataset_path_prefix + args.data_name + '/task_lists/' + \
        '_minc=' + str(args.repo_min_coarse_label) + \
        '_maxc=' + str(args.repo_max_coarse_label) + \
        '.repo'
    with open(repo_task_list_path, 'rb') as fp:
        repo_task_list = pickle.load(fp)

    s_class=repo_task_list[target_data.task_id].fine_labels.sort()
    t_class=repo_task_list[teacher_learnware.task_id].fine_labels.sort()
    
    print('Test acc Imporvement:')
    print(acc_improve)

def create_args():
 # create a parser
    parser = argparse.ArgumentParser()

    # repository arguments
    parser.add_argument('--repo_min_coarse_label', type=int, default=0)
    parser.add_argument('--repo_max_coarse_label', type=int, default=19)
    parser.add_argument('--cus_min_coarse_label', type=int, default=0)
    parser.add_argument('--cus_max_coarse_label', type=int, default=19)
    parser.add_argument('--repo_min_fine_label', type=int, default=0)
    parser.add_argument('--repo_max_fine_label', type=int, default=199)
    parser.add_argument('--cus_min_fine_label', type=int, default=0)
    parser.add_argument('--cus_max_fine_label', type=int, default=199)
    parser.add_argument('--n_learnwares_per_task', type=int, default=4)
    parser.add_argument('--learnware_checkpoint', type=int, nargs='+', default=[50, 100, 150, 200])
    parser.add_argument('--spec_type', type=str, default='text', choices=['text'])
    # task arguments
    parser.add_argument('--data_name', type=str, default='CIFAR-100', choices=['CIFAR-100','CUB-200'])
    parser.add_argument('--network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet'])
    # network arguments
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--channel', type=int, default=0.25)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--distill_loss', type=str, default='HintLoss', choices=['HintLoss','RKDLoss'])
    parser.add_argument('--method', type=str, default='raw', choices=['raw'])
    parser.add_argument('--n_training_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=3)
    # optimizer arguments
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--point', type=int, nargs='+', default=[50, 100, 150])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--mo', type=float, default=0.9)
    # experiment environment arguments
    parser.add_argument('--devices', type=str, default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    parser.add_argument('--id_server', type=int, default=0)

    # distilling arguments
    parser.add_argument('--alpha', type=int, default=0.2)
    parser.add_argument('--alpha', type=int, default=0.2)

    args = parser.parse_args()
    # args.data_name='CUB-200'
    # args.network_name='mobile_net'
    # args.channel=0.25
    # args.point=[150, 170, 180]
    if args.network_name == 'resnet':
        args.width = -1
        args.dropout_rate = -1
        args.channel = -1
    elif args.network_name == 'wide_resnet':
        args.channel = -1
    elif args.network_name == 'mobile_net':
        args.depth = -1
        args.width = -1
        args.dropout_rate = -1
    if args.data_name == 'CIFAR-100':
        assert args.repo_min_coarse_label >= 0
        assert args.repo_max_coarse_label <= 19
        assert args.cus_min_coarse_label >= 0
        assert args.cus_max_coarse_label <= 19
    elif args.data_name == 'CUB-200':
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        args.batch_size = 128
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 199
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 199
    else:
        pass

    return args


if __name__=="__main__":
    # set random seed
    random.seed(648)
    np.random.seed(648)
    torch.manual_seed(648)
    torch.cuda.manual_seed(648)
    torch.backends.cudnn.deterministic = True

    args=create_args()
    
    n_cus=10
    cus_learnwares_path=get_learnwares_path(args,'cus')
    target_data_path=[]
    for i in range(0, n_cus):
        # random cus_task
        target_data_path.append(cus_learnwares_path[\
                                random.randint(0,len(cus_learnwares_path))])
    
    experiment_1(args, target_data_path)
