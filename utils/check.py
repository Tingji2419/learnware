# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-03-10 16:59:24
"""

import global_variable as GV
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GV.DEVICES
import argparse
import random
import importlib
import platform
import pickle
import sys
sys.path.append("..")

import numpy as np

import torch
from torch import nn
from torchvision import models

from task_generator import Task
from build_repository import Learnware

def check_learnware(args):
    if args.method == 'raw':
        if args.split == 'repo':
            learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
                '_minc=' + str(args.repo_min_coarse_label) + \
                '_maxc=' + str(args.repo_max_coarse_label) + \
                '_network=' + str(args.network_name) + \
                '_depth=' + str(args.depth) + \
                '_width=' + str(args.width) + \
                '_channel=' + str(args.channel) + \
                '_dropout=' + str(args.dropout_rate) + \
                '_spec=' + str(args.spec_type) + \
                '_method=' + str(args.method) + \
                '_lr=' + str(args.lr) + \
                '_point=' + str(args.point) + \
                '_gamma=' + str(args.gamma) + \
                '_wd=' + str(args.wd) + \
                '_mo=' + str(args.mo) + \
                '_tau=' + str(args.tau) + \
                '_lp=' + str(args.learnware_checkpoint) + \
                '.repo/'
        elif args.split == 'cus':
            learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
                '_minc=' + str(args.cus_min_coarse_label) + \
                '_maxc=' + str(args.cus_max_coarse_label) + \
                '_network=' + str(args.network_name) + \
                '_depth=' + str(args.depth) + \
                '_width=' + str(args.width) + \
                '_channel=' + str(args.channel) + \
                '_dropout=' + str(args.dropout_rate) + \
                '_spec=' + str(args.spec_type) + \
                '_method=' + str(args.method) + \
                '_lr=' + str(args.lr) + \
                '_point=' + str(args.point) + \
                '_gamma=' + str(args.gamma) + \
                '_wd=' + str(args.wd) + \
                '_mo=' + str(args.mo) + \
                '_tau=' + str(args.tau) + \
                '_lp=' + str(args.learnware_checkpoint) + \
                '.cus/'
    
    # alpha appears in learnware_save_dir
    # alpha is the weight of the loss of the fitnet
    elif args.method == 'fitnet':
        if args.split == 'repo':
            learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
                '_minc=' + str(args.repo_min_coarse_label) + \
                '_maxc=' + str(args.repo_max_coarse_label) + \
                '_network=' + str(args.network_name) + \
                '_depth=' + str(args.depth) + \
                '_width=' + str(args.width) + \
                '_channel=' + str(args.channel) + \
                '_dropout=' + str(args.dropout_rate) + \
                '_spec=' + str(args.spec_type) + \
                '_method=' + str(args.method) + \
                '_lr=' + str(args.lr) + \
                '_point=' + str(args.point) + \
                '_gamma=' + str(args.gamma) + \
                '_wd=' + str(args.wd) + \
                '_mo=' + str(args.mo) + \
                '_tau=' + str(args.tau) + \
                '_alpha=' + str(args.alpha) + \
                '_lp=' + str(args.learnware_checkpoint) + \
                '.repo/'
        elif args.split == 'cus':
            learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
                '_minc=' + str(args.cus_min_coarse_label) + \
                '_maxc=' + str(args.cus_max_coarse_label) + \
                '_network=' + str(args.network_name) + \
                '_depth=' + str(args.depth) + \
                '_width=' + str(args.width) + \
                '_channel=' + str(args.channel) + \
                '_dropout=' + str(args.dropout_rate) + \
                '_spec=' + str(args.spec_type) + \
                '_method=' + str(args.method) + \
                '_lr=' + str(args.lr) + \
                '_point=' + str(args.point) + \
                '_gamma=' + str(args.gamma) + \
                '_wd=' + str(args.wd) + \
                '_mo=' + str(args.mo) + \
                '_tau=' + str(args.tau) + \
                '_alpha=' + str(args.alpha) + \
                '_lp=' + str(args.learnware_checkpoint) + \
                '.cus/'

    learnware_file_name = \
        '_id=' + str(args.id) + \
        '_epoch=' + str(args.epoch) + \
        '.lw'
    
    with open(learnware_save_dir + learnware_file_name, 'rb') as fp:
        learnware = pickle.load(fp)
    
    print(type(learnware))
    learnware.show()



if __name__ == '__main__':
    # set random seed
    random.seed(648)
    np.random.seed(648)
    torch.manual_seed(648)
    torch.cuda.manual_seed(648)
    torch.backends.cudnn.deterministic = True

    # create a parser
    parser = argparse.ArgumentParser()

    # repository arguments
    parser.add_argument('--split', type=str, default='repo', choices=['repo', 'cus'])
    parser.add_argument('--repo_min_coarse_label', type=int, default=0)
    parser.add_argument('--repo_max_coarse_label', type=int, default=19)
    parser.add_argument('--cus_min_coarse_label', type=int, default=0)
    parser.add_argument('--cus_max_coarse_label', type=int, default=19)
    parser.add_argument('--repo_min_fine_label', type=int, default=0)
    parser.add_argument('--repo_max_fine_label', type=int, default=199)
    parser.add_argument('--cus_min_fine_label', type=int, default=0)
    parser.add_argument('--cus_max_fine_label', type=int, default=199)
    parser.add_argument('--learnware_checkpoint', type=int, nargs='+', default=[50, 100, 150, 200])
    # learnware arguments
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    # task arguments
    parser.add_argument('--data_name', type=str, default='CIFAR-100', choices=['CIFAR-100', 'CUB-200', 'StanfordDogs', 'Car-196'])
    parser.add_argument('--network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet', 'mobile_net'])
    # network arguments
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--channel', type=float, default=0.25)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--spec_type', type=str, default='text', choices=['text', 'linear'])
    parser.add_argument('--method', type=str, default='raw', choices=['raw', 'fitnet', 'rkd'])
    parser.add_argument('--tau', type=float, default=3)
    parser.add_argument('--alpha', type=float, default=0.01)
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

    args = parser.parse_args()

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

    if args.gamma == -1:
        args.point = [-1]

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
    elif args.data_name == 'StanfordDogs':
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        args.batch_size = 128
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 119
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 119
    elif args.data_name == 'Car-196':
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        args.batch_size = 128
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 195
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 195
    else:
        pass

    check_learnware(args)