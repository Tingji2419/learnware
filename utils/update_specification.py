# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2022-05-06 14:03:13
"""

import global_variable as GV
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GV.DEVICES
import argparse
import random
import importlib
import platform
import pickle
import copy
import sys
sys.path.append("..")

import numpy as np

import torch
from torch import nn
from torchvision import models

from Train import general_train
from build_repository import Learnware
from task_generator import Task

def display_args(args):
    print('===== repository arguments =====')
    print('repo_min_coarse_label = %d' % (args.repo_min_coarse_label))
    print('repo_max_coarse_label = %d' % (args.repo_max_coarse_label))
    print('cus_min_coarse_label = %d' % (args.cus_min_coarse_label))
    print('cus_max_coarse_label = %d' % (args.cus_max_coarse_label))
    print('learnware_checkpoint = %s' % str(args.learnware_checkpoint))
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('network_name = %s' % (args.network_name))
    print('===== network arguments =====')
    print('depth = %d' % (args.depth))
    print('width = %d' % (args.width))
    print('dropout_rate = %f' % (args.dropout_rate))
    print('===== training procedure arguments =====')
    print('method = %s' % (args.method))
    print('batch_size = %d' % (args.batch_size))
    print('tau = %f' % (args.tau))
    print('===== optimizer arguments =====')
    print('lr = %f' % (args.lr))
    print('point = %s' % str(args.point))
    print('gamma = %f' % (args.gamma))
    print('weight_decay = %f' % (args.wd))
    print('momentum = %f' % (args.mo))
    print('===== experiment environment arguments =====')
    print('devices = %s' % (args.devices))
    print('flag_debug = %r' % (args.flag_debug))
    print('n_workers = %d' % (args.n_workers))



def update_repo_linear_specification(args, old_repo_learnware_dir,
                                     general_data_loader, general_embedding, flag='repo'):
    new_repo_learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
        '_minc=' + str(args.repo_min_coarse_label) + \
        '_maxc=' + str(args.repo_max_coarse_label) + \
        '_network=' + str(args.network_name) + \
        '_depth=' + str(args.depth) + \
        '_width=' + str(args.width) + \
        '_channel=' + str(args.channel) + \
        '_dropout=' + str(args.dropout_rate) + \
        '_spec=' + 'linear' + \
        '_method=' + str(args.method) + \
        '_lr=' + str(args.lr) + \
        '_point=' + str(args.point) + \
        '_gamma=' + str(args.gamma) + \
        '_wd=' + str(args.wd) + \
        '_mo=' + str(args.mo) + \
        '_tau=' + str(args.tau) + \
        '_lp=' + str(args.learnware_checkpoint) + \
        '.repo/'
    if not os.path.exists(new_repo_learnware_save_dir):
        os.makedirs(new_repo_learnware_save_dir)
    
    old_repo_learnware_names = os.listdir(old_repo_learnware_save_dir)
    for i in range(0, len(old_repo_learnware_names)):
        old_repo_learnware_path = old_repo_learnware_save_dir + old_repo_learnware_names[i]
        with open(old_repo_learnware_path, 'rb') as fp:
            old_learnware = pickle.load(fp)
        
        new_learnware = copy.deepcopy(old_learnware)
        new_learnware.generate_specification('linear',
                                             data_loader=general_data_loader,
                                             network=None,
                                             embedding=general_embedding,
                                             flag='repo')

        new_repo_learnware_path = new_repo_learnware_save_dir + old_repo_learnware_names[i]
        with open(new_repo_learnware_path, 'wb') as fp:
            pickle.dump(new_learnware, fp)

        print('%d of %d repo learnwares updated.' % (i + 1, len(old_repo_learnware_names)))



def update_cus_linear_specification(args, old_cus_learnware_dir, general_network, flag='cus'):
    new_cus_learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
        '_minc=' + str(args.cus_min_coarse_label) + \
        '_maxc=' + str(args.cus_max_coarse_label) + \
        '_network=' + str(args.network_name) + \
        '_depth=' + str(args.depth) + \
        '_width=' + str(args.width) + \
        '_channel=' + str(args.channel) + \
        '_dropout=' + str(args.dropout_rate) + \
        '_spec=' + 'linear' + \
        '_method=' + str(args.method) + \
        '_lr=' + str(args.lr) + \
        '_point=' + str(args.point) + \
        '_gamma=' + str(args.gamma) + \
        '_wd=' + str(args.wd) + \
        '_mo=' + str(args.mo) + \
        '_tau=' + str(args.tau) + \
        '_lp=' + str(args.learnware_checkpoint) + \
        '.cus/'
    if not os.path.exists(new_cus_learnware_save_dir):
        os.makedirs(new_cus_learnware_save_dir)
    
    old_cus_learnware_names = os.listdir(old_cus_learnware_save_dir)
    for i in range(0, len(old_cus_learnware_names)):
        old_cus_learnware_path = old_cus_learnware_save_dir + old_cus_learnware_names[i]
        with open(old_cus_learnware_path, 'rb') as fp:
            old_learnware = pickle.load(fp)
        
        CusData = importlib.import_module('dataloaders.' + args.data_name)
        cus_data_path = GV.dataset_path_prefix + args.data_name + '/'
        cus_data_loader = CusData.generate_data_loader(cus_data_path, old_learnware.task, 'train', 'simple', args.batch_size, args.n_workers)

        new_learnware = copy.deepcopy(old_learnware)
        new_learnware.generate_specification('linear',
                                             data_loader=cus_data_loader,
                                             network=general_network,
                                             embedding=None,
                                             flag='cus')
        
        new_cus_learnware_path = new_cus_learnware_save_dir + old_cus_learnware_names[i]
        with open(new_cus_learnware_path, 'wb') as fp:
            pickle.dump(new_learnware, fp)

        print('%d of %d cus learnwares updated.' % (i + 1, len(old_cus_learnware_names)))



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
    parser.add_argument('--repo_min_coarse_label', type=int, default=0)
    parser.add_argument('--repo_max_coarse_label', type=int, default=19)
    parser.add_argument('--cus_min_coarse_label', type=int, default=0)
    parser.add_argument('--cus_max_coarse_label', type=int, default=19)
    parser.add_argument('--repo_min_fine_label', type=int, default=0)
    parser.add_argument('--repo_max_fine_label', type=int, default=199)
    parser.add_argument('--cus_min_fine_label', type=int, default=0)
    parser.add_argument('--cus_max_fine_label', type=int, default=199)
    parser.add_argument('--learnware_checkpoint', type=int, nargs='+', default=[50, 100, 150, 200])
    # task arguments
    parser.add_argument('--data_name', type=str, default='CUB-200', choices=['CIFAR-100', 'CUB-200', 'StanfordDogs', 'Car-196'])
    parser.add_argument('--network_name', type=str, default='mobile_net', choices=['resnet', 'wide_resnet', 'mobile_net'])
    # network arguments
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--channel', type=float, default=0.25)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--spec_type', type=str, default='text', choices=['text', 'linear'])
    parser.add_argument('--method', type=str, default='raw', choices=['raw'])
    parser.add_argument('--batch_size', type=int, default=128)
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

    display_args(args)

    # build general network
    general_model_path = GV.save_path_prefix + args.data_name + '/general_models/' + \
        'general_model.record'
    with open(general_model_path, 'rb') as fp:
        record = pickle.load(fp)
    GeneralNetwork = importlib.import_module('networks.' + 'resnet')
    general_args = copy.copy(args)
    general_args.depth = 50
    general_network = GeneralNetwork.MyNetwork(general_args, n_classes=0)
    best_state_dict = {key:value for (key,value) in record['best_state_dict'].items() if not key.startswith('fc')}
    general_network.load_state_dict(best_state_dict)
    general_network.cuda()
    print('===== general network ready. =====')

    # build general data loader
    GeneralData = importlib.import_module('dataloaders.' + args.data_name)
    general_data_path = GV.save_path_prefix + args.data_name + '/general_data/'
    general_data_loader = GeneralData.generate_data_loader(general_data_path, 'all', 'train', 'simple', args.batch_size, args.n_workers)
    print('===== general data loader ready. =====')

    # load general embedding
    general_embedding_path = GV.save_path_prefix + args.data_name + '/general_data/embedding.npy'
    general_embedding = np.load(general_embedding_path)
    print('===== general embedding ready. =====')

    old_repo_learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
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

    # update_repo_linear_specification(args, old_repo_learnware_save_dir, general_data_loader, general_embedding)

    old_cus_learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
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
    
    update_cus_linear_specification(args, old_cus_learnware_save_dir, general_network)