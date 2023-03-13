# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-03-13 17:25:50
"""

from utils import global_variable as GV
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
from torch.nn import functional as F
from torchvision import models

from Train import train
from task_generator import Task
from learnware import Learnware

def display_args(args):
    print('===== repository arguments =====')
    print('n_repo_tasks = %d' % (args.n_repo_tasks))
    print('repo_min_coarse_label = %d' % (args.repo_min_coarse_label))
    print('repo_max_coarse_label = %d' % (args.repo_max_coarse_label))
    print('repo_min_fine_label = %d' % (args.repo_min_fine_label))
    print('repo_max_fine_label = %d' % (args.repo_max_fine_label))  
    print('n_learnwares_per_task = %d' % (args.n_learnwares_per_task))
    print('learnware_checkpoint = %s' % str(args.learnware_checkpoint))
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('network_name = %s' % (args.network_name))
    print('===== network arguments =====')
    print('depth = %d' % (args.depth))
    print('width = %d' % (args.width))
    print('dropout_rate = %f' % (args.dropout_rate))
    print('===== training procedure arguments =====')
    print('n_training_epochs = %d' % (args.n_training_epochs))
    print('n_warmup_epochs = %d' % (args.n_warmup_epochs))
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



def build_one_task(args, task, task_list_file_name, task_id, learnware_save_dir):
    data_path = GV.dataset_path_prefix + args.data_name + '/'
    train_data_loader = Data.generate_data_loader(data_path, task, 'train', 'augment',
                                                  args.batch_size, args.n_workers)
    test_data_loader = Data.generate_data_loader(data_path, task, 'test', 'simple',
                                                 args.batch_size, args.n_workers)
    print('===== task %d: datasets ready. =====' % (task_id))

    network = Network.MyNetwork(args, n_classes=task.get_n_fine_labels())
    network = network.cuda()
    print('===== task %d: network ready. =====' % (task_id))

    learnware_components = train(args, train_data_loader, test_data_loader, network)

    for i in range(0, args.n_learnwares_per_task):
        state_dict, epoch, train_loss_list, train_acc_list, test_acc_list = \
            learnware_components[i]

        # create a new learnware
        learnware = Learnware(args, task_list_file_name, task_id, task,
                              state_dict, epoch, train_loss_list, train_acc_list,
                              test_acc_list)
        
        # generate text specification
        learnware.generate_text_specification()

        # generate linear specification
        GeneralData = importlib.import_module('dataloaders.' + args.data_name)
        general_data_path = GV.save_path_prefix + args.data_name + '/general_data/'
        general_data_loader = GeneralData.generate_data_loader(general_data_path, 'all', 'train', 'simple', args.batch_size, args.n_workers)
        
        general_embedding_path = GV.save_path_prefix + args.data_name + '/general_data/embedding.npy'
        general_embedding = np.load(general_embedding_path)

        learnware.generate_linear_specification(general_data_loader, general_embedding)

        # save learnware
        learnware_file_name = \
            '_id=' + str(task_id) + \
            '_epoch=' + str(epoch) + \
            '.lw'
        
        with open(learnware_save_dir + learnware_file_name, 'wb') as fp:
            pickle.dump(learnware, fp)

    

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
    parser.add_argument('--n_repo_tasks', type=int, default=25)
    parser.add_argument('--repo_min_coarse_label', type=int, default=0)
    parser.add_argument('--repo_max_coarse_label', type=int, default=19)
    parser.add_argument('--repo_min_fine_label', type=int, default=0)
    parser.add_argument('--repo_max_fine_label', type=int, default=199)
    parser.add_argument('--n_learnwares_per_task', type=int, default=4)
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
    parser.add_argument('--n_training_epochs', type=int, default=200)
    parser.add_argument('--n_warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=3)
    # optimizer arguments
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--point', type=int, nargs='+', default=[60, 120, 160])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--wd', type=float, default=0.0005)
    parser.add_argument('--mo', type=float, default=0.9)
    # experiment environment arguments
    parser.add_argument('--devices', type=str, default=GV.DEVICES)
    parser.add_argument('--flag_debug', action='store_true', default=False)
    parser.add_argument('--n_workers', type=int, default=GV.WORKERS)
    parser.add_argument('--id_server', type=int, default=0)

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
    
    assert len(args.learnware_checkpoint) == args.n_learnwares_per_task
    assert max(args.learnware_checkpoint) <= args.n_training_epochs
    if args.data_name == 'CIFAR-100':
        assert args.repo_min_coarse_label >= 0
        assert args.repo_max_coarse_label <= 19
    elif args.data_name == 'CUB-200':
        args.repo_min_coarse_label = args.repo_min_fine_label
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.batch_size = 128
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 199
    elif args.data_name == 'StanfordDogs':
        args.repo_min_coarse_label = args.repo_min_fine_label
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.batch_size = 128
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 119
    elif args.data_name == 'Car-196':
        args.repo_min_coarse_label = args.repo_min_fine_label
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.batch_size = 128
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 195
    else:
        pass

    display_args(args)

    repo_task_list_path = GV.dataset_path_prefix + args.data_name + '/task_lists/' + \
        '_minc=' + str(args.repo_min_coarse_label) + \
        '_maxc=' + str(args.repo_max_coarse_label) + \
        '.repo'
    with open(repo_task_list_path, 'rb') as fp:
        repo_task_list = pickle.load(fp)
    assert args.n_repo_tasks <= len(repo_task_list)
    
    # import modules
    Data = importlib.import_module('dataloaders.' + args.data_name)
    Network = importlib.import_module('networks.' + args.network_name)

    if args.n_repo_tasks != 0:
        repo_learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
            '_minc=' + str(args.repo_min_coarse_label) + \
            '_maxc=' + str(args.repo_max_coarse_label) + \
            '_network=' + str(args.network_name) + \
            '_depth=' + str(args.depth) + \
            '_width=' + str(args.width) + \
            '_channel=' + str(args.channel) + \
            '_dropout=' + str(args.dropout_rate) + \
            '_nwarm=' + str(args.n_warmup_epochs) + \
            '_lr=' + str(args.lr) + \
            '_point=' + str(args.point) + \
            '_gamma=' + str(args.gamma) + \
            '_wd=' + str(args.wd) + \
            '_mo=' + str(args.mo) + \
            '_tau=' + str(args.tau) + \
            '_lp=' + str(args.learnware_checkpoint) + \
            '.repo/'
        if not os.path.exists(repo_learnware_save_dir):
            os.makedirs(repo_learnware_save_dir)
        repo_task_list_file_name = \
            '_minc=' + str(args.repo_min_coarse_label) + \
            '_maxc=' + str(args.repo_max_coarse_label) + \
            '.repo'

        assert (args.id_server + 1) * args.n_repo_tasks <= len(repo_task_list)
        left = args.id_server * args.n_repo_tasks
        for i in range(left, left + args.n_repo_tasks):
            build_one_task(args, repo_task_list[i], repo_task_list_file_name, 
                           i, repo_learnware_save_dir)
            print('%d of %d repo tasks complete.' % (i + 1, args.n_repo_tasks))
    
    display_args(args)