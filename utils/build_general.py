# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2022-05-05 22:44:53
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

from Train import general_train

def display_args(args):
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('network_name = %s' % (args.network_name))
    print('===== network arguments =====')
    print('depth = %d' % (args.depth))
    print('===== training procedure arguments =====')
    print('method = %s' % (args.method))
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

if __name__ == '__main__':
    # set random seed
    random.seed(648)
    np.random.seed(648)
    torch.manual_seed(648)
    torch.cuda.manual_seed(648)
    torch.backends.cudnn.deterministic = True

    # create a parser
    parser = argparse.ArgumentParser()

    # task arguments
    parser.add_argument('--data_name', type=str, default='CIFAR-100', choices=['CIFAR-100', 'CUB-200', 'StanfordDogs', 'Car-196'])
    parser.add_argument('--network_name', type=str, default='resnet', choices=['resnet'])
    # network arguments
    parser.add_argument('--depth', type=int, default=50)
    # training procedure arguments
    parser.add_argument('--method', type=str, default='raw', choices=['raw'])
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

    args = parser.parse_args()

    if args.gamma == -1:
        args.point = [-1]
    
    display_args(args)

    general_data_path = GV.save_path_prefix + args.data_name + '/general_data/'

    Data = importlib.import_module('dataloaders.' + args.data_name)

    # train a general model for CIFAR-100
    if args.data_name == 'CIFAR-100':
        Network = importlib.import_module('networks.' + args.network_name + '_' + 'cifar')

        # generate data_loader
        train_data_loader = Data.generate_data_loader(general_data_path, 'all', 'train', 'augment', args.batch_size, args.n_workers)
        print('===== train data loader ready. =====')
        test_data_loader = Data.generate_data_loader(general_data_path, 'all', 'test', 'simple', args.batch_size, args.n_workers)
        print('===== test data loader ready. =====')

        # generate network
        network = Network.MyNetwork(args, n_classes=train_data_loader.dataset.get_n_classes())
        network = network.cuda()
        print('===== network ready. =====')

        general_model_save_path = GV.save_path_prefix + args.data_name + '/general_models/' + \
            '_network=' + str(args.network_name) + \
            '_depth=' + str(args.depth) + \
            '_nwarm=' + str(args.n_warmup_epochs) + \
            '_lr=' + str(args.lr) + \
            '_point=' + str(args.point) + \
            '_gamma=' + str(args.gamma) + \
            '_wd=' + str(args.wd) + \
            '_mo=' + str(args.mo) + \
            '.model'
        
        if os.path.exists(general_model_save_path):
            print('===== general model already exists. =====')
        else:
            print('===== general model does not exist. =====')
            record = general_train(args, train_data_loader, test_data_loader, network)
            with open(general_model_save_path, 'wb') as f:
                pickle.dump(record, f)
            print('===== general model pretrained, best test acc = %f =====' % (record['best_test_acc']))

    # use a ResNet-50 pre-trained on ImageNet for other datasets
    else:
        network = models.resnet50(pretrained=True)
        del network.fc

        record = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'best_test_acc': 0,
            'best_epoch': 0,
            'best_state_dict': network.state_dict(),
        }

        general_model_save_path = GV.save_path_prefix + args.data_name + '/general_models/' + \
            'general_model.record'

        if os.path.exists(general_model_save_path):
            print('===== general model already exists. =====')
        else:
            print('===== general model does not exist. =====')
            with open(general_model_save_path, 'wb') as f:
                pickle.dump(record, f)
            print('===== general model obtained. =====')



    # compute general embeddings
    general_embedding_save_path = GV.save_path_prefix + args.data_name + '/general_data/embedding.npy'
    if not os.path.exists(general_embedding_save_path):
        with open(general_model_save_path, 'rb') as fp:
            record = pickle.load(fp)
        network.load_state_dict(record['best_state_dict'])

        embedding_data_loader = Data.generate_data_loader(general_data_path, 'all', 'train', 'simple', args.batch_size, args.n_workers)

        # generate embedding
        for batch_index, batch in enumerate(embedding_data_loader):
            images, labels, _, _ = batch
            images = images.float().cuda()
            labels = labels.long().cuda()

            with torch.no_grad():
                embedding = network.forward(images, flag_embedding=True)
            
            if batch_index == 0:
                embedding_all = embedding.detach().cpu().numpy()
            else:
                embedding_all = np.concatenate((embedding_all, embedding.detach().cpu().numpy()), axis=0)

        # save embedding
        np.save(general_embedding_save_path, embedding_all)
    else:
        embedding_all = np.load(general_embedding_save_path)
        print(embedding_all.shape)