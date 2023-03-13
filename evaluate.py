# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-03-30 20:49:13
"""

from utils import global_variable as GV
import os
os.environ['CUDA_VISIBLE_DEVICES'] = GV.DEVICES
import argparse
import random
import importlib
import platform
import pickle
import copy

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn, randint
from torchvision import models

from Train import train, distill_train
from learnware import Learnware
from task_generator import Task

def display_args(args):
    print('===== customer arguments =====')
    print('n_cus_tasks = %d' % (args.n_cus_tasks))
    print('cus_min_coarse_label = %d' % (args.cus_min_coarse_label))
    print('cus_max_coarse_label = %d' % (args.cus_max_coarse_label))
    print('cus_min_fine_label = %d' % (args.cus_min_fine_label))
    print('cus_max_fine_label = %d' % (args.cus_max_fine_label))   
    print('spec_type = %s' % (args.spec_type))
    print('===== task arguments =====')
    print('data_name = %s' % (args.data_name))
    print('network_name = %s' % (args.network_name))
    print('===== network arguments =====')
    print('depth = %d' % (args.depth))
    print('width = %d' % (args.width))
    print('dropout_rate = %f' % (args.dropout_rate))
    print('===== training procedure arguments =====')
    print('method = %s' % (args.method))
    print('n_training_epochs = %d' % (args.n_training_epochs))
    print('n_warmup_epochs = %d' % (args.n_warmup_epochs))
    print('batch_size = %d' % (args.batch_size))
    print('tau = %f' % (args.tau))
    print('alpha = %f' % (args.alpha))
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



def retrain(args, selected_repo_learnware, target_cus_learnware):
    data_path = GV.dataset_path_prefix + target_cus_learnware.args.data_name + '/'

    teacher_args = selected_repo_learnware.args
    student_args = target_cus_learnware.args

    # generate data loaders for target_cus_learnware
    Data = importlib.import_module('dataloaders.' + student_args.data_name)
    train_data_loader = Data.generate_data_loader(data_path, target_cus_learnware.task, 'train', 'augment',
                                                  student_args.batch_size, student_args.n_workers)
    test_data_loader = Data.generate_data_loader(data_path, target_cus_learnware.task, 'test', 'simple',
                                                 student_args.batch_size, student_args.n_workers)
    # generate teacher network
    tearcher_Network = importlib.import_module('networks.' + teacher_args.network_name)
    teacher_network = tearcher_Network.MyNetwork(teacher_args,
        n_classes=selected_repo_learnware.task.get_n_fine_labels())
    teacher_network.load_state_dict(selected_repo_learnware.state_dict)
    teacher_network.cuda()
    
    # generate student network
    student_Network = importlib.import_module('networks.' + student_args.network_name)
    student_network = student_Network.MyNetwork(student_args,
        n_classes=target_cus_learnware.task.get_n_fine_labels())
    # student_network.load_state_dict(target_cus_learnware.state_dict)
    student_network.cuda()
    
    new_target_cus_learnware_component = distill_train(args, train_data_loader, test_data_loader, teacher_network, student_network)

    state_dict, epoch, train_loss_list, guide_loss_list, train_acc_list, test_acc_list = \
        new_target_cus_learnware_component
    new_target_cus_learnware = copy.deepcopy(target_cus_learnware)
    new_target_cus_learnware.args = args
    new_target_cus_learnware.state_dict = state_dict
    new_target_cus_learnware.epoch = epoch
    new_target_cus_learnware.train_loss_list = train_loss_list
    new_target_cus_learnware.train_acc_list = train_acc_list
    new_target_cus_learnware.test_acc_list = test_acc_list

    return new_target_cus_learnware



def get_learnware_paths(args, split):
    if split == 'repo':
        learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
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
    elif split == 'cus':
        pass

    target_learnware_paths = []
    for lw in os.listdir(learnware_save_dir):
        # epoch = 200
        # if lw.endswith('200.lw'):
        #     target_learnware_paths.append(learnware_save_dir + lw)
        
        # epoch = 50, 100, 150, 200
        target_learnware_paths.append(learnware_save_dir + lw)

    return target_learnware_paths



def load_learnware(path):
    with open(path, 'rb') as fp:
        learnware = pickle.load(fp)
    return learnware



def search_learnware(args, repo_learnwares, target_task):
    # generate text key
    if args.spec_type == 'text':
        fine_label = np.unique(target_task.fine_labels)

        with open(GV.dataset_path_prefix + 'specification2vec/spec2vec.dict', 'rb') as fo:
            spec2vec_dict = pickle.load(fo, encoding='bytes')
        with open(GV.dataset_path_prefix + self.args.data_name + '/' + 'fine_label2name.dict', 'rb') as fo:
            fine2name_dic = pickle.load(fo, encoding='bytes')
        key_text = np.zeros([1, len(list(spec2vec_dict.values())[0])])
        
        for i in fine_label:
            if fine2name_dic[i] not in spec2vec_dict.keys():
                part = fine2name_dic[i].split('_')
                miss = 0
                for j in part:
                    if j not in spec2vec_dict.keys():   
                        miss += 1
                        continue
                    key_text += (spec2vec_dict[j]) 
                key_text /= (len(part) - miss)
            else:
                key_text += spec2vec_dict[fine2name_dic[i]]
        key_text = key_text / len(fine_label)
    
    # generate linear key
    elif args.spec_type == 'linear':
        general_model_save_path = GV.save_path_prefix + args.data_name + '/general_models/' + \
                'general_model.record'
        with open(general_model_save_path, 'rb') as fp:
            record = pickle.load(fp)

        if args.data_name == 'CIFAR-100':
            Network = importlib.import_module('networks.' + 'resnet' + '_' + 'cifar')
            general_args = copy.copy(args)
            general_args.depth = 50
            network = Network.MyNetwork(general_args, n_classes=100)
        else:
            network = models.resnet50(pretrained=True)
            del network.fc
        
        network.load_state_dict(record['best_state_dict'])
        network = network.cuda()

        Data = importlib.import_module('dataloaders.' + args.data_name)
        data_path = GV.dataset_path_prefix + args.data_name + '/'
        target_data_loader = Data.generate_data_loader(data_path, target_task, 'train', 'simple',
                                                        args.batch_size, args.n_workers)

        for batch_index, batch in enumerate(target_data_loader):
            images, labels, _, _ = batch
            images = images.float().cuda()
            labels = labels.long().cuda()

            with torch.no_grad():
                embedding = network.forward(images, flag_embedding=True)
            
            if batch_index == 0:
                embedding_all = embedding
                label_all = labels
            else:
                embedding_all = torch.cat((embedding_all, embedding), dim=0)
                label_all = torch.cat((label_all, labels), dim=0)

        label_all = F.one_hot(label_all, target_task.get_n_fine_labels()).float()
        key_linear = torch.linalg.lstsq(embedding_all, label_all)[0]


    learnware_scores = []
    for i in range(len(repo_learnwares)):
        candidate_repo_learnware = repo_learnwares[i]
        if args.spec_type == 'text':
            learnware_scores.append(
                -1 * np.linalg.norm(
                    candidate_repo_learnware.specification_text - key_text
                )
            )
        elif args.spec_type == 'linear':
            W1 = candidate_repo_learnware.specification_linear
            W2 = key_linear

            S11 = torch.matmul(W1, W1.t())
            S12 = torch.matmul(W1, W2.t())
            S21 = torch.matmul(W2, W1.t())
            S22 = torch.matmul(W2, W2.t())

            learnware_scores.append(
                (torch.trace(torch.matmul(S12, S21)) / torch.sqrt(
                    torch.trace(torch.matmul(S11, S11)) * torch.trace(torch.matmul(S22, S22)))
                ).cpu().item()
            )

    learnware_scores = np.array(learnware_scores)
    selected_id = np.argmax(learnware_scores)

    return repo_learnwares[selected_id], learnware_scores, selected_id



def check_selected_repo_learnware(args, task, selected_repo_learnware):
    selected_repo_task = selected_repo_learnware.task

    print('===== target cus task =====')
    print('coarse labels:')
    print(task.coarse_labels)
    print('fine labels:')
    print(task.fine_labels)
    print('===== selected repo task =====')
    print('coarse labels:')
    print(selected_repo_task.coarse_labels)
    print('fine labels:')
    print(selected_repo_task.fine_labels)



def evaluate_one_task(args, task, task_list_file_name, task_id, learnware_save_dir):
    Data = importlib.import_module('dataloaders.' + args.data_name)
    data_path = GV.dataset_path_prefix + args.data_name + '/'
    train_data_loader = Data.generate_data_loader(data_path, task, 'train', 'augment',
                                                  args.batch_size, args.n_workers)
    test_data_loader = Data.generate_data_loader(data_path, task, 'test', 'simple',
                                                 args.batch_size, args.n_workers)
    print('===== task %d: datasets ready. =====' % (task_id))

    Network = importlib.import_module('networks.' + args.network_name)
    network = Network.MyNetwork(args, n_classes=task.get_n_fine_labels())
    network = network.cuda()
    print('===== task %d: network ready. =====' % (task_id))

    if args.method == 'raw':
        learnware_components = train(args, train_data_loader, test_data_loader, network)

    elif args.method in {'fitnet', 'rkd', 'ncm'}:
        selected_repo_learnware, learnware_scores, selected_id = search_learnware(
            args, repo_learnwares, task)

        # TODO
        # Based on retrain, implement ditillation




if __name__ == '__main__':
    # set random seed
    random.seed(648)
    np.random.seed(648)
    torch.manual_seed(648)
    torch.cuda.manual_seed(648)
    torch.backends.cudnn.deterministic = True

    # create a parser
    parser = argparse.ArgumentParser()

    # customer arguments
    parser.add_argument('--n_cus_tasks', type=int, default=1)
    parser.add_argument('--cus_min_coarse_label', type=int, default=0)
    parser.add_argument('--cus_max_coarse_label', type=int, default=19)
    parser.add_argument('--cus_min_fine_label', type=int, default=0)
    parser.add_argument('--cus_max_fine_label', type=int, default=199)
    parser.add_argument('--spec_type', type=str, default='linear', choices=['text', 'linear'])
    # task arguments
    parser.add_argument('--data_name', type=str, default='CIFAR-100', choices=['CIFAR-100', 'CUB-200', 'StanfordDogs', 'Car-196'])
    parser.add_argument('--network_name', type=str, default='wide_resnet', choices=['resnet', 'wide_resnet'])
    # network arguments
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--channel', type=int, default=0.25)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    # training procedure arguments
    parser.add_argument('--method', type=str, default='fitnet', choices=['raw', 'fitnet', 'rkd', 'ncm'])
    parser.add_argument('--n_training_epochs', type=int, default=200)
    parser.add_argument('--n_warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tau', type=float, default=3)
    parser.add_argument('--alpha', type=float, default=0.1) # weight of distillation term
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

    if args.data_name == 'CIFAR-100':
        assert args.cus_min_coarse_label >= 0
        assert args.cus_max_coarse_label <= 19
    elif args.data_name == 'CUB-200':
        args.cus_min_coarse_label = args.cus_min_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        args.batch_size = 128
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 199
    elif args.data_name == 'StanfordDogs':
        args.cus_min_coarse_label = args.cus_min_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        args.batch_size = 128
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 119
    elif args.data_name == 'Car-196':
        args.cus_min_coarse_label = args.cus_min_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        args.batch_size = 128
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 195
    else:
        pass

    display_args(args)

    # load repo learnwares
    repo_args = copy.copy(args)

    if repo_args.data_name == 'CIFAR-100':
        repo_args.repo_min_coarse_label, repo_args.repo_max_coarse_label = 0, 19
    elif repo_args.data_name == 'CUB-200':
        repo_args.repo_min_coarse_label, repo_args.repo_max_coarse_label = 0, 199
    elif repo_args.data_name == 'StanfordDogs':
        repo_args.repo_min_coarse_label, repo_args.repo_max_coarse_label = 0, 119
    elif repo_args.data_name == 'Car-196':
        repo_args.repo_min_coarse_label, repo_args.repo_max_coarse_label = 0, 195

    if repo_args.network_name == 'resnet':
        pass
    elif repo_args.network_name == 'wide_resnet':
        repo_args.depth = 40
        repo_args.width = 2
    elif repo_args.network_name == 'mobile_net':
        repo_args.channel = 1
    
    repo_args.learnware_checkpoint = [50, 100, 150, 200]

    repo_learnware_paths = get_learnware_paths(repo_args, 'repo')
    repo_learnwares = [load_learnware(path) for path in repo_learnware_paths]
    print('===== repo learnwares loaded. =====')

    cus_task_list_path = GV.dataset_path_prefix + args.data_name + '/task_lists/' + \
        '_minc=' + str(args.cus_min_coarse_label) + \
        '_maxc=' + str(args.cus_max_coarse_label) + \
        '.cus'
    with open(cus_task_list_path, 'rb') as fp:
        cus_task_list = pickle.load(fp)
    assert args.n_cus_tasks <= len(cus_task_list)

    if args.n_cus_tasks != 0:
        cus_learnware_save_dir = GV.save_path_prefix + args.data_name + '/learnwares/' + \
            '_minc=' + str(args.cus_min_coarse_label) + \
            '_maxc=' + str(args.cus_max_coarse_label) + \
            '_network=' + str(args.network_name) + \
            '_depth=' + str(args.depth) + \
            '_width=' + str(args.width) + \
            '_channel=' + str(args.channel) + \
            '_dropout=' + str(args.dropout_rate) + \
            '_nwarm=' + str(args.n_warmup_epochs) + \
            '_spec=' + str(args.spec_type) + \
            '_method=' + str(args.method) + \
            '_lr=' + str(args.lr) + \
            '_point=' + str(args.point) + \
            '_gamma=' + str(args.gamma) + \
            '_wd=' + str(args.wd) + \
            '_mo=' + str(args.mo) + \
            '_tau=' + str(args.tau) + \
            '_alpha=' + str(args.alpha) + \
            '.cus/'
        if not os.path.exists(cus_learnware_save_dir):
            os.makedirs(cus_learnware_save_dir)
        cus_task_list_file_name = \
            '_minc=' + str(args.cus_min_coarse_label) + \
            '_maxc=' + str(args.cus_max_coarse_label) + \
            '.cus'
        
        assert (args.id_server + 1) * args.n_cus_tasks <= len(cus_task_list)
        left = args.id_server * args.n_cus_tasks
        for i in range(left, left + args.n_cus_tasks):
            evaluate_one_task(args, cus_task_list[i], cus_task_list_file_name, 
                              i, cus_learnware_save_dir)
            print('%d of %d cus tasks complete.' % (i + 1, args.n_cus_tasks))


    # # iterate over cus learnwares
    # cus_args = copy.copy(args)
    # cus_args.method = 'raw'
    # cus_learnware_paths = get_learnware_paths(cus_args, 'cus')
    # for i in range(0, len(cus_learnware_paths)):
    #     with open(cus_learnware_paths[i], 'rb') as fp:
    #         target_cus_learnware = pickle.load(fp)
    #     target_cus_learnware_name = cus_learnware_paths[i].split('/')[-1]

    #     selected_repo_learnware, learnware_scores, selected_id = \
    #         search_learnware(args, repo_learnwares, target_cus_learnware)
        
    #     # check_selected_repo_learnware(args, target_cus_learnware, selected_repo_learnware)

    #     new_target_cus_learnware = retrain(args, selected_repo_learnware, target_cus_learnware)

    #     new_target_cus_learnware_save_path = new_cus_learnware_save_dir + target_cus_learnware_name
    #     with open(new_target_cus_learnware_save_path, 'wb') as fp:
    #         pickle.dump(new_target_cus_learnware, fp)
    #     print('%d of %d cus tasks retrained.' % (i + 1, len(cus_learnware_paths)))        