# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2022-07-03 18:29:48
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

class Learnware():
    def __init__(self, args, task_list_file_name, task_id, task,
                 state_dict, epoch, train_loss_list, train_acc_list, test_acc_list):
        self.args = args
        self.task_list_file_name = task_list_file_name
        self.task_id = task_id
        self.task = task

        self.state_dict = state_dict
        self.epoch = epoch
        self.train_loss_list = train_loss_list
        self.train_acc_list = train_acc_list
        self.test_acc_list = test_acc_list
    
    def generate_text_specification(self):
        fine_label = np.unique(self.task.fine_labels)

        with open(GV.dataset_path_prefix + 'specification2vec/spec2vec.dict', 'rb') as fo:
            spec2vec_dict = pickle.load(fo, encoding='bytes')
        with open(GV.dataset_path_prefix + self.args.data_name + '/' + 'fine_label2name.dict', 'rb') as fo:
            fine2name_dic = pickle.load(fo, encoding='bytes')
        self.specification_text = np.zeros([1, len(list(spec2vec_dict.values())[0])])
        for i in fine_label:
            if fine2name_dic[i] not in spec2vec_dict.keys():
                part = fine2name_dic[i].split('_')
                miss = 0
                for j in part:
                    if j not in spec2vec_dict.keys():   
                        miss += 1
                        continue
                    self.specification_text += (spec2vec_dict[j]) 
                self.specification_text /= (len(part) - miss)
            else:
                self.specification_text += spec2vec_dict[fine2name_dic[i]]
        self.specification_text = self.specification_text / len(fine_label)

    def generate_linear_specification(self, general_data_loader, general_embedding):
        # build network
        Network = importlib.import_module('networks.' + self.args.network_name)
        network = Network.MyNetwork(self.args, self.task.get_n_fine_labels())
        network.load_state_dict(self.state_dict)
        network.cuda()
        network.eval()

        # get pseudo label
        for batch_index, batch in enumerate(general_data_loader):
            images, labels, _, _ = batch
            images = images.float().cuda()
            labels = labels.long().cuda()

            with torch.no_grad():
                logits = network(images)
                pseudo_label = torch.argmax(logits, dim=1)
            
            if batch_index == 0:
                pseudo_label_all = pseudo_label
            else:
                pseudo_label_all = torch.cat((pseudo_label_all, pseudo_label), 0)
        pseudo_label_all = F.one_hot(pseudo_label_all, self.task.get_n_fine_labels()).float()

        # get embedding
        general_embedding = torch.from_numpy(general_embedding).float().cuda()

        self.specification_linear = torch.linalg.lstsq(general_embedding, pseudo_label_all)[0]

    def show(self):
        print('===== learnware information =====')
        print('task_list_file_name = %s' % (self.task_list_file_name))
        print('task_id = %d' % (self.task_id))
        self.task.show()
        print('epoch = %d' % (self.epoch))
        print('train_acc = %f' % (self.train_acc_list[-1]))
        print('test_acc = %f' % (self.test_acc_list[-1]))
        print('=================================')