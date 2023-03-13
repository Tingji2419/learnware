# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-03-09 13:58:00
"""

import os
import pickle
import random
import argparse
import scipy.io
import numpy as np

from utils import global_variable as GV

class Task():
    def __init__(self, coarse_labels, fine_labels, train_indexes, test_indexes):
        self.coarse_labels = coarse_labels
        self.fine_labels = fine_labels
        self.train_indexes = train_indexes
        self.test_indexes = test_indexes
    
    def get_n_coarse_labels(self):
        return len(self.coarse_labels)
    
    def get_n_fine_labels(self):
        return len(self.fine_labels)
    
    def get_n_train_instances(self):
        return len(self.train_indexes)
    
    def get_n_test_instances(self):
        return len(self.test_indexes)
    
    def show(self):
        print('===== task information =====')
        print('n_coarse_labels = %d' % (self.get_n_coarse_labels()))
        print('coarse labels: %s' % (str(self.coarse_labels)))
        print('n_fine_labels = %d' % (self.get_n_fine_labels()))
        print('fine labels: %s' % (str(self.fine_labels)))
        print('n_train_instances = %d' % (self.get_n_train_instances()))
        print('n_train_instances_per_class = %d' % (self.get_n_train_instances() // self.get_n_fine_labels()))
        print('n_test_instances = %d' % (self.get_n_test_instances()))
        print('n_test_instances_per_class = %d' % (self.get_n_test_instances() // self.get_n_fine_labels()))
        print('============================')



def get_dicts(args):
    train_coarse2fine_dict_path = GV.dataset_path_prefix + args.data_name + '/train_coarse2fine.dict'
    train_fine2index_dict_path = GV.dataset_path_prefix + args.data_name + '/train_fine2index.dict'
    test_coarse2fine_dict_path = GV.dataset_path_prefix + args.data_name + '/test_coarse2fine.dict'
    test_fine2index_dict_path = GV.dataset_path_prefix + args.data_name + '/test_fine2index.dict'
    
    if os.path.exists(train_coarse2fine_dict_path) and \
        os.path.exists(train_fine2index_dict_path) and \
        os.path.exists(test_coarse2fine_dict_path) and \
        os.path.exists(test_fine2index_dict_path):
        
        with open(train_coarse2fine_dict_path, 'rb') as fp:
            train_coarse2fine_dict = pickle.load(fp)
        with open(train_fine2index_dict_path, 'rb') as fp:
            train_fine2index_dict = pickle.load(fp)
        with open(test_coarse2fine_dict_path, 'rb') as fp:
            test_coarse2fine_dict = pickle.load(fp)
        with open(test_fine2index_dict_path, 'rb') as fp:
            test_fine2index_dict = pickle.load(fp)

        return train_coarse2fine_dict, train_fine2index_dict, \
            test_coarse2fine_dict, test_fine2index_dict

    elif os.path.exists(train_fine2index_dict_path) and \
        os.path.exists(test_fine2index_dict_path):

        train_coarse2fine_dict = None
        test_coarse2fine_dict = None
        with open(train_fine2index_dict_path, 'rb') as fp:
            train_fine2index_dict = pickle.load(fp)
        with open(test_fine2index_dict_path, 'rb') as fp:
            test_fine2index_dict = pickle.load(fp)

        return train_coarse2fine_dict, train_fine2index_dict, \
            test_coarse2fine_dict, test_fine2index_dict

    else:
        data_path = GV.dataset_path_prefix + args.data_name + '/'
        train_data_file_path = data_path + 'train'
        test_data_file_path = data_path + 'test'

        if args.data_name == 'StanfordDogs': 
            train_fine_labels = scipy.io.loadmat(os.path.join(GV.dataset_path_prefix + args.data_name, 'train_list.mat'))['labels']
            test_fine_labels = scipy.io.loadmat(os.path.join(GV.dataset_path_prefix + args.data_name, 'test_list.mat'))['labels']
            
            train_fine_labels = np.array(train_fine_labels) - 1
            test_fine_labels = np.array(test_fine_labels) - 1
            train_coarse2fine_dict = None
            test_coarse2fine_dict = None

        elif args.data_name == 'Car-196':
            meta = scipy.io.loadmat(os.path.join(GV.dataset_path_prefix + args.data_name, 'cars_annos.mat'))['annotations']['class'][0]
            flag = scipy.io.loadmat(os.path.join(GV.dataset_path_prefix + args.data_name, 'cars_annos.mat'))['annotations']['test'][0]
            meta = np.array([i[0][0] for i in meta]) - 1

            train_fine2index_dict = {}
            test_fine2index_dict = {}
            for i in range(np.min(meta), np.max(meta) + 1):
                x = np.argwhere(meta == i)
                train_fine2index_dict[i] = np.array([i for i in x[:, 0] if flag[i] == 0])
                test_fine2index_dict[i] = np.array([i for i in x[:, 0] if flag[i] == 1])
            with open(train_fine2index_dict_path, 'wb') as fp:
                pickle.dump(train_fine2index_dict, fp)
            with open(test_fine2index_dict_path, 'wb') as fp:
                pickle.dump(test_fine2index_dict, fp)

            train_coarse2fine_dict = None
            test_coarse2fine_dict = None

            return train_coarse2fine_dict, train_fine2index_dict, \
                test_coarse2fine_dict, test_fine2index_dict           

        elif args.data_name == 'CIFAR-100':
            with open(train_data_file_path, 'rb') as fp:
                train_data = pickle.load(fp, encoding='bytes')
            with open(test_data_file_path, 'rb') as fp:
                test_data = pickle.load(fp, encoding='bytes')
            train_coarse_labels = np.array(train_data[b'coarse_labels'])
            train_fine_labels = np.array(train_data[b'fine_labels'])
            test_coarse_labels = np.array(test_data[b'coarse_labels'])
            test_fine_labels = np.array(test_data[b'fine_labels'])

            train_coarse2fine_dict = {}
            for i in range(np.min(train_coarse_labels), np.max(train_coarse_labels) + 1):
                x = np.argwhere(train_coarse_labels == i)
                train_coarse2fine_dict[i] = np.unique(train_fine_labels[x][:, 0])
            with open(train_coarse2fine_dict_path, 'wb') as fp:
                pickle.dump(train_coarse2fine_dict, fp)

            test_coarse2fine_dict = {}
            for i in range(np.min(test_coarse_labels), np.max(test_coarse_labels) + 1):
                x = np.argwhere(test_coarse_labels == i)
                test_coarse2fine_dict[i] = np.unique(test_fine_labels[x][:, 0])
            with open(test_coarse2fine_dict_path, 'wb') as fp:
                pickle.dump(test_coarse2fine_dict, fp)            
        
        elif args.data_name == 'CUB-200':
            with open(train_data_file_path, 'rb') as fp:
                train_data = pickle.load(fp, encoding='bytes')
            with open(test_data_file_path, 'rb') as fp:
                test_data = pickle.load(fp, encoding='bytes')
            train_fine_labels = np.array(train_data['labels'])
            test_fine_labels = np.array(test_data['labels'])
            train_coarse2fine_dict = None
            test_coarse2fine_dict = None

        else:
            pass

        train_fine2index_dict = {}
        for i in range(np.min(train_fine_labels), np.max(train_fine_labels) + 1):
            x = np.argwhere(train_fine_labels == i)
            train_fine2index_dict[i] = x[:, 0]
        with open(train_fine2index_dict_path, 'wb') as fp:
            pickle.dump(train_fine2index_dict, fp)
        
        test_fine2index_dict = {}
        for i in range(np.min(test_fine_labels), np.max(test_fine_labels) + 1):
            x = np.argwhere(test_fine_labels == i)
            test_fine2index_dict[i] = x[:, 0]
        with open(test_fine2index_dict_path, 'wb') as fp:
            pickle.dump(test_fine2index_dict, fp)

    return train_coarse2fine_dict, train_fine2index_dict, \
        test_coarse2fine_dict, test_fine2index_dict



def check_dicts(args, train_coarse2fine_dict, train_fine2index_dict,
                test_coarse2fine_dict, test_fine2index_dict):
    if args.data_name == 'CIFAR-100':
        assert(train_coarse2fine_dict is not None)
        assert(test_coarse2fine_dict is not None)
    else:
        assert(train_coarse2fine_dict is None)
        assert(test_coarse2fine_dict is None)
    
    assert(train_fine2index_dict is not None)
    assert(test_fine2index_dict is not None)

    print('data name: %s' % (args.data_name))
    print('train fine labels:')
    print(train_fine2index_dict.keys())
    print('test fine labels:')
    print(test_fine2index_dict.keys())

    for i in range(0, 5):
        print('number of train indexes of class %d:' % (i))
        print(len(train_fine2index_dict[i]))
        print('number of test indexes of class %d:' % (i))
        print(len(test_fine2index_dict[i]))
    
    n_train_instances = 0
    for i in range(0, train_fine2index_dict.__len__()):
        n_train_instances += len(train_fine2index_dict[i])
    n_test_instances = 0
    for i in range(0, test_fine2index_dict.__len__()):
        n_test_instances += len(test_fine2index_dict[i])
    print('n_train_instances: %d' % (n_train_instances))
    print('n_test_instances: %d' % (n_test_instances))



def get_spec2vec_dict(args):
    # generate spec2vec_dict
    if os.path.exists(GV.dataset_path_prefix + 'specification2vec/spec2vec.dict') : pass
    else:   
        with open(GV.dataset_path_prefix + 'specification2vec/glove.6B.50d.txt', 'r', encoding='utf8') as file:
            words_list = file.readlines()
            spec2vec_dict = dict()
            for line in words_list:
                parts = line.split()
                spec2vec_dict[parts[0]] = np.asarray(parts[1:], dtype='float32')
        with open(GV.dataset_path_prefix + 'specification2vec/spec2vec.dict', 'wb') as fp:
            pickle.dump(spec2vec_dict, fp)
    
    # generate fine_label2name.dict
    if os.path.exists(GV.dataset_path_prefix+args.data_name + '/fine_label2name.dict') : pass
    elif args.data_name == 'CIFAR-100':
        with open(GV.dataset_path_prefix + 'CIFAR-100/meta', 'rb') as fo:
            meta = pickle.load(fo, encoding='bytes')
        fine_label2name_dic = {index:key.decode() for index, key in enumerate(meta[b'fine_label_names'])}
        with open(GV.dataset_path_prefix + 'CIFAR-100/fine_label2name.dict', 'wb') as fp:
            pickle.dump(fine_label2name_dic, fp) 

    elif args.data_name == 'CUB-200':
        with open(GV.dataset_path_prefix + 'CUB-200/classes.txt','r') as fo:
            meta=[(line[:-1].split('.')[-1]) for line in fo]
        fine_label2name_dic = {index:key.lower() for index, key in enumerate(meta)}
        with open(GV.dataset_path_prefix + 'CUB-200/fine_label2name.dict', 'wb') as fp:
            pickle.dump(fine_label2name_dic, fp) 

    elif args.data_name == 'StanfordDogs':
        meta=scipy.io.loadmat(os.path.join(GV.dataset_path_prefix + args.data_name, 'file_list.mat'))['annotation_list']
        raw_list=[str(meta[i][0]) for i in range(len(meta))]
        name_list=np.unique([i[i.find('-') + 1:i.find('/')] for i in raw_list])
        fine_label2name_dic = {index:key.lower() for index, key in enumerate(name_list)}
        with open(GV.dataset_path_prefix + 'StanfordDogs/fine_label2name.dict', 'wb') as fp:
            pickle.dump(fine_label2name_dic, fp) 

    elif args.data_name == 'Car-196':
        meta = scipy.io.loadmat(os.path.join(GV.dataset_path_prefix + args.data_name, 'cars_annos.mat'))['class_names'][0]
        meta = [i[0].replace(' ', '_').replace('/','') for i in meta]
        fine_label2name_dic = {index:key.lower() for index, key in enumerate(meta)}
        with open(GV.dataset_path_prefix + 'Car-196/fine_label2name.dict', 'wb') as fp:
            pickle.dump(fine_label2name_dic, fp) 

    else: pass



def sample_one_task(args, minc, maxc, train_coarse2fine_dict, train_fine2index_dict,
                    test_coarse2fine_dict, test_fine2index_dict):
    coarse_label_lists = np.arange(minc, maxc + 1)
    _n_coarse_labels = np.random.randint(1, len(coarse_label_lists) + 1)
    selected_coarse_labels = np.random.choice(coarse_label_lists, _n_coarse_labels, replace=False)

    if args.data_name == 'CIFAR-100':
        selected_fine_labels = []
        for c in selected_coarse_labels:
            fine_label_lists = train_coarse2fine_dict[c]
            _n_fine_labels = np.random.randint(1 + int(_n_coarse_labels == 1), len(fine_label_lists) + 1)
            _selected_fine_labels = np.random.choice(fine_label_lists, _n_fine_labels, replace=False)
            selected_fine_labels += list(_selected_fine_labels)
        selected_fine_labels = np.stack(selected_fine_labels)
    else:
        # In CUB-200, Car-196, and StanfordDogs we treat 'coarse' same to 'fine' 
        selected_fine_labels = np.stack(selected_coarse_labels)

    selected_train_indexes = []
    selected_test_indexes = []
    target_n_train_indexes = np.random.randint(1, len(train_fine2index_dict[0]) + 1)
    for f in selected_fine_labels:
        _n_train_indexes = min(target_n_train_indexes, len(train_fine2index_dict[f]))
        train_index_lists = train_fine2index_dict[f]
        test_index_lists = test_fine2index_dict[f]
        _selected_train_indexes = np.random.choice(train_index_lists, _n_train_indexes, replace=False)
        _selected_test_indexes = test_index_lists
        selected_train_indexes += list(_selected_train_indexes)
        selected_test_indexes += list(_selected_test_indexes)
    selected_train_indexes = np.stack(selected_train_indexes)
    selected_test_indexes = np.stack(selected_test_indexes)

    return Task(selected_coarse_labels, selected_fine_labels,
                selected_train_indexes, selected_test_indexes)



def generate_tasks(args, train_coarse2fine_dict, train_fine2index_dict,
                   test_coarse2fine_dict, test_fine2index_dict):
        
    if os.path.exists(GV.dataset_path_prefix + args.data_name + '/task_lists/') == False:
        os.mkdir(GV.dataset_path_prefix + args.data_name + '/task_lists/')

    repo_task_list_path = GV.dataset_path_prefix + args.data_name + '/task_lists/' + \
        '_minc=' + str(args.repo_min_coarse_label) + \
        '_maxc=' + str(args.repo_max_coarse_label) + \
        '.repo'

    repo_task_list = []
    for i in range(0, args.n_repo_tasks):
        repo_task_list.append(
            sample_one_task(args, args.repo_min_coarse_label, args.repo_max_coarse_label,
                            train_coarse2fine_dict, train_fine2index_dict,
                            test_coarse2fine_dict, test_fine2index_dict)
        )

    with open(repo_task_list_path, 'wb') as fp:
        pickle.dump(repo_task_list, fp)

    cus_task_list_path = GV.dataset_path_prefix + args.data_name + '/task_lists/' + \
        '_minc=' + str(args.cus_min_coarse_label) + \
        '_maxc=' + str(args.cus_max_coarse_label) + \
        '.cus'
    cus_task_list = []
    for i in range(0, args.n_cus_tasks):
        cus_task_list.append(
            sample_one_task(args, args.cus_min_coarse_label, args.cus_max_coarse_label,
                            train_coarse2fine_dict, train_fine2index_dict,
                            test_coarse2fine_dict, test_fine2index_dict)
        )
    with open(cus_task_list_path, 'wb') as fp:
        pickle.dump(cus_task_list, fp)



def check_tasks(args):
    repo_task_list_path = GV.dataset_path_prefix + args.data_name + '/task_lists/' + \
        '_minc=' + str(args.repo_min_coarse_label) + \
        '_maxc=' + str(args.repo_max_coarse_label) + \
        '.repo'
    with open(repo_task_list_path, 'rb') as fp:
        repo_task_list = pickle.load(fp)

    cus_task_list_path = GV.dataset_path_prefix + args.data_name + '/task_lists/' + \
        '_minc=' + str(args.cus_min_coarse_label) + \
        '_maxc=' + str(args.cus_max_coarse_label) + \
        '.cus'
    with open(cus_task_list_path, 'rb') as fp:
        cus_task_list = pickle.load(fp)

    print(len(repo_task_list))
    print(len(cus_task_list))

    if args.data_name == 'CIFAR-100':
        repo_class_cnt = np.zeros(100)
        cus_class_cnt = np.zeros(100)
    elif args.data_name == 'CUB-200':
        repo_class_cnt = np.zeros(200)
        cus_class_cnt = np.zeros(200)
    elif args.data_name == 'StanfordDogs':
        repo_class_cnt = np.zeros(120)
        cus_class_cnt = np.zeros(120)
    elif args.data_name == 'Car-196':
        repo_class_cnt = np.zeros(196)
        cus_class_cnt = np.zeros(196)

    for i in range(len(repo_task_list)):
        task = repo_task_list[i]
        # print('===== This is repo task %d =====' % (i + 1))
        # print(task.get_n_coarse_labels())
        # print(task.get_n_fine_labels())
        # print(task.get_n_train_instances())
        # print(task.get_n_test_instances())
        repo_class_cnt[task.fine_labels] += 1

        if i == 49:
            break
    
    for i in range(len(cus_task_list)):
        task = cus_task_list[i]
        # print('===== This is cus task %d =====' % (i + 1))
        # print(task.get_n_coarse_labels())
        # print(task.get_n_fine_labels())
        # print(task.get_n_train_instances())
        # print(task.get_n_test_instances())
        cus_class_cnt[task.fine_labels] += 1
    
        if i == 9:
            break
    
    print(repo_class_cnt)
    print(cus_class_cnt)



if __name__ == '__main__':
    random.seed(648)
    np.random.seed(648)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='CUB-200', choices=['CIFAR-100', 'CUB-200', 'StanfordDogs', 'Car-196'])
    parser.add_argument('--n_repo_tasks', type=int, default=1000)
    parser.add_argument('--n_cus_tasks', type=int, default=1000)
    parser.add_argument('--repo_min_coarse_label', type=int, default=0)
    parser.add_argument('--repo_max_coarse_label', type=int, default=19)
    parser.add_argument('--cus_min_coarse_label', type=int, default=0)
    parser.add_argument('--cus_max_coarse_label', type=int, default=19)
    parser.add_argument('--repo_min_fine_label', type=int, default=0)
    parser.add_argument('--repo_max_fine_label', type=int, default=199)
    parser.add_argument('--cus_min_fine_label', type=int, default=0)
    parser.add_argument('--cus_max_fine_label', type=int, default=199)

    args = parser.parse_args()
    
    if args.data_name == 'CIFAR-100':
        assert args.repo_min_coarse_label >= 0
        assert args.repo_max_coarse_label <= 19
        assert args.cus_min_coarse_label >= 0
        assert args.cus_max_coarse_label <= 19
    elif args.data_name == 'CUB-200':
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 199
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 199
    elif args.data_name == 'StanfordDogs':
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 119
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 119
    elif args.data_name == 'Car-196':
        args.repo_max_coarse_label = args.repo_max_fine_label
        args.cus_max_coarse_label = args.cus_max_fine_label
        assert args.repo_min_fine_label >= 0
        assert args.repo_max_fine_label <= 195
        assert args.cus_min_fine_label >= 0
        assert args.cus_max_fine_label <= 195
    else:
        pass

    # In CUB-200, Car-186, and StanfordDogs, 'train/test_coarse2fine_dict' are 'None' 
    # train_coarse2fine_dict, train_fine2index_dict, \
    #     test_coarse2fine_dict, test_fine2index_dict = get_dicts(args)
    
    # check_dicts(args, train_coarse2fine_dict, train_fine2index_dict, 
    #             test_coarse2fine_dict, test_fine2index_dict) 

    # generate_tasks(args, train_coarse2fine_dict, train_fine2index_dict,
    #                test_coarse2fine_dict, test_fine2index_dict)

    # get_spec2vec_dict(args)

    check_tasks(args)