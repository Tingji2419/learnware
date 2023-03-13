# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-02-14 15:42:31
"""

import pickle
from PIL import Image

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

import sys
sys.path.append("..")
from utils import global_variable as GV
from task_generator import Task

class MyDataset(Dataset):
    def __init__(self, data_path, task, split, augmentation):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.task = task
        self.split = split
        self.augmentation = augmentation

        self.transform_augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

        self.transform_simple = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

        self.transform_raw = transforms.Compose([
            transforms.ToTensor()
        ])

        self.features, self.fine_labels, self.coarse_labels = self.read_data()

    def read_data(self):
        if self.split == 'train':
            data_file_path = self.data_path + 'train'
            if self.task != 'all':
                indexes_needed = self.task.train_indexes
        elif self.split == 'test':
            data_file_path = self.data_path + 'test'
            if self.task != 'all':
                indexes_needed = self.task.test_indexes

        with open(data_file_path, 'rb') as fp:
            data = pickle.load(fp, encoding='bytes')
        features = np.array(data[b'data'])
        fine_labels = np.array(data[b'fine_labels'])
        coarse_labels = np.array(data[b'coarse_labels'])

        if self.task != 'all':
            features_needed = features[indexes_needed]
            fine_labels_needed = fine_labels[indexes_needed]
            coarse_labels_needed = coarse_labels[indexes_needed]
        else:
            features_needed = features
            fine_labels_needed = fine_labels
            coarse_labels_needed = coarse_labels

        self.fine_label2y = {}
        self.y2fine_label = {}
        current_y = 0
        fine_label_table = np.sort(np.unique(fine_labels_needed))
        for fine_label in fine_label_table:
            if fine_label in self.fine_label2y.keys():
                continue
            else:
                self.fine_label2y[fine_label] = current_y
                self.y2fine_label[current_y] = fine_label
                current_y += 1

        return features_needed, fine_labels_needed, coarse_labels_needed

    def __len__(self):
        return len(self.fine_labels)
    
    def __getitem__(self, index):
        feature = self.features[index, :]
        # reshape feature to the shape of (height, width, depth)
        feature_r = feature[:1024].reshape(32, 32)
        feature_g = feature[1024:2048].reshape(32, 32)
        feature_b = feature[2048:].reshape(32, 32)
        feature = np.dstack((feature_r, feature_g, feature_b))
        image = Image.fromarray(feature)

        # data preprocess
        raw_image = self.transform_raw(image)
        if self.augmentation == 'augment':
            aug_image = self.transform_augment(image)
        elif self.augmentation == 'simple':
            aug_image = self.transform_simple(image)
        
        fine_label = self.fine_labels[index]
        coarse_label = self.coarse_labels[index]
        y = self.fine_label2y[fine_label]

        return aug_image, y, (fine_label, coarse_label), raw_image

    def get_n_classes(self):
        if self.task != 'all':
            assert len(np.unique(self.fine_labels)) == len(self.task.fine_labels)
        return len(np.unique(self.fine_labels))



def generate_data_loader(data_path, task, split, augmentation, batch_size, n_workers):
    my_dataset = MyDataset(data_path, task, split, augmentation)
    my_data_loader = DataLoader(my_dataset, batch_size, shuffle=True, num_workers=n_workers)
    return my_data_loader



if __name__ == '__main__':
    data_path = GV.dataset_path_prefix + 'CIFAR-100' + '/'
    n_workers = GV.WORKERS
    split = 'train'
    augmentation = 'augment'
    batch_size = 2

    task_list_file_path =  GV.dataset_path_prefix + 'CIFAR-100' + '/task_lists/' + \
        '_minc=0_maxc=19.repo'
    with open(task_list_file_path, 'rb') as fp:
        task_list = pickle.load(fp)
    task = task_list[0]
    task.show()

    my_dataset = MyDataset(data_path, task, split, augmentation)
    print(my_dataset.__len__())
    print(my_dataset.get_n_classes())
    for index, (aug_image, y, (fine_label, coarse_label), raw_image) in enumerate(my_dataset):
        print(aug_image.size())
        print(y)
        print(fine_label)
        print(coarse_label)
        print(raw_image.size())
        break

    my_data_loader = generate_data_loader(data_path, task, split, augmentation, batch_size, n_workers)
    for batch_index, batch in enumerate(my_data_loader):
        image, y, (fine_label, coarse_label), raw_image = batch
        print(image.size())
        print(y)
        print(fine_label)
        print(coarse_label)
        print(raw_image.size())
        break