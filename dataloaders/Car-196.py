# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-04-14 15:48:19
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
import sys
import os
import scipy.io

sys.path.append("..")
from task_generator import Task
from utils import global_variable as GV

class MyDataset(Dataset):
    def __init__(self, data_path, task, split, augmentation):
        super(MyDataset, self).__init__()
        self.data_path = data_path
        self.task = task
        self.split = split
        self.augmentation = augmentation

        self.transform_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize([256, 256]),
            transforms.RandomResizedCrop([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_simple = transforms.Compose([
            transforms.Resize([224, 224]),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.transform_raw = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])

        self.features, self.fine_labels, self.coarse_labels = self.read_data()

    def read_data(self):
        if os.path.exists(self.data_path + 'data') == False:
            print('===== Reforming Data... =====')
            self.reform_data()
            
        data_file_path = self.data_path + 'data'
        if self.split == 'train':
            if self.task != 'all':
                indexes_needed = self.task.train_indexes
        elif self.split == 'test':
            if self.task != 'all':
                indexes_needed = self.task.test_indexes

        with open(data_file_path, 'rb') as fp:
            data = pickle.load(fp, encoding='bytes')
        features = np.array(data['features'])
        fine_labels = np.array(data['labels'])
        
        if self.task != 'all':
            features_needed = features[indexes_needed]
            fine_labels_needed = fine_labels[indexes_needed]
            coarse_labels_needed = fine_labels_needed   # In Car-196 we treat 'coarse' same to 'fine' 
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

    def reform_data(self):
        meta = scipy.io.loadmat(os.path.join(GV.dataset_path_prefix , 'Car-196/cars_annos.mat'))['annotations'][0]
        split = [i[0] for i in meta]
        labels = [i[5] for i in meta]
        labels = np.array([i[0][0] for i in labels])-1

        split = [item[0] for item in split]
        data = list(zip(split, labels))

        saves = {}
        data_list = []
        label_list = []
        for image_name, label in data:
            image_path = os.path.join(self.data_path, image_name)
            image = Image.open(image_path).convert('RGB')
            data_list.append(np.array(image))
            label_list.append(label)

        saves['features'] = np.array(data_list)
        saves['labels'] = np.array(label_list)
        with open(self.data_path + 'data', 'wb') as fp:
            data = pickle.dump(saves, fp)

    def __len__(self):
        return len(self.fine_labels)
    
    def __getitem__(self, index):
        feature = self.features[index]
        image = Image.fromarray(feature)

        # data preprocess
        raw_image = self.transform_raw(image)
        if self.augmentation == 'augment':
            aug_image = self.transform_augment(image)
        elif self.augmentation == 'simple':
            aug_image = self.transform_simple(image)
        else:
            aug_image = self.transform_simple(image)
        
        fine_label = self.fine_labels[index]
        coarse_label = self.coarse_labels[index]
        y = self.fine_label2y[fine_label]

        return aug_image, y, (fine_label, coarse_label), raw_image

    def get_n_classes(self):
        assert len(np.unique(self.fine_labels)) == len(self.task.fine_labels)
        return len(np.unique(self.fine_labels))



def generate_data_loader(data_path, task, split, augmentation, batch_size, n_workers):
    my_dataset = MyDataset(data_path, task, split, augmentation)
    my_data_loader = DataLoader(my_dataset, batch_size, shuffle=True, num_workers=n_workers)
    return my_data_loader



if __name__ == '__main__':
    data_path = GV.dataset_path_prefix + 'Car-196' + '/'
    n_workers = GV.WORKERS
    split = 'train'
    augmentation = 'simple'
    batch_size = 2

    task_list_file_path =  GV.dataset_path_prefix + 'Car-196' + '/task_lists/' + \
        '_minc=0_maxc=195.repo'
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
        print(raw_image.size())
        break

    my_data_loader = generate_data_loader(data_path, task, split, augmentation, batch_size, n_workers)
    for batch_index, batch in enumerate(my_data_loader):
        image, y, (fine_label, coarse_label), raw_image = batch
        print(image.size())
        print(y)
        print(fine_label)
        print(raw_image.size())
        break