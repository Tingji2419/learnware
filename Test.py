# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-02-14 15:58:15
"""

import torch
from torch.nn import functional as F
from utils import global_variable as GV

def test(args, data_loader, network):
    acc = 0
    network.eval()
    for _, batch in enumerate(data_loader):
        images, labels, _, _ = batch
        images = images.float().cuda()
        labels = labels.long().cuda()
        
        with torch.no_grad():
            logits = network.forward(images)
        prediction = torch.argmax(logits, dim=1)
        acc += torch.sum((prediction == labels).float()).cpu().item()

    acc /= data_loader.dataset.__len__()
    return acc



def test_ncm(args, train_data_loader, test_data_loader, network):
    assert(train_data_loader.dataset.get_n_classes() == test_data_loader.dataset.get_n_classes())
    n_classes = train_data_loader.dataset.get_n_classes()
    n_dimension = network.fc.in_features
    class_center = torch.zeros((n_classes, n_dimension)).cuda()
    class_count = torch.zeros(n_classes).cuda()

    network.eval()
    for batch_index, batch in enumerate(train_data_loader):
        images, labels = batch
        images = images.float().cuda()
        labels = labels.long().cuda()
        
        with torch.no_grad():
            embeddings = network.forward(images, flag_embedding=True)
            for i in range(0, n_classes):
                index_of_class_i = (labels == i)
                class_center[i] += torch.sum(embeddings[index_of_class_i], dim=0)
                class_count[i] += index_of_class_i.size()[0]

    class_count = class_count.unsqueeze(1)
    class_center = class_center / class_count
    class_center = F.normalize(class_center, p=2, dim=1)

    accuracy = 0
    network.eval()
    for batch_index, batch in enumerate(test_data_loader):
        images, labels = batch
        images = images.float().cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            embeddings = network.forward(images, flag_embedding=True)
            logits = torch.mm(embeddings, class_center.t())
        prediction = torch.argmax(logits, dim=1)
        accuracy += torch.sum((prediction == labels).float()).cpu().item()

    accuracy /= test_data_loader.dataset.__len__()
    return accuracy