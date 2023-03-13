# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2022-02-14 15:58:10
"""

import os
import copy
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.nn import functional as F

from Test import test, test_ncm
from utils import global_variable as GV
from networks.loss import HintLoss, RKDLoss

def general_train(args, train_data_loader, test_data_loader, network):
    def train_an_epoch(epoch):
        network.train()
        
        train_loss = 0
        train_acc = 0
        for batch_index, batch in enumerate(train_data_loader):
            images, labels, _, _ = batch
            images = images.float().cuda()
            labels = labels.long().cuda()
            bs = labels.size()[0]

            logits = network.forward(images)

            train_loss_value = train_loss_function(logits, labels)

            optimizer.zero_grad()
            train_loss_value.backward()
            optimizer.step()

            prediction = torch.argmax(logits, dim=1)

            train_loss += train_loss_value.cpu().item() * images.size()[0]
            train_acc += torch.sum((prediction == labels).float()).cpu().item()

        train_loss /= train_data_loader.dataset.__len__()
        train_acc /= train_data_loader.dataset.__len__()
        test_acc = test(args, test_data_loader, network)

        return train_loss, train_acc, test_acc

    train_loss_function = nn.CrossEntropyLoss()

    optimizer = SGD(params=network.parameters(), lr=args.lr, weight_decay=args.wd,
                    momentum=args.mo, nesterov=True)
    
    if args.gamma != -1:
        scheduler = MultiStepLR(optimizer, args.point, args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimizer, args.n_training_epochs, 0.001 * args.lr)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    best_test_acc = 0
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, args.n_training_epochs + 1):
        if epoch <= args.n_warmup_epochs:
            lr = args.lr * (epoch / (args.n_warmup_epochs + 1))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step(epoch - 1)

        train_loss, train_acc, test_acc = train_an_epoch(epoch)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print('epoch %d finish: train_loss = %f, train_accuracy = %f, test_accuracy = %f' % (
                epoch, train_loss, train_acc, test_acc)
        )
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(network.state_dict())
    
    record = {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'test_acc': test_acc_list,
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'best_state_dict': best_state_dict,
    }

    return record



def train(args, train_data_loader, test_data_loader, network):
    def train_an_epoch(epoch):
        network.train()

        train_loss = 0
        train_acc = 0
        for batch_index, batch in enumerate(train_data_loader):
            images, labels, _, _ = batch
            images = images.float().cuda()
            labels = labels.long().cuda()
            bs = labels.size()[0]

            logits = network.forward(images)

            train_loss_value = train_loss_function(logits, labels)

            optimizer.zero_grad()
            train_loss_value.backward()
            optimizer.step()

            prediction = torch.argmax(logits, dim=1)

            train_loss += train_loss_value.cpu().item() * images.size()[0]
            train_acc += torch.sum((prediction == labels).float()).cpu().item()
        
        train_loss /= train_data_loader.dataset.__len__()
        train_acc /= train_data_loader.dataset.__len__()
        test_acc = test(args, test_data_loader, network)

        return train_loss, train_acc, test_acc

    train_loss_function = nn.CrossEntropyLoss()
    
    optimizer = SGD(params=network.parameters(), lr=args.lr, weight_decay=args.wd,
                    momentum=args.mo, nesterov=True)

    if args.gamma != -1:
        scheduler = MultiStepLR(optimizer, args.point, args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimizer, args.n_training_epochs, 0.001 * args.lr)

    learnware_components = []
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(1, args.n_training_epochs + 1):
        if epoch <= args.n_warmup_epochs:
            lr = args.lr * (epoch / (args.n_warmup_epochs + 1))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step(epoch - 1)

        train_loss, train_acc, test_acc = train_an_epoch(epoch)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        if args.flag_debug:
            print('epoch %d finish: train_loss = %f, train_accuracy = %f, test_accuracy = %f' % (
                epoch, train_loss, train_acc, test_acc
            ))

        if epoch in args.learnware_checkpoint:
            learnware_components.append((
                copy.deepcopy(network.state_dict()),
                epoch,
                copy.deepcopy(train_loss_list),
                copy.deepcopy(train_acc_list),
                copy.deepcopy(test_acc_list.copy())
            ))

    return learnware_components



def distill_train(args, train_data_loader, test_data_loader, teacher_network, student_network):
    learnware_components = []

    train_loss_function = nn.CrossEntropyLoss()
    if args.method == 'fitnet':
        guide_loss_function = HintLoss(s_shape=student_network.fc.in_features,
                                       t_shape=teacher_network.fc.in_features)
        guide_loss_function.cuda()
    elif args.method == 'rkd':
        guide_loss_function = RKDLoss()
        guide_loss_function.cuda()
    else:
        pass
    
    all_parameters = list(student_network.parameters()) + list(guide_loss_function.parameters())
    optimizer = SGD(params=all_parameters, lr=args.lr, weight_decay=args.wd,
        momentum=args.mo, nesterov=True)

    if args.gamma != -1:
        scheduler = MultiStepLR(optimizer, args.point, args.gamma)
    else:
        scheduler = CosineAnnealingLR(optimizer, args.n_training_epochs, 0.001 * args.lr)

    train_loss_list = []
    guide_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(1, args.n_training_epochs + 1):
        train_loss = 0
        guide_loss = 0
        train_acc = 0

        student_network.train()
        teacher_network.eval()

        for batch_index, batch in enumerate(train_data_loader):
            images, labels, _, _ = batch
            images = images.float().cuda()
            labels = labels.long().cuda()
            bs = labels.size()[0]

            logits, embeddings = student_network.forward(images, flag_both=True)
            with torch.no_grad():
                teacher_logits, teacher_embeddings = teacher_network(images, flag_both=True)

            train_loss_value = train_loss_function(logits, labels)
            guide_loss_value = guide_loss_function(embeddings, teacher_embeddings)
            total_loss_value = train_loss_value + args.alpha * guide_loss_value

            optimizer.zero_grad()
            total_loss_value.backward()
            optimizer.step()

            prediction = torch.argmax(logits, dim=1)

            train_loss += train_loss_value.cpu().item() * images.size()[0]
            guide_loss += guide_loss_value.cpu().item() * images.size()[0]
            train_acc += torch.sum((prediction == labels).float()).cpu().item()

        train_loss /= train_data_loader.dataset.__len__()
        train_loss_list.append(train_loss)
        guide_loss /= train_data_loader.dataset.__len__()
        guide_loss_list.append(guide_loss)
        train_acc /= train_data_loader.dataset.__len__()
        train_acc_list.append(train_acc)
        test_acc = test(args, test_data_loader, student_network)
        test_acc_list.append(test_acc)

        if args.flag_debug:
            print('epoch %d finish: train_loss = %f, guide_loss = %f, train_accuracy = %f, test_accuracy = %f' % (
                epoch, train_loss, args.alpha * guide_loss, train_acc, test_acc
            ))

        scheduler.step()
    
    final_learnware = (
        copy.deepcopy(student_network.state_dict()),
        args.n_training_epochs,
        copy.deepcopy(train_loss_list),
        copy.deepcopy(guide_loss_list),
        copy.deepcopy(train_acc_list),
        copy.deepcopy(test_acc_list.copy())
    )
    return final_learnware_component