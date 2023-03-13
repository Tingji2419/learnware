# -*- coding: utf-8 -*-
"""
@Author: Su Lu, Ting-Ji Huang

@Date: 2020-12-08 19:52:05
"""

import platform
import GPUtil
import os

# determine the operating system and the GPUs available currently

if platform.platform().startswith('Windows'):
    WORKERS = 0
    n_gpus = 1
elif platform.platform().startswith('Linux'):
    WORKERS = 8
    n_gpus = 1

gpu_list = GPUtil.getAvailable(order='memory', limit=100, maxLoad=1.1, maxMemory=1.1)
DEVICES = ''
for i in range(0, n_gpus):
    DEVICES += (str(gpu_list[i]) + ',')

if os.path.exists('/home/lus/DATASETS/learnware/'):
    dataset_path_prefix = '/home/lus/DATASETS/learnware/'
elif os.path.exists('/user/lusu/DATASETS/learnware'):
    dataset_path_prefix = '/user/lusu/DATASETS/learnware'
elif os.path.exists('/data/lus/DATASETS/learnware/'):
    dataset_path_prefix = '/data/lus/DATASETS/learnware/'
elif os.path.exists('/dev/shm/lus/DATASETS/learnware/'):
    dataset_path_prefix = '/dev/shm/lus/DATASETS/learnware/'
elif os.path.exists('/data/huangtj/DATASETS/learnware/'):
    dataset_path_prefix = '/data/huangtj/DATASETS/learnware/'

if os.path.exists('/home/lus/SAVES/learnware/'):
    save_path_prefix = '/home/lus/SAVES/learnware/'
elif os.path.exists('/user/lusu/SAVES/learnware/'):
    save_path_prefix = '/user/lusu/SAVES/learnware/'
elif os.path.exists('/data/huangtj/SAVES/learnware/'):
    save_path_prefix = '/data/huangtj/SAVES/learnware/'
