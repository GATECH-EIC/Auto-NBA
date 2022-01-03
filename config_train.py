# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
import genotypes
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'Auto-NBA'

C.world_size = 1  # num of nodes
C.multiprocessing_distributed = False
C.rank = 0  # node rank
C.dist_backend = 'nccl'
C.dist_url = 'tcp://eic-2019gpu5.ece.rice.edu:10001'  # url used to set up distributed training
# C.dist_url = 'tcp://127.0.0.1:10001'

C.gpu = None

C.dataset = 'imagenet'

if 'cifar' in C.dataset:
    """Data Dir and Weight Dir"""
    C.dataset_path = "/home/yf22/dataset/"

    if C.dataset == 'cifar10':
        C.num_classes = 10
    elif C.dataset == 'cifar100':
        C.num_classes = 100
    else:
        print('Wrong dataset.')
        sys.exit()

    """Image Config"""

    C.num_train_imgs = 50000
    C.num_eval_imgs = 10000

    """ Settings for network, this would be different for each kind of model"""
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1

    """Train Config"""

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 5e-4

    C.betas=(0.5, 0.999)
    C.num_workers = 4


    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/finetune'

    # C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    # C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    # C.stride_list = [1, 1, 2, 2, 1, 2, 1]

    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 1, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1504

    C.use_hswish = False
    C.use_se = False

    C.num_bits_list = 32  #  [4, 6, 8]

    C.operator_only = None
    C.bit_only = None

    if C.operator_only == True:
        C.num_bits_list = 32

    if C.bit_only == True:
        genotypes.PRIMITIVES = ['k3_e6']
        C.mode = 'soft'

    ########################################

    C.batch_size = 96
    C.niters_per_epoch = C.num_train_imgs // C.batch_size
    C.image_height = 32 # this size is after down_sampling
    C.image_width = 32

    C.save = "finetune"
    ########################################

    C.nepochs = 600

    C.eval_epoch = 1

    C.label_smoothing = False

    C.cutmix = False
    C.beta = 1
    C.cutmix_prob = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.01
    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [80, 120, 160]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    C.load_path = 'ckpt/search'

    C.eval_only = False

    C.efficiency_metric = 'flops'


elif C.dataset == 'imagenet':
    C.dataset_path = "/data1/ILSVRC/Data/CLS-LOC" # Specify path to ImageNet-1000

    C.num_workers = 4  # workers per gpu
    C.batch_size = 512

    C.num_classes = 1000

    """Image Config"""

    C.num_train_imgs = 1281167
    C.num_eval_imgs = 50000

    """ Settings for network, this would be different for each kind of model"""
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1

    """Train Config"""

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 4e-5

    C.betas=(0.5, 0.999)


    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/finetune'

    # C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    # C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    # C.stride_list = [1, 1, 2, 2, 1, 2, 1]

    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 2, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1984

    C.use_hswish = True
    C.use_se = True

    C.num_bits_list = 32  #  [4, 6, 8]

    C.operator_only = None
    C.bit_only = None

    if C.operator_only == True:
        C.num_bits_list = 32

    if C.bit_only == True:
        genotypes.PRIMITIVES = ['k3_e6']
        C.mode = 'soft'
        
    ########################################
    C.niters_per_epoch = C.num_train_imgs // C.batch_size
    C.image_height = 224 # this size is after down_sampling
    C.image_width = 224

    C.save = "finetune"
    ########################################

    # C.nepochs = 360
    C.nepochs = 180

    C.eval_epoch = 1

    C.autoaugment = False

    C.label_smoothing = False

    C.cutmix = False
    C.beta = 1
    C.cutmix_prob = 1

    # C.lr_schedule = 'multistep'
    # C.lr = 0.1

    C.lr_schedule = 'cosine'
    C.lr = 0.2

    # linear 
    C.decay_epoch = 100
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [90, 180, 270]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0

    C.load_path = 'ckpt/search'

    C.eval_only = False

    C.efficiency_metric = 'flops'


else:
    print('Wrong dataset.')
    sys.exit()