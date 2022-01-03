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

C.dataset = 'cifar100'

if 'cifar' in C.dataset:
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
    C.num_workers = 8

    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/search'

    # C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    # C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    # C.stride_list = [1, 1, 2, 2, 1, 2, 1]

    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 1, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1504

    C.enable_skip = True
    if not C.enable_skip:
        if 'skip' in genotypes.PRIMITIVES:
            genotypes.PRIMITIVES.remove('skip')


    C.num_bits_list = [4,6,8] # [4,5,6,7,8]

    C.early_stop_by_skip = False

    C.perturb_alpha = False
    C.epsilon_alpha = 0.3

    C.optim_mode = 'bilevel' # 'onelevel'

    C.train_mode = 'iterwise'  # 'epochwise'

    C.sample_func = 'gumbel_softmax'
    C.temp_init = 5
    C.temp_decay = 0.975

    ## Gumbel Softmax settings for operator
    C.mode = 'proxy_hard'  # "soft", "hard", "fake_hard", "random_hard", "proxy_hard"
    if C.mode == 'soft':
        C.hard = False
    else:
        C.hard = True

    C.offset = True and C.mode == 'proxy_hard'

    C.act_num = 2

    C.bit_heterogenous_search = True  #If true, always use soft mode for updaing weight

    ## Gumbel Softmax settings for bitwidth
    C.mode_bit = 'proxy_hard'  # "soft", "hard", "fake_hard", "random_hard", "proxy_hard", "baseline"
    C.act_num_bit = 2

    if type(C.num_bits_list) == list:
        assert C.act_num_bit <= len(C.num_bits_list)

    C.offset_bit = True and C.mode_bit == 'proxy_hard' and type(C.num_bits_list) == list


    C.operator_only = None
    C.bit_only = None

    if C.operator_only == True:
        C.num_bits_list = 32

    if C.bit_only == True:
        genotypes.PRIMITIVES = ['k3_e6']
        C.mode = 'soft'


    C.pretrain_epoch = 30
    C.pretrain_aline = True

    if C.pretrain_aline:
        C.pretrain_mode = C.mode
        C.pretrain_act_num = C.act_num
    else:
        C.pretrain_mode = 'soft'
        C.pretrain_act_num = 1

    C.arch_one_hot_loss_weight = None
    C.arch_mse_loss_weight = None

    C.num_sample = 10

    C.update_hw_freq = 5

    C.hw_aware_nas = False
    ########################################

    C.batch_size = 64
    C.niters_per_epoch = C.num_train_imgs // 2 // C.batch_size
    C.image_height = 32 
    C.image_width = 32
    C.save = "search"

    C.nepochs = 90 + C.pretrain_epoch
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.025
    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0.001

    ########################################

    C.train_portion = 0.5  # 0.8

    C.unrolled = False

    C.arch_learning_rate = 3e-4

    C.arch_update_frec = 1

    # hardware cost
    C.efficiency_metric = 'latency' # 'flops'

    # hardware cost weighted coefficients
    C.enable_mix_lr = False
    C.alpha_weight = 1
    C.beta_weight = 1

    # latency, customized for single-path FPGA predictor
    C.latency_weight = 1e-10  # 1e-7 - 1e-14
    C.fps_max = 400
    C.fps_min = 100

    C.hw_update_freq = 100   # update hw every xx arch param update
    C.hw_update_iter = 2000
    C.hw_update_mode = 'differentiable' # 'random'
    C.hw_update_fix_comp_mode = True
    C.hw_update_temp = 1

    # FLOPs
    C.flops_mode = 'single_path' # 'single_path', 'multi_path'
    C.flops_weight = 0
    C.flops_max = 3e8
    C.flops_min = 5e7
    C.flops_decouple = False



elif C.dataset == 'imagenet':
    """Data Dir and Weight Dir"""
    C.dataset_path = "/data1/ILSVRC/Data/CLS-LOC" # Specify path to ImageNet-100
    C.batch_size = 192
    C.num_workers = 16

    """Image Config"""

    C.num_classes = 100

    C.num_train_imgs = 128000
    C.num_eval_imgs = 50000

    """ Settings for network, this would be different for each kind of model"""
    C.bn_eps = 1e-5
    C.bn_momentum = 0.1

    """Train Config"""

    C.opt = 'Sgd'

    C.momentum = 0.9
    C.weight_decay = 5e-4

    C.betas=(0.5, 0.999)

    """ Search Config """
    C.grad_clip = 5

    C.pretrain = 'ckpt/search'

    # C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    # C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    # C.stride_list = [1, 1, 2, 2, 1, 2, 1]

    C.num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    C.num_channel_list = [16, 24, 32, 64, 112, 184, 352]
    C.stride_list = [1, 2, 2, 2, 1, 2, 1]

    C.stem_channel = 16
    C.header_channel = 1984


    C.enable_skip = True
    if not C.enable_skip:
        if 'skip' in genotypes.PRIMITIVES:
            genotypes.PRIMITIVES.remove('skip')


    C.num_bits_list = 32  #  [4, 6, 8]

    C.early_stop_by_skip = False

    C.perturb_alpha = False
    C.epsilon_alpha = 0.3

    C.optim_mode = 'bilevel' # 'onelevel'

    C.train_mode = 'iterwise'  # 'epochwise'

    C.sample_func = 'gumbel_softmax'
    C.temp_init = 5
    C.temp_decay = 0.956

    ## Gumbel Softmax settings for operator
    C.mode = 'proxy_hard'  # "soft", "hard", "fake_hard", "random_hard", "proxy_hard"
    if C.mode == 'soft':
        C.hard = False
    else:
        C.hard = True

    C.offset = True and C.mode == 'proxy_hard'

    C.act_num = 2

    C.bit_heterogenous_search = True  #If True, always use soft mode for updaing weight

    ## Gumbel Softmax settings for bitwidth
    C.mode_bit = 'proxy_hard'  # "soft", "hard", "fake_hard", "random_hard", "proxy_hard", "baseline"
    C.act_num_bit = 2

    if type(C.num_bits_list) == list:
        assert C.act_num_bit <= len(C.num_bits_list)

    C.offset_bit = True and C.mode_bit == 'proxy_hard' and type(C.num_bits_list) == list


    C.operator_only = None
    C.bit_only = None

    if C.operator_only == True:
        C.num_bits_list = 32

    if C.bit_only == True:
        genotypes.PRIMITIVES = ['k3_e6']
        C.mode = 'soft'


    C.pretrain_epoch = 45
    C.pretrain_aline = True

    if C.pretrain_aline:
        C.pretrain_mode = C.mode
        C.pretrain_act_num = C.act_num
    else:
        C.pretrain_mode = 'soft'
        C.pretrain_act_num = 1

    C.arch_one_hot_loss_weight = None
    C.arch_mse_loss_weight = None

    C.num_sample = 10

    C.update_hw_freq = 5

    C.hw_aware_nas = False
    ########################################

    C.niters_per_epoch = int(C.num_train_imgs // C.batch_size * 0.8)
    C.image_height = 224
    C.image_width = 224
    C.save = "search"

    C.nepochs = 75 + C.pretrain_epoch
    C.eval_epoch = 1

    C.lr_schedule = 'cosine'
    C.lr = 0.1
    # linear 
    C.decay_epoch = 20
    # exponential
    C.lr_decay = 0.97
    # multistep
    C.milestones = [50, 100, 200]
    C.gamma = 0.1
    # cosine
    C.learning_rate_min = 0


    ########################################

    C.train_portion = 0.8

    C.unrolled = False

    C.arch_learning_rate = 3e-4

    C.arch_update_frec = 1

    # hardware cost
    C.efficiency_metric = 'flops' # 'flops'

    # hardware cost weighted coefficients
    C.enable_mix_lr = False
    C.alpha_weight = 0.7
    C.beta_weight = 0.3

    # latency, customized for single-path FPGA predictor
    C.latency_weight = 0
    C.fps_max = 200
    C.fps_min = 0

    C.hw_update_freq = 20   # update hw every xx arch param update
    C.hw_update_iter = 10000
    C.hw_update_mode = 'random' # 'differentiable'
    C.hw_update_fix_comp_mode = True
    C.hw_update_temp = 1

    # FLOPs
    C.flops_mode = 'single_path' # 'single_path', 'multi_path'
    C.flops_weight = 0
    C.flops_max = 6e8
    C.flops_min = 1e8
    C.flops_decouple = False


else:
    print('Wrong dataset.')
    sys.exit()