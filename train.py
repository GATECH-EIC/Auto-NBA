from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

import torchvision

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from tensorboardX import SummaryWriter

from config_train import config

from datasets import prepare_train_data, prepare_train_data_autoaugment, prepare_test_data

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_train import config
import genotypes

from model_search import FBNet as Network
from model_infer import FBNet_Infer

from lr import LambdaLR

from thop import profile
from thop.count_hooks import count_convNd

from quantize import QConv2d

custom_ops = {QConv2d: count_convNd}

import argparse

import operations
operations.USE_HSWISH = config.use_hswish
operations.USE_SE = config.use_se


parser = argparse.ArgumentParser(description='DNA')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to ImageNet-100')
parser.add_argument('-b', '--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers per gpu')
parser.add_argument('--world_size', type=int, default=None,
                    help='number of nodes')
parser.add_argument('--rank', type=int, default=None,
                    help='node rank')
parser.add_argument('--dist_url', type=str, default=None,
                    help='url used to set up distributed training')
args = parser.parse_args()


best_acc = 0
best_epoch = 0


def main():
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.world_size is not None:
        config.world_size = args.world_size
    if args.world_size is not None:
        config.rank = args.rank
    if args.dist_url is not None:
        config.dist_url = args.dist_url

    config.distributed = config.world_size > 1 or config.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    config.ngpus_per_node = ngpus_per_node
    config.num_workers = config.num_workers * ngpus_per_node

    if config.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.world_size = ngpus_per_node * config.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    global best_acc
    global best_epoch

    config.gpu = gpu
    pretrain = config.pretrain

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

    # logging.info("config = %s", str(config))
    # # preparation ################
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # seed = config.seed
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)


    if config.distributed:
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.rank = config.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        print("Rank: {}".format(config.rank))


    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
        if type(pretrain) == str:
            config.save = pretrain
        else:
            config.save = 'ckpt/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))

        logger = SummaryWriter(config.save)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        logging.info("args = %s", str(config))
    else:
        logger = None


    # Model #######################################

    state = torch.load(os.path.join(config.load_path, 'arch.pt'))
    alpha = state['alpha']
    beta = state['beta']

    # alpha = torch.zeros(sum(config.num_layer_list), len(genotypes.PRIMITIVES)).cuda()
    # alpha[:,0] = 10

    # if type(config.num_bits_list) is list:
    #     beta = torch.zeros(sum(config.num_layer_list), len(genotypes.PRIMITIVES), len(config.num_bits_list)).cuda()
    # else:
    #     beta = torch.zeros(sum(config.num_layer_list), len(genotypes.PRIMITIVES), 1).cuda()
    # beta[:,:,0] = 10   

    model = FBNet_Infer(alpha, beta, config=config)


    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
        flops, params = profile(model, inputs=(torch.randn(1, 3, config.image_height, config.image_width),), custom_ops=custom_ops)
        bitops = model.forward_bitops(size=(3, config.image_height, config.image_width))
        
        logging.info("params = %fM, FLOPs = %fM, BitOPs = %fG", params / 1e6, flops / 1e6, bitops / 1e9)

        if config.efficiency_metric == 'latency':
            fps, searched_hw = model.eval_latency(cifar='cifar' in config.dataset, iteration=100000, mode='random', fix_comp_mode=True, temp=1)
            logging.info("FPS of Searched Arch:" + str(fps))


    print('config.gpu:', config.gpu)
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()


    # for param, val in model.named_parameters():
    #     print(param, val.device)
        
    #     if val.device.type == 'cpu':
    #         print('This tensor is on CPU.')
    #         sys.exit()


    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ##############################
    # total_iteration = config.nepochs * config.niters_per_epoch
    
    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        print("Wrong Learning Rate Schedule Type.")
        sys.exit()

    cudnn.benchmark = True


    # if use multi machines, the pretrained weight and arch need to be duplicated on all the machines
    if type(pretrain) == str and os.path.exists(pretrain + "/weights_latest.pt"):
        pretrained_model = torch.load(pretrain + "/weights_latest.pt")
        partial = pretrained_model['state_dict']

        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        optimizer.load_state_dict(pretrained_model['optimizer'])
        lr_policy.load_state_dict(pretrained_model['lr_scheduler'])
        start_epoch = pretrained_model['epoch'] + 1

        best_acc = pretrained_model['best_acc']
        best_epoch = pretrained_model['best_epoch']

        print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)

    else:
        start_epoch = 0
        print('No checkpoint. Train from scratch.')


    # data loader ############################
    if 'cifar' in config.dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        if config.dataset == 'cifar10':
            train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
        elif config.dataset == 'cifar100':
            train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
            test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
        else:
            print('Wrong dataset.')
            sys.exit()


    elif config.dataset == 'imagenet':
        if config.autoaugment:
            train_data = prepare_train_data_autoaugment(dataset=config.dataset,
                                          datadir=config.dataset_path+'/train')   
        else:        
            train_data = prepare_train_data(dataset=config.dataset,
                                          datadir=config.dataset_path+'/train')
        test_data = prepare_test_data(dataset=config.dataset,
                                        datadir=config.dataset_path+'/val')

    else:
        print('Wrong dataset.')
        sys.exit()

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, num_workers=config.num_workers, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=config.num_workers)

    if config.eval_only:
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            logging.info('Eval: acc = %f', infer(0, model, test_loader, logger))
        sys.exit(0)

    # tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in range(start_epoch, config.nepochs):
        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            # tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
            logging.info("[Epoch %d/%d] lr=%f" % (epoch + 1, config.nepochs, optimizer.param_groups[0]['lr']))

        if config.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, optimizer, lr_policy, logger, epoch, config)
        torch.cuda.empty_cache()
        lr_policy.step()

        # if config.dataset == 'imagenet' and epoch < 250:
        #     eval_epoch = 10
        # else:
        #     eval_epoch = config.eval_epoch
        
        eval_epoch = config.eval_epoch

        #validation
        if (epoch+1) % eval_epoch == 0:
            # if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            #     tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))

            with torch.no_grad():
                acc = infer(epoch, model, test_loader, logger)

            if config.distributed:
                acc = reduce_tensor(acc, config.world_size)

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                logger.add_scalar('acc/val', acc, epoch)
                logging.info("Epoch:%d Acc:%.3f Best Acc:%.3f Best Epoch:%d" % (epoch, acc, best_acc, best_epoch))

                state = {}
                state['state_dict'] = model.state_dict()
                state['optimizer'] = optimizer.state_dict()
                state['lr_scheduler'] = lr_policy.state_dict()
                state['epoch'] = epoch 
                state['acc'] = acc
                state['best_acc'] = best_acc
                state['best_epoch'] = best_epoch

                torch.save(state, os.path.join(config.save, 'weights_%d.pt'%epoch))
                torch.save(state, os.path.join(config.save, 'weights_latest.pt'))

                # save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))
                # save(model, os.path.join(config.save, 'weights_latest.pt'))

    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
        torch.save(state, os.path.join(config.save, 'weights.pt'))
        # save(model, os.path.join(config.save, 'weights.pt'))



def train(train_loader, model, optimizer, lr_policy, logger, epoch, config):
    model.train()

    # bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    # pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    # dataloader_model = iter(train_loader)

    for step, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # input, target = dataloader_model.next()

        start_time = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        data_time = time.time() - start_time

        if config.label_smoothing:
            criterion = loss_label_smoothing
        else:
            criterion = model.module._criterion

        r = np.random.rand(1)
        if config.cutmix and config.beta > 0 and r < config.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(config.beta, config.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            logit = model(input)
            loss = criterion(logit, target_a) * lam + criterion(logit, target_b) * (1. - lam)

        else:
            logit = model(input)
            loss = criterion(logit, target)


        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        total_time = time.time() - start_time

        if step % 10 == 0:
            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % (epoch + 1, config.nepochs, step + 1, len(train_loader), loss.item(), total_time, data_time))
                logger.add_scalar('loss/train', loss, epoch*len(train_loader)+step)

    torch.cuda.empty_cache()
    del loss


def loss_label_smoothing(outputs, labels):
    """
    loss function for label smoothing regularization
    """
    alpha = 0.1
    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value= alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    loss = -torch.sum(log_prob * smoothed_labels) / N

    return loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def infer(epoch, model, test_loader, logger):
    model.eval()
    prec1_list = []

    for i, (input, target) in enumerate(test_loader):
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        output = model(input_var)
        prec1, = accuracy(output.data, target_var, topk=(1,))
        prec1_list.append(prec1)

    acc = sum(prec1_list)/len(prec1_list)

    return acc


def reduce_tensor(rt, n):
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
