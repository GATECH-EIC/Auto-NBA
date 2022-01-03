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

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from datasets import prepare_train_data, prepare_test_data, prepare_train_data_for_search, prepare_test_data_for_search

import time

from tensorboardX import SummaryWriter

from config_search import config

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from architect import Architect
from model_search import FBNet as Network
from model_infer import FBNet_Infer

from lr import LambdaLR
from perturb import Random_alpha

import argparse

from thop import profile
from thop.count_hooks import count_convNd

from quantize import QConv2d

custom_ops = {QConv2d: count_convNd}


parser = argparse.ArgumentParser(description='DNA')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to ImageNet-100')
parser.add_argument('-b', '--batch_size', type=int, default=None,
                    help='batch size')
parser.add_argument('--num_workers', type=int, default=None,
                    help='number of workers')
parser.add_argument('--flops_weight', type=float, default=None,
                    help='weight of FLOPs loss')
parser.add_argument('--gpu', nargs='+', type=int, default=None,
                    help='specify gpus')
parser.add_argument('--world_size', type=int, default=None,
                    help='number of nodes')
parser.add_argument('--rank', type=int, default=None,
                    help='node rank')
parser.add_argument('--dist_url', type=str, default=None,
                    help='url used to set up distributed training')
parser.add_argument('--seed', type=int, default=12345,
                    help='random seed')
args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)


def main():
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.flops_weight is not None:
        config.flops_weight = args.flops_weight
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
    config.gpu = gpu
    pretrain = config.pretrain

    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))

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


    model = Network(config=config)

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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()


    architect = Architect(model, config)

    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.module.stem.parameters())
    parameters += list(model.module.cells.parameters())
    parameters += list(model.module.header.parameters())
    parameters += list(model.module.fc.parameters())
    
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        print("Wrong Optimizer Type.")
        sys.exit()

    if config.optim_mode == 'onelevel':
        architect.set_weight_optimizer(optimizer)

    # lr policy ##############################
    total_iteration = config.nepochs * config.niters_per_epoch
    
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


    # if use multi machines, the pretrained weight and arch need to be duplicated on all the machines
    if type(pretrain) == str and os.path.exists(pretrain + "/weights_latest.pt"):
        partial = torch.load(pretrain + "/weights_latest.pt")
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        pretrain_arch = torch.load(pretrain + "/arch_latest.pt")
        
        model.module.alpha.data = pretrain_arch['alpha'].data
        model.module.beta.data = pretrain_arch['beta'].data
        start_epoch = pretrain_arch['epoch'] + 1

        optimizer.load_state_dict(pretrain_arch['optimizer'])
        lr_policy.load_state_dict(pretrain_arch['lr_scheduler'])
        architect.optimizer.load_state_dict(pretrain_arch['arch_optimizer'])

        print('Resume from Epoch %d. Load pretrained weight and arch.' % start_epoch)
    else:
        start_epoch = 0
        print('No checkpoint. Search from scratch.')


    # # data loader ###########################
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
        train_data = prepare_train_data_for_search(dataset=config.dataset,
                                          datadir=config.dataset_path+'/train', num_class=config.num_classes)
        test_data = prepare_test_data_for_search(dataset=config.dataset,
                                        datadir=config.dataset_path+'/val', num_class=config.num_classes)

    else:
        print('Wrong dataset.')
        sys.exit()


    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_portion * num_train))

    if config.distributed:
        if config.optim_mode == 'onelevel':
            train_data_model = train_data
            train_sampler_model = torch.utils.data.distributed.DistributedSampler(train_data_model)
        else:
            train_data_model = torch.utils.data.Subset(train_data, indices[:split])
            train_data_arch = torch.utils.data.Subset(train_data, indices[split:num_train])

            train_sampler_model = torch.utils.data.distributed.DistributedSampler(train_data_model)
            train_sampler_arch = torch.utils.data.distributed.DistributedSampler(train_data_arch)
    else:
        if config.optim_mode == 'onelevel':
            train_sampler_model = None
        else:
            train_sampler_model = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
            train_sampler_arch = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train])

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, 
        sampler=train_sampler_model, shuffle=(train_sampler_model is None),
        pin_memory=False, num_workers=config.num_workers, drop_last=True)

    if config.optim_mode == 'onelevel':
        train_loader_arch = None
    else:
        train_loader_arch = torch.utils.data.DataLoader(
            train_data, batch_size=config.batch_size,
            sampler=train_sampler_arch, shuffle=(train_sampler_arch is None),
            pin_memory=False, num_workers=config.num_workers, drop_last=True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=config.num_workers)


    # tbar = tqdm(range(config.nepochs), ncols=80)

    for epoch in range(start_epoch, config.nepochs):
        if config.distributed:
            train_sampler_model.set_epoch(epoch)
            train_sampler_arch.set_epoch(epoch)

        if config.perturb_alpha:
            epsilon_alpha = 0.03 + (config.epsilon_alpha - 0.03) * epoch / config.nepochs

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                logging.info('Epoch %d epsilon_alpha %e', epoch, epsilon_alpha)
        else:
            epsilon_alpha = 0

        if epoch < config.pretrain_epoch:
            update_arch = False
            model.module.set_search_mode(mode=config.pretrain_mode, act_num=config.pretrain_act_num)

        else:
            model.module.set_search_mode(mode=config.mode, act_num=config.act_num)

            if config.train_mode == 'iterwise':
                update_arch = True
            else:
                update_arch = (epoch % 2 == 1)

        # if update_arch:
        #     temp = config.temp_init * config.temp_decay ** (epoch - config.pretrain_epoch)
        # else:
        #     temp = config.temp_init

        temp = config.temp_init * config.temp_decay ** epoch

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            logging.info("Temperature: " + str(temp))
            logging.info("[Epoch %d/%d] lr=%f" % (epoch + 1, config.nepochs, optimizer.param_groups[0]['lr']))
            logging.info("update arch: " + str(update_arch))


        if config.optim_mode == 'onelevel':
            print('One-Level Optimization')
            model.module.set_stage(stage='update_arch')
            
            train_onelevel(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, 
                update_arch=update_arch, epsilon_alpha=epsilon_alpha, temp=temp)
        else:
            print('Bi-Level Optimization')

            if config.train_mode == 'iterwise':
                train_iterwise(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, 
                    update_arch=update_arch, epsilon_alpha=epsilon_alpha, temp=temp, arch_update_frec=config.arch_update_frec)
            else:
                train_epochwise(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, 
                    update_arch=update_arch, epsilon_alpha=epsilon_alpha, temp=temp)      

        lr_policy.step()
        torch.cuda.empty_cache()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))
                save(model, os.path.join(config.save, 'weights_latest.pt'))

            if pretrain == True:
                acc = infer(epoch, model, test_loader, logger, temp=temp)

                if config.distributed:
                    acc = reduce_tensor(acc, config.world_size)

                if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                    logger.add_scalar('acc/val', acc, epoch)
                    logging.info("Epoch %d: acc %.3f"%(epoch, acc))

            else:
                acc, metric = infer(epoch, model, test_loader, logger, temp=temp, finalize=True)

                if config.distributed:
                    acc = reduce_tensor(acc, config.world_size)

                if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                    logger.add_scalar('acc/val', acc, epoch)
                    logging.info("Epoch %d: acc %.3f"%(epoch, acc))

                    state = {}
                    
                    if config.efficiency_metric == 'flops':
                        logger.add_scalar('flops/val', metric, epoch)
                        logging.info("Epoch %d: FLOPs %.3f"%(epoch, metric))
                        state['flops'] = metric

                    # For latency aware search, the returned metris FPS
                    if config.efficiency_metric == 'latency':
                        logger.add_scalar('fps/val', metric, epoch)
                        logging.info("Epoch %d: FPS %.3f"%(epoch, metric))
                        state['fps'] = metric

                    state['alpha'] = getattr(model.module, 'alpha')
                    state['beta'] = getattr(model.module, 'beta')
                    state['acc'] = acc
                    state['epoch'] = epoch

                    state['optimizer'] = optimizer.state_dict()
                    state['lr_scheduler'] = lr_policy.state_dict()

                    state['arch_optimizer'] = architect.optimizer.state_dict()

                    torch.save(state, os.path.join(config.save, "arch_%d.pt"%(epoch)))
                    torch.save(state, os.path.join(config.save, "arch_latest.pt"))

                if config.efficiency_metric == 'flops':
                    if config.flops_weight > 0 and update_arch:
                        if metric < config.flops_min:
                            architect.flops_weight /= 2
                        elif metric > config.flops_max:
                            architect.flops_weight *= 2

                        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                            logger.add_scalar("arch/flops_weight", architect.flops_weight, epoch+1)
                            logging.info("arch_flops_weight = " + str(architect.flops_weight))


                # For latency aware search, the returned metris FPS
                elif config.efficiency_metric == 'latency':
                    if config.latency_weight > 0 and update_arch:
                        if metric < config.fps_min:
                            architect.latency_weight *= 2
                        elif metric > config.fps_max:
                            architect.latency_weight /= 2

                        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                            logger.add_scalar("arch/latency_weight", architect.latency_weight, epoch+1)
                            logging.info("arch_latency_weight = " + str(architect.latency_weight))

        if config.early_stop_by_skip and update_arch:
            groups = config.num_layer_list[1:-1]
            num_block = groups[0]

            current_arch = getattr(model.module, 'alpha').data[1:-1].argmax(-1)

            early_stop = False

            for group_id in range(len(groups)):
                num_skip = 0
                for block_id in range(num_block):
                    if current_arch[group_id * num_block + block_id] == 8:
                        num_skip += 1
                if num_skip >= 2:
                    early_stop = True

            if early_stop:
                print('Early Stop at epoch %d.' % epoch)
                break


    if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
        if update_arch:
            torch.save(state, os.path.join(config.save, "arch.pt"))

        if config.efficiency_metric == 'latency':
            model_infer = FBNet_Infer(getattr(model.module, 'alpha'), getattr(model.module, 'beta'), config=config)

            fps, searched_hw = model_infer.eval_latency(cifar='cifar' in config.dataset, iteration=100000, mode=config.hw_update_mode,
                                             fix_comp_mode=config.hw_update_fix_comp_mode, temp=config.hw_update_temp, hardware=model.module.searched_hw if config.hw_aware_nas else None)

            opt_hw_final = {'opt_hw': searched_hw, 'FPS': fps}
            torch.save(opt_hw_final, os.path.join(config.save, "opt_hw.pt"))

            flops, params = profile(model_infer, inputs=(torch.randn(1, 3, config.image_height, config.image_width),), custom_ops=custom_ops)
            bitops = model_infer.forward_bitops(size=(3, config.image_height, config.image_width))
            
            logging.info("params = %fM, FLOPs = %fM, BitOPs = %fG", params / 1e6, flops / 1e6, bitops / 1e9)
            logging.info("FPS of Final Arch: %f", fps)



def train_iterwise(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True, epsilon_alpha=0, temp=1, arch_update_frec=1):
    model.train()

    # bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    # pbar = tqdm(range(len(train_loader_model)), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)
    dataloader_arch = iter(train_loader_arch)

    for step in range(len(train_loader_model)):
        start_time = time.time()

        input, target = dataloader_model.next()

        data_time = time.time() - start_time

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)


        if update_arch and step % arch_update_frec == 0:
            # pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_arch)))

            try:
                input_search, target_search = dataloader_arch.next()
            except:
                dataloader_arch = iter(train_loader_arch)
                input_search, target_search = dataloader_arch.next()

            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            # if config.bit_heterogenous_search:
            model.module.set_stage(stage='update_arch')

            loss_arch = architect.step(input_search, target_search, temp=temp)

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                # if (step+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(train_loader_arch)+step)
                if config.efficiency_metric == 'flops':
                    logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch*len(train_loader_arch)+step)
                elif config.efficiency_metric == 'latency':
                    logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(train_loader_arch)+step)

        if epsilon_alpha and update_arch:
            Random_alpha(model, epsilon_alpha)

        # if config.bit_heterogenous_search:
        model.module.set_stage(stage='update_weight')

        logit = model(input, temp)
        loss = model.module._criterion(logit, target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_time = time.time() - start_time

        if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
            if (step+1) % 10 == 0:
                logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % 
                            (epoch + 1, config.nepochs, step + 1, len(train_loader_model), loss.item(), total_time, data_time))
                logger.add_scalar('loss_weight/train', loss, epoch*len(train_loader_model)+step)


    torch.cuda.empty_cache()
    del loss
    if update_arch: del loss_arch


def train_epochwise(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True, epsilon_alpha=0, temp=1):
    model.train()

    if update_arch:
        dataloader_arch = iter(train_loader_arch)

        for step in range(len(dataloader_arch)):
            start_time = time.time()

            try:
                input_search, target_search = dataloader_arch.next()
            except:
                dataloader_arch = iter(train_loader_arch)
                input_search, target_search = dataloader_arch.next()

            data_time = time.time() - start_time

            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            loss_arch = architect.step(input_search, target_search, temp=temp)

            total_time = time.time() - start_time

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                if (step+1) % 10 == 0:
                    logger.add_scalar('loss_arch/train', loss_arch, epoch*len(train_loader_arch)+step)
                    if config.efficiency_metric == 'flops':
                        logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch*len(train_loader_arch)+step)
                    elif config.efficiency_metric == 'latency':
                        logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(train_loader_arch)+step)

                    logging.info("[Epoch %d/%d][Step %d/%d] Loss_arch=%.3f Time=%.3f Data Time=%.3f" % 
                                (epoch + 1, config.nepochs, step + 1, len(train_loader_arch), loss_arch.item(), total_time, data_time))

            torch.cuda.empty_cache()
            del loss_arch

    else:
        dataloader_model = iter(train_loader_model)

        for step in range(len(train_loader_model)):
            start_time = time.time()

            input, target = dataloader_model.next()

            data_time = time.time() - start_time

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            if epsilon_alpha and update_arch:
                Random_alpha(model, epsilon_alpha)

            logit = model(input, temp)
            loss = model.module._criterion(logit, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_time = time.time() - start_time

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                if (step+1) % 10 == 0:
                    logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % 
                                (epoch + 1, config.nepochs, step + 1, len(train_loader_model), loss.item(), total_time, data_time))
                    logger.add_scalar('loss_weight/train', loss, epoch*len(train_loader_model)+step)

            torch.cuda.empty_cache()
            del loss


def train_onelevel(train_loader_model, train_loader_arch, model, architect, optimizer, lr_policy, logger, epoch, update_arch=True, epsilon_alpha=0, temp=1):
    model.train()

    if update_arch:
        dataloader_arch = iter(train_loader_model)

        for step in range(len(dataloader_arch)):
            start_time = time.time()

            input_search, target_search = dataloader_arch.next()

            data_time = time.time() - start_time

            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            loss = architect.step(input_search, target_search, temp=temp)

            total_time = time.time() - start_time

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % config.ngpus_per_node == 0):
                if (step+1) % 10 == 0:
                    logger.add_scalar('loss/train', loss, epoch*len(train_loader_model)+step)
                    if config.efficiency_metric == 'flops':
                        logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch*len(train_loader_model)+step)
                    elif config.efficiency_metric == 'latency':
                        logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(train_loader_model)+step)

                    logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % 
                                (epoch + 1, config.nepochs, step + 1, len(train_loader_model), loss.item(), total_time, data_time))

            torch.cuda.empty_cache()
            del loss

    else:
        dataloader_model = iter(train_loader_model)

        for step in range(len(train_loader_model)):
            start_time = time.time()

            input, target = dataloader_model.next()

            data_time = time.time() - start_time

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logit = model(input, temp)
            loss = model.module._criterion(logit, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            total_time = time.time() - start_time

            if not config.multiprocessing_distributed or (config.multiprocessing_distributed and config.rank % ngpus_per_node == 0):
                if (step+1) % 10 == 0:
                    logging.info("[Epoch %d/%d][Step %d/%d] Loss=%.3f Time=%.3f Data Time=%.3f" % 
                                (epoch + 1, config.nepochs, step + 1, len(train_loader_model), loss.item(), total_time, data_time))
                    logger.add_scalar('loss/train', loss, epoch*len(train_loader_model)+step)

            torch.cuda.empty_cache()
            del loss


def infer(epoch, model, test_loader, logger, temp=1, finalize=False):
    model.eval()
    prec1_list = []

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input, volatile=True).cuda()
            target_var = Variable(target, volatile=True).cuda()

            output = model(input_var)
            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list)/len(prec1_list)

    if finalize:
        model_infer = FBNet_Infer(getattr(model.module, 'alpha'), getattr(model.module, 'beta'), config=config)
        if config.efficiency_metric == 'flops':
            flops = model_infer.forward_flops((3, config.image_height, config.image_width))
            return acc, flops

        elif config.efficiency_metric == 'latency':
            fps, _ = model_infer.eval_latency(cifar='cifar' in config.dataset, iteration=config.hw_update_iter, mode=config.hw_update_mode,
                                             fix_comp_mode=config.hw_update_fix_comp_mode, temp=config.hw_update_temp)
            return acc, fps
    else:
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
