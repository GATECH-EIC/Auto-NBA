import torch
import numpy as np
import sys
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchcontrib
import numpy as np
from pdb import set_trace as bp
from thop import profile
from operations import *
from genotypes import PRIMITIVES


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        # self.network_momentum = args.momentum
        # self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args

        self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
        
        self.alpha_weight = args.alpha_weight
        self.beta_weight = args.beta_weight

        if self._args.enable_mix_lr:
            self.optimizer_alpha = torch.optim.Adam([self.model.module._arch_params['alpha']], lr=args.alpha_weight*args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
            self.optimizer_beta = torch.optim.Adam([self.model.module._arch_params['beta']], lr=args.beta_weight*args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)

        self.flops_weight = args.flops_weight

        self.flops_decouple = args.flops_decouple

        self.latency_weight = args.latency_weight

        self.mode = args.mode

        self.mode_bit = args.mode_bit

        self.offset = args.offset

        self.offset_bit = args.offset_bit

        self.weight_optimizer = None

        self.hw_update_cnt = 0

        self.hw_aware_nas = args.hw_aware_nas

        self.hw_update_freq = args.hw_update_freq
        self.hw_update_iter = args.hw_update_iter
        self.hw_update_mode = args.hw_update_mode
        self.hw_update_fix_comp_mode = args.hw_update_fix_comp_mode
        self.hw_update_temp = args.hw_update_temp

        print("architect initialized!")


    def set_weight_optimizer(self, weight_optimizer):
        self.weight_optimizer = weight_optimizer


    def step(self, input_valid, target_valid, temp=1):
        self.optimizer.zero_grad()

        if self.mode == 'proxy_hard' and self.offset:
            alpha_old = self.model.module._arch_params['alpha'].data.clone()

        if self.mode_bit == 'proxy_hard' and self.offset_bit:
            beta_old = self.model.module._arch_params['beta'].data.clone()

        if self.weight_optimizer is not None:
            self.weight_optimizer.zero_grad()

        if self._args.efficiency_metric == 'flops':
            loss, loss_flops = self._backward_step_flops(input_valid, target_valid, temp)

        elif self._args.efficiency_metric == 'latency':
            loss, loss_latency = self._backward_step_latency(input_valid, target_valid, temp)

        else:
            print('Wrong efficiency metric.')
            sys.exit()

        if self._args.arch_one_hot_loss_weight:
            prob_alpha = F.softmax(getattr(self.model.module, 'alpha'), dim=-1)
            prob_beta = F.softmax(getattr(self.model.module, 'beta'), dim=-1)
            loss += self._args.arch_one_hot_loss_weight * (torch.mean(- prob_alpha * torch.log(prob_alpha)) + torch.mean(- prob_beta * torch.log(prob_beta)))

        if self._args.arch_mse_loss_weight:
            prob_alpha = F.softmax(getattr(self.model.module, 'alpha'), dim=-1)
            prob_beta = F.softmax(getattr(self.model.module, 'beta'), dim=-1)
            loss += self._args.arch_mse_loss_weight * (torch.mean(-torch.pow((prob_alpha - 0.5), 2)) + torch.mean(-torch.pow((prob_beta - 0.5), 2)))

        loss.backward()

        if self._args.enable_mix_lr:
            self.optimizer.step()
            self.optimizer.zero_grad()

        ## decouple the efficiency loss of alpha and beta
        if self._args.efficiency_metric == 'flops':
            if self.flops_weight > 0:
                loss_flops.backward()


        elif self._args.efficiency_metric == 'latency':
            if self.latency_weight > 0:
                loss_latency.backward()

        else:
            print('Wrong efficiency metric:', self._args.efficiency_metric)
            sys.exit()

        if self._args.enable_mix_lr:
            self.optimizer_alpha.step()
            self.optimizer_beta.step()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
            

        # update weight is one-level optimization
        if self.weight_optimizer is not None:
            self.weight_optimizer.step()


        if self.mode == 'proxy_hard' and self.offset:
            alpha_new = self.model.module._arch_params['alpha'].data

            for i, cell in enumerate(self.model.module.cells):
                # print('active list:', cell.active_list)
                # print('old:', alpha_old[i])
                # print('new:', alpha_new[i])

                offset = torch.log(sum(torch.exp(alpha_old[i][cell.active_list])) / sum(torch.exp(alpha_new[i][cell.active_list])))

                # print('active op:', cell.active_list)

                for active_op in cell.active_list:
                    self.model.module._arch_params['alpha'][i][active_op].data += offset.data

                # print('add offset:', alpha_new[i])

        if self.mode_bit == 'proxy_hard' and self.offset_bit:
            beta_new = self.model.module._arch_params['beta'].data

            for i, cell in enumerate(self.model.module.cells):
                # print('active list:', cell.active_list)
                # print('old:', alpha_old[i])
                # print('new:', alpha_new[i])

                for op_id, op in enumerate(cell._ops):
                    if op.active_bit_list is not None:
                        offset = torch.log(sum(torch.exp(beta_old[i][op_id][op.active_bit_list])) / sum(torch.exp(beta_new[i][op_id][op.active_bit_list])))

                        # print('active bitwidth:', op.active_bit_list)

                        for active_bit in op.active_bit_list:
                            self.model.module._arch_params['beta'][i][op_id][active_bit].data += offset.data

        return loss


    def _backward_step_latency(self, input_valid, target_valid, temp=1):
        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        # latency = self.model.module.forward_latency((3, self._args.image_height, self._args.image_width), temp)

        if self.latency_weight > 0:
            cifar = 'cifar' in self._args.dataset

            if self.hw_aware_nas:
                if self.hw_update_cnt == 0:
                    self.model.module.search_for_hw(cifar=cifar, iteration=self.hw_update_iter, mode=self.hw_update_mode, fix_comp_mode=self.hw_update_fix_comp_mode, temp=self.hw_update_temp)

            else:
                if self.hw_update_cnt == 0 or self.hw_update_cnt % self.hw_update_freq == 0:
                    self.model.module.search_for_hw(cifar=cifar, iteration=self.hw_update_iter, mode=self.hw_update_mode, fix_comp_mode=self.hw_update_fix_comp_mode, temp=self.hw_update_temp)

            self.hw_update_cnt += 1
            latency = self.model.module.forward_hw_latency(cifar=cifar)
        else:
            latency = 0

        self.latency_supernet = latency
        loss_latency = self.latency_weight * latency

        return loss, loss_latency



    def _backward_step_flops(self, input_valid, target_valid, temp=1):
        # print('Param on CPU:', [name for name, param in self.model.named_parameters() if param.device.type == 'cpu'])
        # print('Buffer on CPU:', [name for name, param in self.model.named_buffers() if param.device.type == 'cpu'])

        logit = self.model(input_valid, temp)
        loss = self.model.module._criterion(logit, target_valid)

        if self.flops_weight > 0:
            if self.flops_decouple:
                flops_alpha = self.model.module.forward_flops((3, self._args.image_height, self._args.image_width), temp, alpha_only=True)
                flops_beta = self.model.module.forward_flops((3, self._args.image_height, self._args.image_width), temp, beta_only=True)
                flops = flops_alpha + flops_beta
            else:
                flops = self.model.module.forward_flops((3, self._args.image_height, self._args.image_width), temp)
        else:
            flops = 0

        self.flops_supernet = flops
        loss_flops = self.flops_weight * flops

        # print(flops, loss_flops, loss)
        return loss, loss_flops


