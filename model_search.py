import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from operations import *
import operations
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile


from analytical_model.analytical_prediction import search_for_best_latency, evaluate_latency



# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits, dim=-1), temperature)

    return y
    
    # if not hard:
    #     return y

    # shape = y.size()
    # _, ind = y.max(dim=-1)
    # y_hard = torch.zeros_like(y).view(-1, shape[-1])
    # y_hard.scatter_(1, ind.view(-1, 1), 1)
    # y_hard = y_hard.view(*shape)
    # # Set gradients w.r.t. y_hard gradients w.r.t. y
    # y_hard = (y_hard - y).detach() + y
    # return y_hard


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, layer_id, stride=1, num_bits_list=[32], mode='soft', mode_bit='soft', act_num=1, act_num_bit=1, hetero=False, flops_mode='sum'):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.layer_id = layer_id
        self.num_bits_list = num_bits_list
        self.mode = mode
        self.mode_bit = mode_bit
        self.act_num = act_num
        self.act_num_bit = act_num_bit
        self.hetero = hetero
        self.flops_mode = flops_mode

        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, layer_id, stride, hetero=self.hetero)
            self._ops.append(op)

        self.register_buffer('active_list', torch.tensor(list(range(len(self._ops)))))


    def forward(self, x, alpha, beta, alpha_param=None, beta_param=None):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        if self.mode == 'soft':
            for i, (w, op) in enumerate(zip(alpha, self._ops)):
                result = result + op(x, self.num_bits_list, beta[i], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[i]) * w 
                # print(type(op), result.shape)

            self.set_active_list(list(range(len(self._ops))))

        elif self.mode == 'hard':
            rank = alpha.argsort(descending=True)
            self.set_active_list(rank[:self.act_num])

            for i in range(self.act_num):
                result = result + self._ops[rank[i]](x, self.num_bits_list, beta[rank[i]], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[rank[i]]) * ((1-alpha[rank[i]]).detach() + alpha[rank[i]])

        elif self.mode == 'fake_hard':
            rank = alpha.argsort(descending=True)
            self.set_active_list(rank[:self.act_num])

            result = result + self._ops[rank[0]](x, self.num_bits_list, beta[rank[0]], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[rank[0]]) * ((1-alpha[rank[0]]).detach() + alpha[rank[0]])
            for i in range(1,self.act_num):
                result = result + self._ops[rank[i]](x, self.num_bits_list, beta[rank[i]], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[rank[i]]) * ((0-alpha[rank[i]]).detach() + alpha[rank[i]])

        elif self.mode == 'random_hard':
            rank = list(alpha.argsort(descending=True))
            result = result + self._ops[rank[0]](x, self.num_bits_list, beta[rank[0]], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[rank[0]]) * ((1-alpha[rank[0]]).detach() + alpha[rank[0]])
            
            forward_op_index = rank.pop(0)
            sampled_op = np.random.choice(rank, self.act_num-1)

            for i in range(len(sampled_op)):
                result = result + self._ops[sampled_op[i]](x, self.num_bits_list, beta[sampled_op[i]], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[sampled_op[i]]) * ((0-alpha[sampled_op[i]]).detach() + alpha[sampled_op[i]])

            sampled_op = sampled_op.tolist()
            sampled_op.insert(0, forward_op_index)
            self.set_active_list(sampled_op)

        elif self.mode == 'proxy_hard':
            assert alpha_param is not None
            # print('alpha_gumbel:', alpha)
            rank = alpha.argsort(descending=True)
            self.set_active_list(rank[:self.act_num])

            alpha = F.softmax(alpha_param[rank[:self.act_num]], dim=-1)

            # print('rank:', rank)
            # print('alpha_param:', alpha_param)
            # print('alpha_softmax:', alpha)

            result = result + self._ops[rank[0]](x, self.num_bits_list, beta[rank[0]], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[rank[0]]) * ((1-alpha[0]).detach() + alpha[0])
            for i in range(1,self.act_num):
                result = result + self._ops[rank[i]](x, self.num_bits_list, beta[rank[i]], mode=self.mode_bit, act_num=self.act_num_bit, beta_param=beta_param[rank[i]]) * ((0-alpha[i]).detach() + alpha[i])

        else:
            print('Wrong search mode:', self.mode)
            sys.exit()

        return result

    # set the active operator list for each block
    def set_active_list(self, active_list):
        if type(active_list) is not torch.Tensor:
            active_list = torch.tensor(active_list).cuda()

        self.active_list.data = active_list.data


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'

        for op in self._ops:
            op.set_stage(stage)


    def forward_flops(self, size, alpha, beta):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        if self.flops_mode == 'sum':
            for i, (w, op) in enumerate(zip(alpha, self._ops)):
                flops, size_out = op.forward_flops(size)

                if type(self.num_bits_list) == list:
                    flops = flops * sum([self.num_bits_list[j] * self.num_bits_list[j] * beta[i][j] for j in range(len(self.num_bits_list))]) /32 /32

                result = result + flops * w


        elif self.flops_mode == 'single_path':
            op_id = alpha.argsort(descending=True)[0]
            flops, size_out = self._ops[op_id].forward_flops(size)

            if type(self.num_bits_list) == list:
                bit_id = beta[op_id].argsort(descending=True)[0]
                result = alpha[op_id] * flops * beta[op_id][bit_id] * self.num_bits_list[bit_id] * self.num_bits_list[bit_id] /32 /32

            else:
                result = alpha[op_id] * flops


        elif self.flops_mode == 'multi_path':
            active_op = alpha.argsort(descending=True)[:self.act_num]

            for op_id in active_op: 
                flops, size_out = self._ops[op_id].forward_flops(size)

                active_bit = beta[op_id].argsort(descending=True)[:self.act_num_bit]

                if type(self.num_bits_list) == list:
                    flops = flops * sum([self.num_bits_list[j] * self.num_bits_list[j] * beta[op_id][j] for j in active_bit]) /32 /32

                result = result + flops * alpha[op_id]

        else:
            print('Wrong flops_mode.')
            sys.exit()

        return result, size_out



class FBNet(nn.Module):
    def __init__(self, config):
        super(FBNet, self).__init__()

        self.hard = config.hard

        self.mode = config.mode
        self.mode_bit = config.mode_bit
        self.act_num = config.act_num
        self.act_num_bit = config.act_num_bit

        self.num_classes = config.num_classes

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

        self.num_bits_list = config.num_bits_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel

        if config.dataset == 'imagenet':
            stride_init = 2
        else:
            stride_init = 1

        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=stride_init, padding=1, bias=False)

        self.cells = nn.ModuleList()

        layer_id = 1

        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list, mode=self.mode, mode_bit=self.mode_bit, act_num=self.act_num, act_num_bit=self.act_num_bit, hetero=config.bit_heterogenous_search, flops_mode=config.flops_mode)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list, mode=self.mode, mode_bit=self.mode_bit, act_num=self.act_num, act_num_bit=self.act_num_bit, hetero=config.bit_heterogenous_search, flops_mode=config.flops_mode)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], layer_id, stride=1, num_bits_list=self.num_bits_list, mode=self.mode, mode_bit=self.mode_bit, act_num=self.act_num, act_num_bit=self.act_num_bit, hetero=config.bit_heterogenous_search, flops_mode=config.flops_mode)
                
                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)

        self._arch_params = self._build_arch_parameters()
        self._reset_arch_parameters()

        self._criterion = nn.CrossEntropyLoss()

        self.sample_func = config.sample_func

        self.searched_hw = None

        self.init_params()


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, input, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
            beta = F.softmax(getattr(self, "beta"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
            beta = gumbel_softmax(getattr(self, "beta"), temperature=temp, hard=self.hard)
    
        out = self.stem(input)

        for i, cell in enumerate(self.cells):
            out = cell(out, alpha[i], beta[i], getattr(self, "alpha")[i], getattr(self, "beta")[i])

        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))

        return out
        ###################################


    def set_search_mode(self, mode='soft', act_num=1):
        self.mode = mode
        self.act_num = act_num

        if self.mode == 'soft':
            self.hard = False
        else:
            self.hard = True

        for cell in self.cells:
            cell.mode = mode
            cell.act_num = act_num


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        for cell in self.cells:
            cell.set_stage(stage)


    def sample_single_path(self):
        block_info = []
        quant_info = []

        for cell in self.cells:
            op_id = cell.active_list[0]
            block_info.append(PRIMITIVES[op_id])

            if type(self.num_bits_list) is list:
                bit_id = cell._ops[op_id].active_bit_list[0]
                quant_info.append(self.num_bits_list[bit_id])
            else:
                quant_info.append(self.num_bits_list)

        return block_info, quant_info


    def search_for_hw(self, cifar=True, iteration=10000, mode='random', fix_comp_mode=True, temp=1):
        block_info, quant_info = self.sample_single_path()

        searched_hw, throughput, block_wise_performance = search_for_best_latency(block_info, quant_info, block_options=PRIMITIVES, quant_options=self.num_bits_list if type(self.num_bits_list) is list else [self.num_bits_list], 
                                                                                    cifar=cifar, edd=False, iteration=iteration, mode=mode, fix_comp_mode=fix_comp_mode, temp=temp)

        self.searched_hw = searched_hw


    def forward_hw_latency(self, cifar=True):
        assert self.searched_hw is not None

        block_info, quant_info = self.sample_single_path()
        throughput, block_wise_performance = evaluate_latency(block_info, quant_info, self.searched_hw, cifar=cifar, edd=False)

        latency = 0
        for layer_id in range(len(block_wise_performance)):
            op_id = PRIMITIVES.index(block_info[layer_id])

            if type(self.num_bits_list) is list:
                bit_id = self.num_bits_list.index(quant_info[layer_id])
            else:
                bit_id = 0

            alpha_ste = (1-self.alpha[layer_id][op_id]).detach() + self.alpha[layer_id][op_id]
            beta_ste = (1-self.beta[layer_id][op_id][bit_id]).detach() + self.beta[layer_id][op_id][bit_id]

            latency += alpha_ste * beta_ste * block_wise_performance[layer_id]

        return latency



    def forward_flops(self, size, temp=1, alpha_only=False, beta_only=False):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
            beta = F.softmax(getattr(self, "beta"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp, hard=self.hard)
            beta = gumbel_softmax(getattr(self, "beta"), temperature=temp, hard=self.hard)

        if alpha_only:
            beta = torch.ones_like(getattr(self, 'beta')).cuda() * 1./len(self.num_bits_list)
        
        if beta_only:
            alpha = torch.ones_like(getattr(self, 'alpha')).cuda() * 1./len(PRIMITIVES)

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size, alpha[i], beta[i])
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def _loss(self, input, target, temp=1):

        logit = self(input, temp)
        loss = self._criterion(logit, target)

        return loss


    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)))

        if type(self.num_bits_list) == list:
            setattr(self, 'beta', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops, len(self.num_bits_list)).cuda(), requires_grad=True)))
        else:
            setattr(self, 'beta', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops, 1).cuda(), requires_grad=True)))

        return {"alpha": self.alpha, "beta": self.beta}


    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)

        getattr(self, "alpha").data = Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)

        if type(self.num_bits_list) == list:
            getattr(self, "beta").data = Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops, len(self.num_bits_list)).cuda(), requires_grad=True)
        else:
            getattr(self, "beta").data = Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops, 1).cuda(), requires_grad=True)


    def clip(self):
        for line in getattr(self, "alpha"):
            max_index = line.argmax()
            line.data.clamp_(0, 1)
            if line.sum() == 0.0:
                line.data[max_index] = 1.0
            line.data.div_(line.sum())

        for ops in getattr(self, "beta"):
            for line in ops:
                max_index = line.argmax()
                line.data.clamp_(0, 1)
                if line.sum() == 0.0:
                    line.data[max_index] = 1.0
                line.data.div_(line.sum())



if __name__ == '__main__':
    model = FBNet(num_classes=10)
    print(model)