import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile


from analytical_model.analytical_prediction import search_for_best_latency, evaluate_latency


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, quant_idx, layer_id, stride=1, num_bits_list=[32]):
        super(MixedOp, self).__init__()
        self.layer_id = layer_id
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, layer_id, stride)

        if type(num_bits_list) == list:
            self.num_bits = num_bits_list[quant_idx[op_idx]]
        else:
            self.num_bits = num_bits_list

    def forward(self, x):
        return self._op(x, num_bits=self.num_bits)


    def forward_flops(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        flops, size_out = self._op.forward_flops(size)
        
        return flops, size_out

    def forward_bitops(self, size):
        flops, size_out = self._op.forward_flops(size)

        bitops = flops * self.num_bits * self.num_bits

        return bitops, size_out



class FBNet_Infer(nn.Module):
    def __init__(self, alpha, beta, config):
        super(FBNet_Infer, self).__init__()

        self.op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)
        self.quant_idx_list = F.softmax(beta, dim=-1).argmax(-1)

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
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], self.op_idx_list[layer_id-1], self.quant_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], self.op_idx_list[layer_id-1], self.quant_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], self.op_idx_list[layer_id-1], self.quant_idx_list[layer_id-1], layer_id, stride=1, num_bits_list=self.num_bits_list)
                
                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.header_channel, self.num_classes)

        self._criterion = nn.CrossEntropyLoss()

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


    def forward(self, input):

        out = self.stem(input)

        for i, cell in enumerate(self.cells):
            out = cell(out)

        out = self.fc(self.avgpool(self.header(out)).view(out.size(0), -1))

        return out
    
    def forward_flops(self, size):

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def forward_bitops(self, size):

        bitops_total = []

        flops, size = self.stem.forward_flops(size)
        bitops_total.append(flops*8*8)

        for i, cell in enumerate(self.cells):
            bitops, size = cell.forward_bitops(size)
            bitops_total.append(bitops)

        flops, size = self.header.forward_flops(size)
        bitops_total.append(flops*8*8)

        return sum(bitops_total)


    def _loss(self, input, target):
        logit = self(input)
        loss = self._criterion(logit, target)

        return loss


    def eval_latency(self, cifar=True, iteration=10000, mode='random', fix_comp_mode=True, temp=1, hardware=None):
        block_info = [PRIMITIVES[op_id] for op_id in self.op_idx_list]

        if type(self.num_bits_list) == list:
            quant_idx_list_layerwise = [self.quant_idx_list[layer_id][self.op_idx_list[layer_id]] for layer_id in range(len(self.op_idx_list))]
            quant_info = [self.num_bits_list[bit_id] for bit_id in quant_idx_list_layerwise]
        else:
            quant_info = [self.num_bits_list for _ in range(len(self.op_idx_list))]
            self.num_bits_list = [self.num_bits_list]

        if hardware is None:
            searched_hw, throughput, block_wise_performance = search_for_best_latency(block_info, quant_info, block_options=PRIMITIVES, quant_options=self.num_bits_list, 
                                                                                    cifar=cifar, edd=False, iteration=iteration, mode=mode, fix_comp_mode=fix_comp_mode, temp=temp)
        else:
            throughput, block_wise_performance = evaluate_latency(block_info, quant_info, hardware, cifar=cifar, edd=False)
            searched_hw = hardware

        return throughput, searched_hw
