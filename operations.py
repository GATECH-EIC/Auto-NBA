from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from thop.count_hooks import count_convNd

import sys
import os.path as osp
from easydict import EasyDict as edict

from quantize import QConv2d, QLinear

from analytical_model.analytical_prediction import search_for_best_latency, evaluate_latency

custom_ops = {QConv2d: count_convNd}


__all__ = ['ConvBlock', 'Skip','ConvNorm', 'OPS']

flops_lookup_table = {}
flops_file_name = "flops_lookup_table.npy"
if osp.isfile(flops_file_name):
    flops_lookup_table = np.load(flops_file_name, allow_pickle=True).item()

latency_lookup_table = {}
latency_file_name = "latency_lookup_table.npy"
if osp.isfile(latency_file_name):
    latency_lookup_table = np.load(latency_file_name, allow_pickle=True).item()


Conv2d = QConv2d
BatchNorm2d = nn.BatchNorm2d

USE_HSWISH = False
USE_SE = False

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert (
            C % g == 0
        ), "Incompatible group size {} for input channel {}".format(g, C)
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class ConvBlock(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out,  layer_id, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, hetero=False):
        super(ConvBlock, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.layer_id = layer_id

        self.register_buffer('active_bit_list', torch.tensor([0]).cuda())

        assert type(expansion) == int
        self.expansion = expansion
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

        self.conv1 = Conv2d(C_in, C_in*expansion, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias, hetero=hetero)
        self.bn1 = BatchNorm2d(C_in*expansion)

        self.conv2 = Conv2d(C_in*expansion, C_in*expansion, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=1, groups=C_in*expansion, bias=bias, hetero=hetero)
        self.bn2 = BatchNorm2d(C_in*expansion)

        if USE_SE:
            self.se = SEModule(C_in*expansion)

        self.conv3 = Conv2d(C_in*expansion, C_out, kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups, bias=bias, hetero=hetero)
        self.bn3 = BatchNorm2d(C_out)

        if USE_HSWISH:
            self.nl = Hswish(inplace=True)
        else:
            self.nl = nn.ReLU(inplace=True)


    # beta, mode, act_num, beta_param are for bit-widths search
    def forward(self, x, num_bits=32, beta=None, mode='soft', act_num=None, beta_param=None):
        if mode == 'random_hard':
            seed = np.random.randint(100)
            np.random.seed(seed)

        identity = x
        x = self.nl(self.bn1(self.conv1(x, num_bits=num_bits, beta=beta, mode=mode, act_num=act_num, beta_param=beta_param)))

        if self.groups > 1:
            x = self.shuffle(x)

        if mode == 'random_hard':
            np.random.seed(seed)

        x = self.nl(self.bn2(self.conv2(x, num_bits=num_bits, beta=beta, mode=mode, act_num=act_num, beta_param=beta_param)))

        if USE_SE:
            x = self.se(x)

        if mode == 'random_hard':
            np.random.seed(seed)

        x = self.bn3(self.conv3(x, num_bits=num_bits, beta=beta, mode=mode, act_num=act_num, beta_param=beta_param))

        if self.C_in == self.C_out and self.stride == 1:
            x += identity

        self.set_active_bit_list(num_bits, mode, beta, act_num)

        return x


    # set the active bit list for each block
    def set_active_bit_list(self, num_bits, mode, beta, act_num):
        if type(num_bits) == list and len(num_bits) > 1 and beta is not None:
            if mode == 'soft':
                self.active_bit_list.data = torch.tensor(range(len(num_bits))).cuda()

            else:
                assert act_num is not None
                # rank = beta.argsort(descending=True)
                # self.active_bit_list = rank[:act_num]
                self.active_bit_list.data = self.conv1.active_bit_list.cuda()


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv1.set_stage(stage)
        self.conv2.set_stage(stage)
        self.conv3.set_stage(stage)


    @staticmethod
    def _flops(h, w, C_in, C_out, expansion=1, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer_id = 1
        layer = ConvBlock(C_in, C_out, layer_id, expansion, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops
    

    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvBlock_H%d_W%d_Cin%d_Cout%d_exp%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvBlock._flops(h_in, w_in, c_in, c_out, self.expansion, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)



class Skip(nn.Module):
    def __init__(self, C_in, C_out, layer_id, stride=1, hetero=False):
        super(Skip, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0, 'C_out=%d'%C_out
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride

        self.layer_id = layer_id

        self.register_buffer('active_bit_list', torch.tensor([0]).cuda())

        self.kernel_size = 1
        self.padding = 0

        if stride == 2 or C_in != C_out:
            self.conv = Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False, hetero=hetero)
            self.bn = BatchNorm2d(C_out)
            self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = Skip(C_in, C_out, stride)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "Skip_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)

        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = Skip._flops(h_in, w_in, c_in, c_out, self.stride)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)


    def forward(self, x, num_bits=32, beta=None, mode='soft', act_num=None, beta_param=None):
        if hasattr(self, 'conv'):
            out = self.conv(x, num_bits=num_bits, beta=beta, mode=mode, act_num=act_num, beta_param=beta_param)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x

        self.set_active_bit_list(num_bits, mode, beta, act_num)

        return out


    # set the active bit list for each block
    def set_active_bit_list(self, num_bits, mode, beta, act_num):
        if hasattr(self, 'conv') and type(num_bits) == list and len(num_bits) > 1 and beta is not None:
            if mode == 'soft':
                self.active_bit_list.data = torch.tensor(range(len(num_bits))).cuda()

            else:
                assert act_num is not None
                # rank = beta.argsort(descending=True)
                # self.active_bit_list = rank[:act_num]
                self.active_bit_list.data = self.conv.active_bit_list.cuda()


    def set_stage(self, stage):
        if hasattr(self, 'conv'):
            assert stage == 'update_weight' or stage == 'update_arch'
            self.conv.set_stage(stage)



class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, hetero=False):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        self.register_buffer('active_bit_list', torch.tensor([0]).cuda())

        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        self.groups = groups
        self.bias = bias

        self.conv = Conv2d(C_in, C_out, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, 
                            dilation=self.dilation, groups=self.groups, bias=bias, hetero=hetero)
        self.bn = BatchNorm2d(C_out)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x, num_bits=32, beta=None, mode='soft', act_num=None, beta_param=None):
        x = self.relu(self.bn(self.conv(x, num_bits=num_bits, beta=beta, mode=mode, act_num=act_num, beta_param=beta_param)))

        self.set_active_bit_list(num_bits, mode, beta, act_num)

        return x


    # set the active bit list for each block
    def set_active_bit_list(self, num_bits, mode, beta, act_num):
        if type(num_bits) == list and len(num_bits) > 1 and beta is not None:
            if mode == 'soft':
                self.active_bit_list.data = torch.tensor(range(len(num_bits))).cuda()

            else:
                assert act_num is not None
                # rank = beta.argsort(descending=True)
                # self.active_bit_list = rank[:act_num]
                self.active_bit_list.data = self.conv.active_bit_list.cuda()


    def set_stage(self, stage):
        assert stage == 'update_weight' or stage == 'update_arch'
        self.conv.set_stage(stage)


    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), custom_ops=custom_ops)
        return flops


    def forward_flops(self, size):
        c_in, h_in, w_in = size

        assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
        c_out = self.C_out

        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2

        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d_group%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.groups)
        if name in flops_lookup_table:
            flops = flops_lookup_table[name]
        else:
            print("not found in flops_lookup_table:", name)
            flops = ConvNorm._flops(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            flops_lookup_table[name] = flops
            np.save(flops_file_name, flops_lookup_table)

        return flops, (c_out, h_out, w_out)



OPS = {
    'k3_e1' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=1, hetero=hetero),
    'k3_e1_g2' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=3, stride=stride, groups=2, hetero=hetero),
    'k3_e3' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=3, stride=stride, groups=1, hetero=hetero),
    'k3_e6' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=3, stride=stride, groups=1, hetero=hetero),
    'k5_e1' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=1, hetero=hetero),
    'k5_e1_g2' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=1, kernel_size=5, stride=stride, groups=2, hetero=hetero),
    'k5_e3' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=3, kernel_size=5, stride=stride, groups=1, hetero=hetero),
    'k5_e6' : lambda C_in, C_out, layer_id, stride, hetero=False: ConvBlock(C_in, C_out, layer_id, expansion=6, kernel_size=5, stride=stride, groups=1, hetero=hetero),
    'skip' : lambda C_in, C_out, layer_id, stride, hetero=False: Skip(C_in, C_out, layer_id, stride)
}

