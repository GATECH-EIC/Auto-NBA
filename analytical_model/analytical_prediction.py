import os 
import sys
import numpy as np
import math
import re
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

dnn_structure=[\
              [4,96,32,11,4],\
              [48,256,16,5,1],\
              [256,384,8,3,1],\
              [192,384,8,3,1],\
              [192,256,8,3,1],\
              [48,256,112,5,1],\
              [4,48,224,5,1]
]


def pack_data(fn,keyword):
    files=os.listdir(fn)
    packed_data=[]
    for f in files:
        if(keyword in f):
            raw=np.load(fn+f,allow_pickle=True)
            for dp in raw:
                packed_data.append([dp[0][0][:]+[dp[0][1]],dp[1][0]])
    return packed_data


def comp_engine_lat(comp_mode,input_params,net_struct):
    result_lat=1
    if comp_mode==0:
        result_lat*=(input_params[2]*input_params[3]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[6]) 
    elif comp_mode==1:
        result_lat*=(input_params[2]*input_params[3]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[6]/input_params[7])
    elif comp_mode==2:
        result_lat*=(input_params[2]*input_params[3]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[4])        
    #print('comp lat ', result_lat)
    return result_lat

def dw_comp_engine_lat(comp_mode,input_params,net_struct):
    if input_params[3] !=1:
        print(input_params)
        raise Exception('input channel & corresponding tiling needs to be set as one for dw conv')
    result_lat=1
    if comp_mode==0:
        result_lat*=(input_params[2]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[6]) 
    elif comp_mode==1:
        result_lat*=(input_params[2]*input_params[0]*input_params[1]*net_struct[3]\
                     *net_struct[3]/input_params[4])  
    else:
        raise Exception('non-supported comp mode')
    return result_lat

def read_if_lat(comp_mode,input_params,net_struct,quant=16):
    tri=max(input_params[4]+net_struct[3]-1,input_params[0])
    tci=max(input_params[5]+net_struct[3]-1,input_params[1])
    if comp_mode==2:
        return math.ceil(input_params[3]*tci*tri/max(min(4,tri),2))*(quant/16)
    else:
        return math.ceil(input_params[3]*tci*tri/max(min(4,input_params[7]),2))*(quant/16)
    
def dw_read_if_lat(comp_mode,input_params,net_struct,quant=16):
    tri=max(input_params[4]+net_struct[3]-1,input_params[0])
    tci=max(input_params[5]+net_struct[3]-1,input_params[1])
    if comp_mode==2:
        return math.ceil(input_params[2]*tci*tri/max(min(4,tri),2))*(quant/16)
    else:
        return math.ceil(input_params[2]*tci*tri/max(min(4,input_params[6]),2))*(quant/16)


def read_we_lat(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==2:
        #print('weight loading',input_params[2]*input_params[3]*net_struct[3] )
        return input_params[2]*input_params[3]*net_struct[3]*(quant/16)
    else: 
        return math.ceil(input_params[2]*input_params[3]*net_struct[3]*net_struct[3]/max(min(4,input_params[6]),2))*(quant/16)

def dw_read_we_lat(comp_mode,input_params,net_struct,quant=16):
    if input_params[3] !=1:
        raise Exception('input channel & corresponding tiling needs to be set as one for dw conv')
    if comp_mode==1:
        return input_params[2]*input_params[3]*net_struct[3]*(quant/16)
    else:
        return math.ceil(input_params[2]*input_params[3]*net_struct[3]*net_struct[3]/max(min(4,input_params[6]),2))*(quant/16)

def write_ofmap(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==2:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[4]),2))
        clear_buffer=input_params[1]*input_params[2]*(input_params[0]/input_params[4])
    else:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[6]),2))
        clear_buffer=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
    #print('clear output', read_write_1, clear_buffer)
    
    return (read_write_1+clear_buffer)*(quant/16)


def dw_write_ofmap(comp_mode,input_params,net_struct,quant=16):
    if comp_mode==1:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[4]),2))
        clear_buffer=input_params[1]*input_params[2]*(input_params[0]/input_params[4])
    else:
        read_write_1=math.ceil(input_params[2]*input_params[0]*input_params[1]/max(min(4,input_params[6]),2))
        clear_buffer=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
    #print('clear output', read_write_1, clear_buffer)
    return (read_write_1+clear_buffer)*(quant/16)



def combined_latency(comp_mode, input_params,net_struct,quant=16):
    outer_loop_tc=net_struct[2]/input_params[1]
    outer_loop_tr=net_struct[2]/input_params[0]
    outer_loop_tm=net_struct[1]/input_params[2]
    outer_loop_tn=net_struct[0]/input_params[3]
    read_if_we_comp=max(comp_engine_lat(comp_mode,input_params,net_struct), read_if_lat(comp_mode,input_params,net_struct,quant=quant))+read_we_lat(comp_mode,input_params,net_struct,quant=quant)
    read_if_we_comp_tn=read_if_we_comp*outer_loop_tn
    inner_lat=write_ofmap(comp_mode,input_params,net_struct,quant=quant)+read_if_we_comp_tn

    return inner_lat*outer_loop_tc*outer_loop_tr*outer_loop_tm
    
    
def dw_combined_latency(comp_mode, input_params,net_struct,quant=16):
    outer_loop_tc=net_struct[2]/input_params[1]
    outer_loop_tr=net_struct[2]/input_params[0]
    outer_loop_tm=net_struct[1]/input_params[2]
    read_if_we_comp=max(dw_comp_engine_lat(comp_mode,input_params,net_struct),\
                        dw_read_if_lat(comp_mode,input_params,net_struct,quant=quant))+\
                        dw_read_we_lat(comp_mode,input_params,net_struct,quant=quant)+\
                        dw_write_ofmap(comp_mode,input_params,net_struct,quant=quant)
    return outer_loop_tc*outer_loop_tr*outer_loop_tm*read_if_we_comp

def resource_consumption(comp_mode,input_params,net_struct,dw=False,quant=16):
    max_bank_size=1125*16
    if not dw:
        if comp_mode==0:
            #TODO: cases using completely LUT
            dsp=input_params[6]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            
            input_bank_size=tri*tci*(input_params[3]/input_params[7])
            input_bram=input_params[7]*math.ceil(input_bank_size*quant/max_bank_size)*2
            output_bank_size=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
            output_bram=input_params[6]*math.ceil(output_bank_size*quant/max_bank_size)
            #TODO: in output channel tiling only; input channel is still tiled, fix in auto_runner side
            #      separate the parallel choice for kernels input_channel and output_channel
            weight_bank_size=net_struct[3]*net_struct[3]*input_params[3]*(input_params[2]/input_params[6])
            weight_bram=input_params[6]*math.ceil(weight_bank_size*quant/max_bank_size)
            total_bram=input_bram+output_bram
            
        elif comp_mode==1:
            dsp=input_params[6]*input_params[7]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            
            input_bank_size=tri*tci*(input_params[3]/input_params[7])
            input_bram=input_params[7]*math.ceil(input_bank_size*quant/max_bank_size)*2
            
            output_bank_size=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
            output_bram=input_params[6]*math.ceil(output_bank_size*quant/max_bank_size)
            
            weight_bank_size=net_struct[3]*net_struct[3]*(input_params[3]/input_params[7])*(input_params[2]/input_params[6])
            weight_bram=input_params[6]*input_params[7]*math.ceil(weight_bank_size*quant/max_bank_size)
            total_bram=input_bram+output_bram
            
        elif comp_mode==2:
            #TODO: adding additional adder tree cost
            dsp=input_params[4]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            input_bank_size=tci*input_params[3]
            input_bram=tri*math.ceil(input_bank_size*quant/max_bank_size)*2
            
            
            output_bank_size=input_params[1]*input_params[2]
            output_bram=input_params[4]*math.ceil(output_bank_size*quant/max_bank_size)
            
            weight_bank_size=net_struct[3]*input_params[2]*input_params[3]
            weight_bram=net_struct[3]*math.ceil(weight_bank_size*quant/max_bank_size)
            
            total_bram=input_bram+output_bram+weight_bram
    else:
        if comp_mode==0:
            #TODO: cases using completely LUT
            dsp=input_params[6]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            
            input_bank_size=tri*tci*(input_params[2]/input_params[6])
            input_bram=input_params[6]*math.ceil(input_bank_size*quant/max_bank_size)*2
            output_bank_size=input_params[0]*input_params[1]*(input_params[2]/input_params[6])
            output_bram=input_params[6]*math.ceil(output_bank_size*quant/max_bank_size)
            #TODO: in output channel tiling only; input channel is still tiled, fix in auto_runner side
            #      separate the parallel choice for kernels input_channel and output_channel
            weight_bank_size=net_struct[3]*net_struct[3]*(input_params[2]/input_params[6])
            weight_bram=input_params[6]*math.ceil(weight_bank_size*quant/max_bank_size)
            total_bram=input_bram+output_bram
        elif comp_mode==1:
            dsp=input_params[4]*(quant/16)
            #BRAM calculation
            tri=max(input_params[4]+net_struct[3]-1,input_params[0])
            tci=max(input_params[5]+net_struct[3]-1,input_params[1])
            input_bank_size=tci*input_params[3]
            input_bram=tri*math.ceil(input_bank_size*quant/max_bank_size)*2
            
            
            output_bank_size=input_params[1]*input_params[2]
            output_bram=input_params[4]*math.ceil(output_bank_size*quant/max_bank_size)
            
            weight_bank_size=net_struct[3]*input_params[2]
            weight_bram=net_struct[3]*math.ceil(weight_bank_size*quant/max_bank_size)
            
            total_bram=input_bram+output_bram+weight_bram
            
            
    return (dsp,total_bram)

def sys_latency(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget):
    #input_params_set
    #[[comp_mode,fw,fh,of,if,f(fw),f(fh),f(of),f(if),quant]...]
    #net_struct
    #[[]....]
    #accelerator_alloc
    #{layer_num:accelerator_num}
    latency_break_down={}
    layer_wise_break_down_to_accel={}
    layer_wise_break_down=[]
    for i in input_params_set.keys():
        latency_break_down[i]=0
        layer_wise_break_down_to_accel[i]=[]
    for i, layer_struct in enumerate(net_struct):
        input_params=input_params_set[accelerator_alloc[i]]
        if dw[i]:
            tmp_lat=dw_combined_latency(input_params[0],input_params[1:9],layer_struct,quant=input_params[-1])
            latency_break_down[accelerator_alloc[i]]+=tmp_lat
            layer_wise_break_down_to_accel[accelerator_alloc[i]].append(tmp_lat)
            layer_wise_break_down.append(tmp_lat)
        else:
            tmp_lat=combined_latency(input_params[0],input_params[1:9],layer_struct,quant=input_params[-1])
            latency_break_down[accelerator_alloc[i]]+=tmp_lat
            layer_wise_break_down_to_accel[accelerator_alloc[i]].append(tmp_lat)
            layer_wise_break_down.append(tmp_lat)
    bottleneck_latency=0
    for i in latency_break_down.keys(): 
        if latency_break_down[i] >bottleneck_latency:
            bottleneck_latency=latency_break_down[i]
    return bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,layer_wise_break_down

def sys_consumption(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget,platform_specs):
    #input_params_set
    #[[comp_mode,fw,fh,of,if,f(fw),f(fh),f(of),f(if),quant]...]
    #net_struct
    #[[]....]
    #accelerator_alloc
    #{layer_num:accelerator_num}
    consumption_breakdown={}
    for i in input_params_set.keys():
        consumption_breakdown[i]=[0,0]
    for i, layer_struct in enumerate(net_struct):
        input_params=input_params_set[accelerator_alloc[i]]
        consumption_breakdown[accelerator_alloc[i]]= [max(consumption_breakdown[accelerator_alloc[i]][0],\
                                                         resource_consumption(input_params[0],input_params[1:9],\
                                                         layer_struct,dw=dw[i],quant=input_params[-1])[0]),\
                                                      max(consumption_breakdown[accelerator_alloc[i]][1],\
                                                         resource_consumption(input_params[0],input_params[1:9],\
                                                         layer_struct,dw=dw[i],quant=input_params[-1])[1])]
                
    total_dsp_used=0
    total_bram_used=0
    for i in consumption_breakdown.keys():
        total_dsp_used+=consumption_breakdown[i][0]
        total_bram_used+=consumption_breakdown[i][1]
    if total_dsp_used>platform_specs['dsp']:
        raise Exception('dsp limit exceeded')
    elif total_bram_used>platform_specs['bram']:
        raise Exception('bram exceeded')
    
    for i in accelerator_wise_budget.keys():
        if consumption_breakdown[i][0] > accelerator_wise_budget[i]['dsp']:
            print("Warning: accelerator "+str(i)+" dsp budget exceeded")
        elif consumption_breakdown[i][1]> accelerator_wise_budget[i]['bram']: 
            print("Warning: accelerator "+str(i)+" bram budget exceeded")
    return (total_dsp_used,total_bram_used), consumption_breakdown


def allocate_layers(net_struct,quant_info,dw,platform_specs,cifar=True):
    dw_quantization_bins={}
    std_quantization_bins={}
    accelerator_alloc={}
    accelerator_wise_budget={}
    accelerator_types=[]
    for i, layer_struct in enumerate(net_struct):
        if dw[i]:
            if quant_info[i] not in dw_quantization_bins.keys():
                #initiate the bins
                dw_quantization_bins[quant_info[i]]=[i]
            else:
                #add layers to the corresponding bins
                dw_quantization_bins[quant_info[i]].append(i)
        else:
            if quant_info[i] not in std_quantization_bins.keys():
                #initiate the bins
                std_quantization_bins[quant_info[i]]=[i]
            else:
                #add layers to the corresponding bins
                std_quantization_bins[quant_info[i]].append(i)
    if cifar:    
        for i, quant_bit in enumerate(std_quantization_bins.keys()):
            for layer in std_quantization_bins[quant_bit]:
                if net_struct[layer][2]>=16:
                    if "a0"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("a0"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                else:
                    if "a1"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("a1"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                    
        for i, quant_bit in enumerate(dw_quantization_bins.keys()):
            for layer in dw_quantization_bins[quant_bit]:
                if net_struct[layer][2]>=16:
                    if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("dwa0"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                else:
                    if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("dwa1"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
    else:
        for i, quant_bit in enumerate(std_quantization_bins.keys()):
            for layer in std_quantization_bins[quant_bit]:
                if net_struct[layer][2]>=28:
                    if "a0"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("a0"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="a0"+"q"+str(quant_bit)
                else:
                    if "a1"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("a1"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="a1"+"q"+str(quant_bit)
                    
        for i, quant_bit in enumerate(dw_quantization_bins.keys()):
            for layer in dw_quantization_bins[quant_bit]:
                if net_struct[layer][2]>=28:
                    if "dwa0"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("dwa0"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="dwa0"+"q"+str(quant_bit)
                else:
                    if "dwa1"+"q"+str(quant_bit) not in accelerator_types:
                        accelerator_types.append("dwa1"+"q"+str(quant_bit))
                    accelerator_alloc[layer]="dwa1"+"q"+str(quant_bit)
    # print("="*20)     
    # print(len(net_struct))
    # print(len(list(accelerator_alloc.keys())))
    # print(accelerator_alloc)
    # print("="*20)  
    #return None
    return accelerator_alloc, accelerator_types, accelerator_wise_budget



########################
##DNA specific utilities
########################

def cifar_convert_to_layers(block_info,quant_info,cifar=True,edd=False):
    #TODO: include EDD cases
    if cifar:
        output_dim=[32]+[32]*4+[16]*4+[8]*4+[8]*4+[4]*4+[4]
        num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1]
        #currently only support 1 
        #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
        #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
        num_channel_list = [16]+[24]*4+[32]*4+[64]*4+[112]*4+[192]*4+[352]
        stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1]
        
    else: 
        output_dim=[112]+[56]*4+[28]*4+[14]*4+[14]*4+[7]*4+[7]
        num_layer_list = [1, 1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1,  1]
        #num_layer_list = [1, 4, 4, 4, 4, 4, 1]
        #num_channel_list = [16, 24, 32, 64, 112, 184, 352]
        num_channel_list = [16]+[24]*4+[32]*4+[64]*4+[112]*4+[192]*4+[352]
        stride_list = [1, 2,1,1,1, 2,1,1,1, 2,1,1,1, 1,1,1,1, 2,1,1,1, 1]
    if edd:
        output_dim=[56,28,28,28,28,14,14,14,14,14,14,7,7,7,7,7]
        num_layer_list= [1,1,1,1,  1,1,1,1,  1,1,1,1,  1,1,1,1]
        num_channel_list =[32,48,48,48, 96,96,96,96, 128,128,128,256, 256,256,256,320]
        stride_list=[2,2,1,1,  1,2,1,1,  1,1,1,2,  1,1,1,1]
    net_struct=[]
    dw=[]
    layer_wise_quant=[]
    layer_block_corr={}
    for i in range(sum(num_layer_list)):
        layer_block_corr[i]=[]
    layer_num=0
    for i, rep_times in enumerate(num_layer_list):
        if "g" not in block_info[i] and block_info[i] != "skip":
            k=int(block_info[i][1])
            e=int(block_info[i][4])
            if num_layer_list[i]==1:
                if i==0:
                    #TODO: confirm if the layer dimension is right
                    net_struct.append([16,16*e,output_dim[0],1,1])
                    net_struct.append([1,16*e,output_dim[0],k,1])
                    net_struct.append([16*e,16,output_dim[0],1,1])
                    dw+=[False,True,False]
                    quant_bit=quant_info.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit]
                    layer_block_corr[0]+=[0,1,2]
                    layer_num+=3
                else:
                    net_struct.append([num_channel_list[i-1],num_channel_list[i-1]*e,output_dim[i],1,stride_list[i]])
                    net_struct.append([1,num_channel_list[i-1]*e,output_dim[i],k,1])
                    net_struct.append([num_channel_list[i-1]*e,num_channel_list[i],output_dim[i],1,1])  
                    dw+=[False,True,False]
                    quant_bit=quant_info.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit]
                    layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2]
                    layer_num+=3
            else:
                raise Exception('Currently not supporting repetive block info input')
        elif "g" in  block_info[i]:
            k=int(block_info[i][1])
            e=int(block_info[i][4])
            if num_layer_list[i]==1:
                if i==0:
                    #TODO: confirm if the layer dimension is right
                    net_struct.append([16/2,16*e/2,output_dim[0],1,1])
                    net_struct.append([16/2,16*e/2,output_dim[0],1,1])
                    net_struct.append([1,16*e,output_dim[0],k,1])
                    net_struct.append([16*e/2,16/2,output_dim[0],1,1])
                    net_struct.append([16*e/2,16/2,output_dim[0],1,1])
                    dw+=[False,False,True,False,False]
                    quant_bit=quant_info.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit,quant_bit,quant_bit]
                    layer_block_corr[0]+=[0,1,2,3,4]
                    layer_num+=5
                else:
                    net_struct.append([num_channel_list[i-1]/2,num_channel_list[i-1]*e/2,output_dim[i],1,stride_list[i]])
                    net_struct.append([num_channel_list[i-1]/2,num_channel_list[i-1]*e/2,output_dim[i],1,stride_list[i]])
                    net_struct.append([1,num_channel_list[i-1]*e,output_dim[i],k,1])
                    net_struct.append([num_channel_list[i-1]*e/2,num_channel_list[i]/2,output_dim[i],1,1])  
                    net_struct.append([num_channel_list[i-1]*e/2,num_channel_list[i]/2,output_dim[i],1,1])
                    dw+=[False,False,True,False,False]
                    quant_bit=quant_info.pop(0)
                    layer_wise_quant+=[quant_bit,quant_bit,quant_bit,quant_bit,quant_bit]
                    layer_block_corr[i]+=[layer_num,layer_num+1,layer_num+2,layer_num+3,layer_num+4]
                    layer_num+=5
            else:
                raise Exception('Currently not supporting repetive block info input')
    return net_struct,dw,layer_wise_quant,layer_block_corr


def design_choice_gen(cifar=True,edd=False):
    #TODO: include imagenet cases
    if cifar:
        acc1_space={'comp_mode':[0,1,2],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
        acc2_space={'comp_mode':[0,1,2],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[16,8,4,2,1],'tcbuff':[16,8,4,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[16,8,4,2,1],'tc':[16,8,4,2,1],'tm':[8,4,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[4,2,1],'tcbuff':[4,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[4,2,1],'tc':[4,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
    else:
        acc1_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[8,4,2,1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[8,4,2,1]}
        acc2_space={'comp_mode':[0,1,2],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[8,4,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
    #design_choices: {comp_mode:[0,1,2],fw:[2,4,6,8]...}
    if edd:
        acc1_space={'comp_mode':[0,1,2],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[16,8,4,2,1],'tnbuff':[16,8,4,2,1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[16,8,4,2,1],'tn':[16,8,4,2,1]}
        acc2_space={'comp_mode':[0,1,2],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[32,16,8,4,2,1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[32,16,8,4,2,1]}
        dw_acc1_space={'comp_mode':[0,1],'trbuff':[28,14,7,2,1],'tcbuff':[28,14,7,2,1],'tmbuff':[16,8,4,2,1],'tnbuff':[1], 'tr':[28,14,7,2,1],'tc':[28,14,7,2,1],'tm':[16,8,4,2,1],'tn':[1]}
        dw_acc2_space={'comp_mode':[0,1],'trbuff':[7,2,1],'tcbuff':[7,2,1],'tmbuff':[32,16,8,4,2,1],'tnbuff':[1], 'tr':[7,2,1],'tc':[7,2,1],'tm':[32,16,8,4,2,1],'tn':[1]}
    
    return (acc1_space,acc2_space,dw_acc1_space,dw_acc2_space)



def random_sample(input_dict, fix_comp_mode=False):
    result_sample=[]
    result_sample_dict={}
    for key in input_dict.keys():
        if key == 'comp_mode' and fix_comp_mode:
            tmp = 1
        else:
            tmp=input_dict[key][np.random.randint(len(input_dict[key]))]

        if "tr"== key or "tc"==key or "tm" == key or "tn" ==key :
            #tmp=np.random.randint(len(input_dict[key]))
            while tmp > result_sample_dict[key+"buff"]:
                tmp=input_dict[key][np.random.randint(len(input_dict[key]))]
            result_sample.append(tmp)
            result_sample_dict[key]=tmp
        else:
            result_sample.append(tmp)
            result_sample_dict[key]=tmp
    return result_sample

def mac_calc(net_struct):
    mac=0
    for i, layer in enumerate(net_struct):
        mac+=layer[0]*layer[1]*layer[2]*layer[2]*layer[3]*layer[3]
    return mac

def capsuled_predictor(input_params_set, block_info_test,quant_info,cifar,edd):
    #generate the layer wise structure, if_layer_is_dw, layer_wise_quant
    net_struct,dw,layer_wise_quant,layer_block_corr=cifar_convert_to_layers(block_info_test,quant_info,cifar=cifar,edd=edd)

    #print(len(net_struct),len(dw))
    #print(mac_calc(net_struct))
    #exit()
    #allocate each layer with its corresponding accelerator
    #{layer_num: <accelerator_type>}
    accelerator_alloc, accelerator_types, accelerator_wise_budget=allocate_layers(net_struct,layer_wise_quant,dw,None,cifar=cifar)
    # print(dw)
    # print(accelerator_alloc)
    # print(accelerator_types)
    

    platform_specs={'dsp':900,'bram':700}
    bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,layer_wise_break_down=sys_latency(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget)
    consumption_used, consumption_breakdown=sys_consumption(input_params_set,net_struct,dw,accelerator_alloc,accelerator_wise_budget,platform_specs)
    bs=min(math.floor(platform_specs['dsp']/consumption_used[0]),math.floor(platform_specs['bram']/consumption_used[1]))
    bs=1
    bottleneck_latency=bottleneck_latency/bs
    for key in latency_break_down.keys():
        latency_break_down[key]=latency_break_down[key]/bs
        consumption_breakdown[key][0]=consumption_breakdown[key][0]*bs
        consumption_breakdown[key][1]=consumption_breakdown[key][1]*bs
        layer_wise_break_down_to_accel[key]=[i/bs for i in layer_wise_break_down_to_accel[key]]
    layer_wise_break_down=[i/bs for i in layer_wise_break_down]
    consumption_used=[i*bs for i in consumption_used]
    block_wise_performance=[]
    for key in layer_block_corr.keys():
        tmp_block_lat=0
        for layer_num in layer_block_corr[key]:
            tmp_block_lat+=layer_wise_break_down[layer_num]
        block_wise_performance.append(tmp_block_lat)
    #print(block_wise_performance)
        
    return bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
           layer_wise_break_down,consumption_used, consumption_breakdown,\
           accelerator_alloc,bs,block_wise_performance,net_struct


USE_CUDA = False

def generate_params(acc1_space,acc2_space,dw_acc1_space,dw_acc2_space, quant_options):
    params_dict = {}
    params_list = []
    for quant_option in quant_options:
        params_dict_part = {}
        for name, space in zip(['acc1_space','acc2_space','dw_acc1_space','dw_acc2_space'], [acc1_space,acc2_space,dw_acc1_space,dw_acc2_space]):
            if USE_CUDA:
                comp_mode = nn.Parameter(Variable(1e-3*torch.ones(len(space['comp_mode'])).cuda(), requires_grad=True))
                trbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['trbuff'])).cuda(), requires_grad=True))
                tcbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['tcbuff'])).cuda(), requires_grad=True))
                tmbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['tmbuff'])).cuda(), requires_grad=True))
                tnbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['tnbuff'])).cuda(), requires_grad=True))

                tr = nn.Parameter(Variable(1e-3*torch.ones(len(space['tr'])).cuda(), requires_grad=True))
                tc = nn.Parameter(Variable(1e-3*torch.ones(len(space['tc'])).cuda(), requires_grad=True))
                tm = nn.Parameter(Variable(1e-3*torch.ones(len(space['tm'])).cuda(), requires_grad=True))
                tn = nn.Parameter(Variable(1e-3*torch.ones(len(space['tn'])).cuda(), requires_grad=True))

            else:
                comp_mode = nn.Parameter(Variable(1e-3*torch.ones(len(space['comp_mode'])), requires_grad=True))
                trbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['trbuff'])), requires_grad=True))
                tcbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['tcbuff'])), requires_grad=True))
                tmbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['tmbuff'])), requires_grad=True))
                tnbuff = nn.Parameter(Variable(1e-3*torch.ones(len(space['tnbuff'])), requires_grad=True))

                tr = nn.Parameter(Variable(1e-3*torch.ones(len(space['tr'])), requires_grad=True))
                tc = nn.Parameter(Variable(1e-3*torch.ones(len(space['tc'])), requires_grad=True))
                tm = nn.Parameter(Variable(1e-3*torch.ones(len(space['tm'])), requires_grad=True))
                tn = nn.Parameter(Variable(1e-3*torch.ones(len(space['tn'])), requires_grad=True))

            params_dict_part[name] = {'comp_mode':comp_mode,'trbuff':trbuff,'tcbuff':tcbuff,'tmbuff':tmbuff,'tnbuff':tnbuff, 
                                      'tr':tr,'tc':tc,'tm':tm,'tn':tn}

            params_list.append(comp_mode)
            params_list.append(trbuff)
            params_list.append(tcbuff)
            params_list.append(tmbuff)
            params_list.append(tnbuff)
            params_list.append(tr)
            params_list.append(tc)
            params_list.append(tm)
            params_list.append(tn)

        params_dict[quant_option] = params_dict_part

    return params_dict, params_list


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if USE_CUDA:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())

    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(F.log_softmax(logits), temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard, y



BLOCK_OPTIONS = ['k3_e1','k3_e3','k3_e6','k5_e1','k5_e6','k5_e3','skip','k3_e1_g2','k5_e1_g2']
QUANT_OPTIONS = [4,6,8]


first_flag = True

# block_options = None
# quant_options = None
acc1_space = acc2_space = dw_acc1_space = dw_acc2_space = None

params_dict = None
params_list = None
optimizer = None
lr_policy = None


def search_for_best_latency(block_info, quant_info, block_options=BLOCK_OPTIONS, quant_options=QUANT_OPTIONS, cifar=True, edd=False, iteration=10000, mode='random', fix_comp_mode=False, temp=1):
    assert mode == 'random' or mode == 'differentiable'
    global first_flag

    global acc1_space, acc2_space, dw_acc1_space, dw_acc2_space

    # global block_options
    # global quant_options
    global acc1_space,acc2_space,dw_acc1_space,dw_acc2_space

    global params_dict, params_list, optimizer, lr_policy

    best_throughput = 0 

    if first_flag:
        # block_options = ['k3_e1','k3_e3','k3_e6','k5_e1','k5_e6','k5_e3','skip','k3_e1_g2','k5_e1_g2']
        # quant_options = [4,6,8]
        acc1_space,acc2_space,dw_acc1_space,dw_acc2_space = design_choice_gen(cifar=cifar, edd=edd)

        if mode == 'differentiable':      
            params_dict, params_list = generate_params(acc1_space, acc2_space, dw_acc1_space, dw_acc2_space, quant_options)  
            optimizer = torch.optim.Adam(params_list, lr=1e-7, betas=(0.5, 0.999))
            lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.1)

        first_flag = False


    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(iteration), file=sys.stdout, bar_format=bar_format, ncols=80)

    for iter_id in pbar:
        design_choice_integrity=False

        if mode == 'random':
            while not design_choice_integrity:
                input_params_set={}
                for quant_option in quant_options:
                    input_params_set["a0q"+str(quant_option)]=random_sample(acc1_space, fix_comp_mode=fix_comp_mode)+[quant_option]
                    input_params_set["a1q"+str(quant_option)]=random_sample(acc2_space, fix_comp_mode=fix_comp_mode)+[quant_option]
                    input_params_set["dwa0q"+str(quant_option)]=random_sample(dw_acc1_space, fix_comp_mode=fix_comp_mode)+[quant_option]
                    input_params_set["dwa1q"+str(quant_option)]=random_sample(dw_acc2_space, fix_comp_mode=fix_comp_mode)+[quant_option]

                for accel in input_params_set.keys():
                    if input_params_set[accel][1] < input_params_set[accel][5] or\
                       input_params_set[accel][2] < input_params_set[accel][6] or\
                       input_params_set[accel][3] < input_params_set[accel][7] or\
                       input_params_set[accel][4] < input_params_set[accel][8]:
                        design_choice_integrity=False
                        break
                    else:
                        design_choice_integrity=True

        elif mode == 'differentiable':
            while not design_choice_integrity:
                input_params_set = {}
                active_list = []

                for quant_option in quant_options:
                    option_list = []

                    for name, space in zip(['acc1_space','acc2_space','dw_acc1_space','dw_acc2_space'], [acc1_space,acc2_space,dw_acc1_space,dw_acc2_space]):
                        if fix_comp_mode:
                            comp_mode = 1
                        else:
                            comp_mode_hard, comp_mode_soft = gumbel_softmax(params_dict[quant_option][name]['comp_mode'], temperature=temp, hard=True)
                            comp_mode_index = comp_mode_hard.argmax()
                            comp_mode = space['comp_mode'][comp_mode_index]
                            active_list.append(comp_mode_soft[comp_mode_index])

                        trbuff_hard, trbuff_soft = gumbel_softmax(params_dict[quant_option][name]['trbuff'], temperature=temp, hard=True)
                        trbuff_index = trbuff_hard.argmax()
                        trbuff = space['trbuff'][trbuff_index]
                        active_list.append(trbuff_soft[trbuff_index])

                        tcbuff_hard, tcbuff_soft = gumbel_softmax(params_dict[quant_option][name]['tcbuff'], temperature=temp, hard=True)
                        tcbuff_index = tcbuff_hard.argmax()
                        tcbuff = space['tcbuff'][tcbuff_index]
                        active_list.append(tcbuff_soft[tcbuff_index])

                        tmbuff_hard, tmbuff_soft = gumbel_softmax(params_dict[quant_option][name]['tmbuff'], temperature=temp, hard=True)
                        tmbuff_index = tmbuff_hard.argmax()
                        tmbuff = space['tmbuff'][tmbuff_index]
                        active_list.append(tmbuff_soft[tmbuff_index])

                        tnbuff_hard, tnbuff_soft = gumbel_softmax(params_dict[quant_option][name]['tnbuff'], temperature=temp, hard=True)
                        tnbuff_index = tnbuff_hard.argmax()
                        tnbuff = space['tnbuff'][tnbuff_index]
                        active_list.append(tnbuff_soft[tnbuff_index])

                        tr_hard, tr_soft = gumbel_softmax(params_dict[quant_option][name]['tr'][trbuff_index:], temperature=temp, hard=True)
                        tr_index = tr_hard.argmax()
                        tr = space['tr'][tr_index + trbuff_index]
                        active_list.append(tr_soft[tr_index])

                        tc_hard, tc_soft = gumbel_softmax(params_dict[quant_option][name]['tc'][tcbuff_index:], temperature=temp, hard=True)
                        tc_index = tc_hard.argmax()
                        tc = space['tc'][tc_index + tcbuff_index]
                        active_list.append(tc_soft[tc_index])

                        tm_hard, tm_soft = gumbel_softmax(params_dict[quant_option][name]['tm'][tmbuff_index:], temperature=temp, hard=True)
                        tm_index = tm_hard.argmax()
                        tm = space['tm'][tm_index + tmbuff_index]
                        active_list.append(tm_soft[tm_index])

                        tn_hard, tn_soft = gumbel_softmax(params_dict[quant_option][name]['tn'][tnbuff_index:], temperature=temp, hard=True)
                        tn_index = tn_hard.argmax()
                        tn = space['tn'][tn_index + tnbuff_index]
                        active_list.append(tn_soft[tn_index])

                        # print([comp_mode, trbuff, tcbuff, tmbuff, tnbuff, tr, tc, tm, tn])
                        # input()
                        option_list.append([comp_mode, trbuff, tcbuff, tmbuff, tnbuff, tr, tc, tm, tn])

                    input_params_set["a0q"+str(quant_option)] = option_list[0] + [quant_option]
                    input_params_set["a1q"+str(quant_option)] = option_list[1] + [quant_option]
                    input_params_set["dwa0q"+str(quant_option)] = option_list[2] + [quant_option]
                    input_params_set["dwa1q"+str(quant_option)] = option_list[3] + [quant_option]

                for accel in input_params_set.keys():
                    if input_params_set[accel][1] < input_params_set[accel][5] or\
                       input_params_set[accel][2] < input_params_set[accel][6] or\
                       input_params_set[accel][3] < input_params_set[accel][7] or\
                       input_params_set[accel][4] < input_params_set[accel][8]:
                        design_choice_integrity=False
                        break
                    else:
                        design_choice_integrity=True

        else:
            print('Wrong mode:', mode)
            sys.exit()
        #print(input_params_set)
        #can be capsuled


        #!!!!!
        #pay attention to the format of input_params_set there is a [quant_option] in the end

        #!!!!
        #bottleneck_latency and block_wise_performance are what you want

        valid = True
        try:
            bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
            layer_wise_break_down,consumption_used, consumption_breakdown,\
            accelerator_alloc,bs,block_wise_performance,net_struct=capsuled_predictor(input_params_set, copy.deepcopy(block_info), copy.deepcopy(quant_info),cifar=cifar,edd=edd)

        except Exception as e:
            print(e)
            valid = False
            pass

        if valid:
            if mode == 'differentiable':
                loss = bottleneck_latency/200e6

                for active_option in active_list:
                    loss = loss * ((1 - active_option).detach() + active_option)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lr_policy.step()

            if 1/(bottleneck_latency/200e6)> best_throughput: 
                best_throughput=1/(bottleneck_latency/200e6)
                best_consumption_used=consumption_used
                best_consumption_breakdown=consumption_breakdown
                best_latency_break_down=latency_break_down
                best_input_params_set=input_params_set
                best_accelerator_alloc=accelerator_alloc
                best_net_struct=net_struct
                best_bs=bs
                best_layer_wise_break_down=layer_wise_break_down
                best_block_wise_performance = block_wise_performance

            pbar.set_description('Iter[%d/%d] - Current bottleneck latency: %.3f, Current throughput:%.2f, Best throughput: %.2f' % (iter_id, iteration, bottleneck_latency/200e6, 1/(bottleneck_latency/200e6), best_throughput))

    return best_input_params_set, best_throughput, best_block_wise_performance

    # print('throughput: ', best_throughput)
    # print('best_bs: ', best_bs)
    # print('latency_break_down: ', best_latency_break_down)
    # print('layer_wise_break_down: ',best_layer_wise_break_down)
    # print('consumption_used: ', best_consumption_used)
    # print('consumption_breakdown: ', best_consumption_breakdown)
    # print('accelerator_alloc', best_accelerator_alloc)
    # print('input_params',best_input_params_set)
    # print('net_struct', net_struct)


def evaluate_latency(block_info, quant_info, input_params_set, cifar=True, edd=False):
    try:
        bottleneck_latency, latency_break_down,layer_wise_break_down_to_accel,\
        layer_wise_break_down,consumption_used, consumption_breakdown,\
        accelerator_alloc,bs,block_wise_performance,net_struct=capsuled_predictor(input_params_set, copy.deepcopy(block_info), copy.deepcopy(quant_info),cifar=cifar,edd=edd)

    except Exception as e:
        print('Invalid hardware.')
        sys.exit()

    throughput = 1/(bottleneck_latency/200e6)

    return throughput, block_wise_performance


if __name__ == '__main__':
    block_info = ['k5_e6', 'k5_e6', 'k3_e1', 'k5_e3', 'k5_e3', 'k5_e6', 'k5_e1_g2', 'k5_e1', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e1_g2', 'k3_e6', 'k5_e6', 'k3_e3', 'k5_e6', 'k5_e6', 'k5_e3', 'k5_e6', 'k5_e3', 'k5_e6']
    quant_info = [6, 8, 6, 6, 4, 8, 6, 6, 6, 8, 6, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6]
    
    best_input_params_set, best_throughput, block_wise_performance = search_for_best_latency(block_info, quant_info, block_options=BLOCK_OPTIONS, quant_options=QUANT_OPTIONS, 
                                                                                            cifar=True, edd=False, iteration=10000, mode='random', fix_comp_mode=True, temp=1)
    
    print('best throughput:', best_throughput)
    print('Layerwise latency', block_wise_performance)

    print('Evaluation...')
   
    throughput, block_wise_performance = evaluate_latency(block_info, quant_info, best_input_params_set, cifar=True, edd=False)

    print('Evaluate throughput:', throughput)
    print('Evaluate layerwise latency', block_wise_performance)