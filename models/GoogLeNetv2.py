#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

import torch
import torch.nn as nn
from .Layers import Separable_Conv2d, Conv2d, Squeeze
from functools import partial
# from torchsummary import summary


class GoogLeNetv2(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(GoogLeNetv2, self).__init__()
        self.num_classes = num_classes
        self.mode = mode     
        self.layers = nn.Sequential(
            Separable_Conv2d(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=1),
            Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # 28x28
            Inceptionv2(192, 64, 64, 64, 64, 96, 32), # 3a
            Inceptionv2(256, 64, 64, 96, 64, 96, 64), # 3b
            Inceptionv2(320, 0, 128, 160, 64, 96, 0, 'MAX', stride=2), # 3c
            
            Inceptionv2(576, 224, 64, 96, 96, 128, 128), # 4a
            Inceptionv2(576, 192, 96, 128, 96, 128, 128), # 4b
            Inceptionv2(576, 128, 128, 160, 128, 160, 128), # 4c
            Inceptionv2(576, 64, 128, 192, 169, 192, 128), # 4d
            Inceptionv2(576, 0, 128, 192, 192, 256, 0, 'MAX', stride=2), # 4e
            
            Inceptionv2(1024, 352, 192, 320, 160, 224, 128), # 5a
            Inceptionv2(1024, 352, 192, 320, 192, 224, 128, 'MAX'), # 5b
            nn.AvgPool2d(7, 1),
            Conv2d(1024, num_classes, kernel_size=1, output=True),
            Squeeze(),
        ) 
        if mode == 'train':
            self.aux1 = InceptionAux(576, num_classes)
            self.aux2 = InceptionAux(576, num_classes)

    def forward(self, x):   
        for idx, layer in enumerate(self.layers):
            if(idx == 9 and self.mode == 'train'):
                aux1 = self.aux1(x)
            elif(idx == 12 and self.mode == 'train'):  
                aux2 = self.aux2(x)
            x = layer(x)
        if self.mode == 'train':
            return x, aux1, aux2
        else:
            return x
    
    def init_weights(self, init_mode='VGG'):
        def init_function(m, init_mode):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if init_mode == 'VGG':
                    torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                elif init_mode == 'XAVIER': 
                    fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    std = (2.0 / float(fan_in + fan_out)) ** 0.5
                    a = (3.0)**0.5 * std
                    with torch.no_grad():
                        m.weight.uniform_(-a, a)
                elif init_mode == 'KAMING':
                     torch.nn.init.kaiming_uniform_(m.weight)
                
                m.bias.data.fill_(0)    
        _ = self.apply(partial(init_function, init_mode=init_mode))
    
class Inceptionv2(nn.Module):
    def __init__(self, input_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel, conv3_double_reduce_channel, conv3_double_channel, pool_reduce_channel, pool_type='AVG', stride=1):
        '''
        pool_type : TYPE, ['AVG', 'MAX']
            DESCRIPTION. The default is 'AVG'.

        '''
        super(Inceptionv2, self).__init__()
        self.stride = stride
        if conv1_channel != 0:
            self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)
        else:
            self.conv1 = None 
        self.conv3_reduce = Conv2d(input_channel, conv3_reduce_channel, kernel_size=1)
        self.conv3 = Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, stride=stride, padding=1)
        self.conv3_double_reduce = Conv2d(input_channel, conv3_double_reduce_channel, kernel_size=1)
        self.conv3_double1 = Conv2d(conv3_double_reduce_channel, conv3_double_channel, kernel_size=3, padding=1)
        self.conv3_double2 = Conv2d(conv3_double_channel, conv3_double_channel, kernel_size=3, stride=stride, padding=1)
        if pool_type == 'MAX':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        elif pool_type == 'AVG':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        if pool_reduce_channel != 0:
            self.pool_reduce = Conv2d(input_channel, pool_reduce_channel, kernel_size=1)
        else:
            self.pool_reduce = None
    
    def forward(self, x):
        if self.conv1 != None:
            output_conv1 = self.conv1(x)
        else:
            output_conv1 = torch.zeros([x.shape[0], 0, x.shape[2]//self.stride, x.shape[3]//self.stride]).cuda()
        output_conv3 = self.conv3(self.conv3_reduce(x))
        output_conv3_double = self.conv3_double2(self.conv3_double1(self.conv3_double_reduce(x)))
        if self.pool_reduce != None:
            output_pool = self.pool_reduce(self.pool(x))
        else:
            output_pool = self.pool(x)
        outputs = torch.cat([output_conv1, output_conv3, output_conv3_double, output_pool], dim=1)
        return outputs  

class InceptionAux(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(InceptionAux, self).__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(5, 3),
            Conv2d(input_channel, 128, 1),
            Conv2d(128, 1024, kernel_size=4),
            Conv2d(1024, num_classes, kernel_size=1, output=True),
            Squeeze()
            )
    
    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    net = GoogLeNetv2(1000)
    # summary(net, (3, 224, 224))