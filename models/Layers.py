#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:30:49 2020

@author: nickwang
"""

import torch
import torch.nn as nn

class Separable_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, depth_multiplier=8):
        super(Separable_Conv2d, self).__init__()
        self.depth_multiplier = depth_multiplier
        in_channels *= depth_multiplier
        self.conv_h = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
        self.conv_w = nn.Conv2d(in_channels, out_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = x.repeat(1, self.depth_multiplier, 1, 1)
        x = self.conv_h(x)
        x = self.conv_w(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.output = output
        if self.output == False:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.output:
            return x
        else:
            x = self.bn(x)
            x = self.relu(x)
            return x
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x, 1)

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    def forward(self, x):
        return torch.squeeze(x)    
    