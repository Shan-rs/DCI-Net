#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 15:25:46 2021

@author: ws-512
"""
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
import math
from torch.autograd.variable import Variable
from MyModules import *

def spatial_fold(input, fold):#smaller
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold): #bigger
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )




class ResidualDenseBlock_3C(nn.Module):#
    def __init__(self, nf, bias=True):
        super(ResidualDenseBlock_3C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * 32, nf, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * 32, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
#        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
#        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x3 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf):
        super(RRDB, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0, bias=True)
        self.RDB1 = ResidualDenseBlock_3C(nf)
        self.RDB2 = ResidualDenseBlock_3C(nf)
        self.RDB3 = ResidualDenseBlock_3C(nf)
        self.RDB4 = ResidualDenseBlock_3C(nf)
        self.RDB5 = ResidualDenseBlock_3C(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
#        mm, n1, pp, qq = x.size()
        x1 = self.lrelu(self.conv1(x))
        out1 = self.RDB1(x1)
        out1 = self.RDB2(out1)
        out1 = self.RDB3(out1)
        out1 = self.RDB4(out1)
        out1 = self.RDB5(out1)
        out1 = out1 + x
        return out1

class Encoder(nn.Module):
    def __init__(self,midchannel):
        super(Encoder, self).__init__()
        ############# 256-256  ##############nn.
        self.conv0 = nn.Conv2d(12, midchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(midchannel, midchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        
        ############# Block1-down 256-256  ##############
        self.RRDB_block1 = RRDB(midchannel)
        self.downtrans_block1 = DownTransitionBlock(midchannel, 2*midchannel)

        ############# Block2-down 128-128  ##############
        self.RRDB_block2 = RRDB(2*midchannel)
        self.downtrans_block2 = DownTransitionBlock(2*midchannel, 4*midchannel)
        
        ############# Block3-down 64-64  ##############
        self.RRDB_block3 = RRDB(4*midchannel)
        self.downtrans_block3 = DownTransitionBlock(4*midchannel, 8*midchannel)
        
        ############# Block4-down 32-32  ##############
        self.RRDB_block4 = RRDB(8*midchannel)
        self.downtrans_block4 = DownTransitionBlock(8*midchannel, 16*midchannel)
#        
        ############# Block5-down 16-16  ##############
#        self.RRDB_block5 = RRDB(16*midchannel)
#        self.downtrans_block5 = DownTransitionBlock(16*midchannel, 32*midchannel)
    def forward(self, x):
        x0 = self.relu1(self.conv1(self.relu0(self.conv0(x)))) ## 256x256
#        print(x0.size())
        x1 = self.RRDB_block1(x0) ## 128 X 128
        x2 = self.RRDB_block2(self.downtrans_block1(x1))
        x3 = self.RRDB_block3(self.downtrans_block2(x2))
        x4 = self.RRDB_block4(self.downtrans_block3(x3))
        x5 = self.downtrans_block4(x4)
#        x5 = self.RRDB_block5(self.downtrans_block4(x4))
#        x6 = self.downtrans_block5(x5)
        return x1, x2, x3, x4, x5

class DynamicDecoder(nn.Module):
    def __init__(self, midchannel):
        super(DynamicDecoder, self).__init__()
        ############# 256-256  ##############nn.
        self.encoder = Encoder(midchannel)
        self.MTL = MidTransmitionLayer(16*midchannel)
        
        self.DME1 = DynamicMutualEnhancement(8*midchannel)
#        self.DME1 = UpTransitionBlock(16*midchannel, 8*midchannel)
        self.RRDB_block1 = RRDB(8*midchannel)
        self.recon1 = REC(8*midchannel)
        
        self.DME2 = DynamicMutualEnhancement(4*midchannel)
#        self.DME2 = UpTransitionBlock(8*midchannel, 4*midchannel)
        self.RRDB_block2 = RRDB(4*midchannel)
        self.recon2 = REC(4*midchannel)
        
        self.DME3 = DynamicMutualEnhancement(2*midchannel)
#        self.DME3 = UpTransitionBlock(4*midchannel, 2*midchannel)
        self.RRDB_block3 = RRDB(2*midchannel)
        self.recon3 = REC(2*midchannel)
        
        self.DME4 = DynamicMutualEnhancement(midchannel)
#        self.DME4 = UpTransitionBlock(2*midchannel, midchannel)
        self.RRDB_block4 = RRDB(midchannel)
        self.recon4 = REC(midchannel)
        
        self.SAA1 = SpectrumAwareAggregation()
        self.SAA2 = SpectrumAwareAggregation()
        self.SAA3 = SpectrumAwareAggregation()


    def forward(self, x):
        x = spatial_fold(x, 2)
        l4, l3, l2, l1, l0 = self.encoder(x)
        ##Input [6,3,256,256]; x5 [6, 32, 256, 256]; x4 [6, 64, 128, 128]; x3 [6, 128, 64, 64]
        ##                     x2 [6, 256, 32, 32]; x1 [6, 512, 16, 16]; x0 [6, 1024, 8, 8]
        h0 = self.MTL(l0)
        h1_in = self.DME1(l1, h0)#def forward(self, xl, xh)
        h1 = self.RRDB_block1(h1_in)
        out1 = self.recon1(h1)
        out1 = spatial_unfold(out1, 2)

        h2_in = self.DME2(l2, h1)
        h2 = self.RRDB_block2(h2_in)
        out2 = self.recon2(h2)
        out2 = spatial_unfold(out2, 2)
        out1_re = F.interpolate(out1, size=out2.size()[2:], mode='bilinear', align_corners=True)
        out12 = self.SAA1(out1_re, out2)
        h3_in = self.DME3(l3, h2)
        h3 = self.RRDB_block3(h3_in)
        out3 = self.recon3(h3)
        out3 = spatial_unfold(out3, 2)
        out12_re = F.interpolate(out12, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out123 = self.SAA2(out12_re, out3)
        
        h4_in = self.DME4(l4, h3)
        h4 = self.RRDB_block4(h4_in)
        out4 = self.recon4(h4)
        out4 = spatial_unfold(out4, 2)
        
        out123_re = F.interpolate(out123, size=out4.size()[2:], mode='bilinear', align_corners=True)
        
        out1234 = self.SAA3(out123_re, out4)
        

        return out1, out12, out123, out1234
    










