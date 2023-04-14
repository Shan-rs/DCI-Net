#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:02:08 2021

@author: ws-512
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.dcn import DeformableConv2d as DeformConv2d

class DownTransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownTransitionBlock, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=True)
#        self.pool = nn.Conv2d(out_planes, out_planes, 3, stride = 2, padding=1, bias=True)#
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        out = self.pool(self.conv(self.relu(x)))
        return out
#
class UpTransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(UpTransitionBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=True)
        self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
    def forward(self, y, x):
        out1 = self.relu(self.conv(x))
        out2  = F.interpolate(out1, size=y.size()[2:], mode='bilinear',align_corners=True)
        out = self.relu(self.conv2(torch.cat([out2, y], 1)))
        return out

class DynamicMutualEnhancement(nn.Module):
    def __init__(self, l_inchannel):
        super(DynamicMutualEnhancement, self).__init__()

        self.convl1 = nn.Conv2d(l_inchannel, 2*l_inchannel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.convl2 = nn.Conv2d(l_inchannel, 2*l_inchannel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.convh1 = nn.Conv2d(2*l_inchannel, l_inchannel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.convh2 = nn.Conv2d(2*l_inchannel, l_inchannel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.Deformconvl1 = DeformConv2d(2*l_inchannel, 2*l_inchannel)#, modulation=False
        self.Deformconvl2 = DeformConv2d(2*l_inchannel, 2*l_inchannel)
        self.Deformconvl3 = DeformConv2d(2*l_inchannel, 2*l_inchannel)
        self.Deformconvl4 = DeformConv2d(2*l_inchannel, 2*l_inchannel)
        self.Deformconvh1 = DeformConv2d(l_inchannel, l_inchannel)#, modulation=False
        self.Deformconvh2 = DeformConv2d(l_inchannel, l_inchannel)
        self.Deformconvh3 = DeformConv2d(l_inchannel, l_inchannel)
        self.Deformconvh4 = DeformConv2d(l_inchannel, l_inchannel)
        self.conv3 = nn.Conv2d(4*l_inchannel, l_inchannel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv4 = nn.Conv2d(2*l_inchannel, l_inchannel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.conv5 = nn.Conv2d(2*l_inchannel, l_inchannel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        #nn.Conv2d(l_inchannel, l_inchannel, 3, stride = 2, padding=1, bias=True)
#        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
    def forward(self, xl, xh):
        
        x1 = self.relu(self.convl1(self.down(xl)))
        x1 = self.Deformconvl1(x1)
        x1 = self.Deformconvl2(x1)
        xlh1 = self.relu(x1*xh)
        xlh1  = F.interpolate(xlh1, size=xl.size()[2:], mode='bilinear',align_corners=True)
#        xlh1 = xlh1[:,:, :wl, :hl]
        x2 = self.relu(self.convl2(xl))
        x2 = self.Deformconvl3(x2)
        x2 = self.Deformconvl4(x2)

        xh2  = F.interpolate(xh, size=xl.size()[2:], mode='bilinear',align_corners=True)
        xlh2 = self.relu(x2*xh2)
        xlh = self.relu(self.conv3(torch.cat([xlh1, xlh2], 1)))
        
#        x3 = self.up(xh)
        x3  = F.interpolate(xh, size=xl.size()[2:], mode='bilinear',align_corners=True)
#        x3 = x3[:,:, :wl, :hl]
        x3 = self.relu(self.convh1(x3)) 
        x3 = self.Deformconvh1(x3)
        x3 = self.Deformconvh2(x3)
        xhl1 = self.relu(x3*xl)
        x4 = self.relu(self.convh2(xh)) 
        x4 = self.Deformconvh3(x4)
        x4 = self.Deformconvh4(x4)
        xl2 = self.down(xl)

        xhl2  = F.interpolate(self.relu(x4*xl2), size=xl.size()[2:], mode='bilinear',align_corners=True)
        xhl = self.relu(self.conv4(torch.cat([xhl1, xhl2], 1)))
        
        out = self.relu(self.conv5(torch.cat([xlh, xhl], 1)))
        return out

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class MidTransmitionLayer(nn.Module):
    def __init__(self, channel):
        super(MidTransmitionLayer, self).__init__()
        self.ca = CALayer(channel)
        self.pa = PALayer(channel)
        self.MTL = nn.Sequential(
                
                nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x1 = self.ca(x)
        x1 = self.pa(x1)
        y = self.MTL(x1)
        y = y + x
        return y

class REC(nn.Module):
    def __init__(self, num_init_features, out_c=12):
        super(REC, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(num_init_features, num_init_features//2, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(num_init_features//2, num_init_features//2, kernel_size=1, stride=1, padding=0, bias=True)
        self.deconv2 = nn.ConvTranspose2d(num_init_features//2, out_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, padding=0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x = self.lrelu(self.deconv1(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.deconv2(x))
        x = self.conv2(x)
        return x

class SpectrumAwareAggregation(nn.Module):
    def __init__(self, ):
        super(SpectrumAwareAggregation, self).__init__()
#        self.conv1 = nn.Conv2d(2*l_inchannel, l_inchannel, kernel_size=1, stride=1, padding=0, bias=True)
#        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.post_precess = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 3, padding=1, bias=True),
)
    def forward(self, x, y): #RGB
#        print(x[:,0,:,:].unsqueeze(1).size())
        xR = x[:,0,:,:].unsqueeze(1)
        xG = x[:,1,:,:].unsqueeze(1)
        xB = x[:,2,:,:].unsqueeze(1)
        yR = y[:,0,:,:].unsqueeze(1)
        yG = y[:,1,:,:].unsqueeze(1)
        yB = y[:,2,:,:].unsqueeze(1)
        
        R = self.softmax(self.avg_pool(torch.cat([xR, yR], 1)))
        G = self.softmax(self.avg_pool(torch.cat([xG, yG], 1)))
        B = self.softmax(self.avg_pool(torch.cat([xB, yB], 1)))
        
        Rw=R.view(-1,2,1)[:,:,:,None,None]
        outR=Rw[:,0,::]*xR+Rw[:,1,::]*yR
        
        Gw=G.view(-1,2,1)[:,:,:,None,None]
        outG=Gw[:,0,::]*xG+Gw[:,1,::]*yG
        
        Bw=B.view(-1,2,1)[:,:,:,None,None]
        outB=Bw[:,0,::]*xB+Bw[:,1,::]*yB
        
        out = torch.cat([outR, outG, outB], 1)
        out = self.post_precess(out)
        return out

