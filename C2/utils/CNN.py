import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class ConvBlock(nn.Module):

    def __init__(self, in_feat, out_feat, k_size, stride=1, padding=True, pool=False, batch_norm=True, pooling_layer=nn.MaxPool2d(kernel_size=2), act=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        pad_cond = 1 if padding else 0
        
        self.conv1  = nn.Conv2d(in_feat, out_feat, k_size, stride, padding=pad_cond)
        self.bn1 = nn.BatchNorm2d(out_feat) if batch_norm else nn.Identity()
        self.act=act
        self.pool = pooling_layer if pool else nn.Identity()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        return x
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_feat, out_feat, k_size, stride=1):
        super(UpSampleBlock, self).__init__()
        self.up=nn.ConvTranspose2d(in_channels=in_feat, out_channels=out_feat,kernel_size=k_size, stride=stride)
        self.conv = nn.Sequential(ConvBlock(in_feat, out_feat, k_size),
                                  ConvBlock(out_feat, out_feat, k_size))
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Adjust Sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        
        x = torch.cat([x2, x1], dim=1) # Skip Connection
        return self.conv(x)
    

class DownSampleBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Sequential(ConvBlock(in_feat, out_feat, k_size=3, stride=1, padding=True, pool=False),
                                  ConvBlock(out_feat, out_feat, k_size=3, padding=True, pool=True))
        

    def forward(self, x):
        return self.conv(x)
    
# class Classifier(nn.Module):
#     def __init__(self,in_feat, out_feat, act=nn.ReLU()):
#         super(Classifier, self).__init__()
#         self.fc1= nn.Linear(in_feat, out_feat)

#     def forward(self,x):
#         return self.fc1(x)
    
class UNet1(nn.Module):
    def __init__(self):
        super(UNet1,self).__init__()
        self.d1 = nn.Sequential(ConvBlock(in_feat=1, out_feat=64, k_size=3), 
                                ConvBlock(in_feat=64, out_feat=64, k_size=3)) #512
        self.d2 = DownSampleBlock(in_feat=64, out_feat=128) #256
        self.d3 = DownSampleBlock(in_feat=128, out_feat=256) #128
        self.d4 = DownSampleBlock(in_feat=256, out_feat=512) #64
        self.d5 = DownSampleBlock(in_feat=512, out_feat=1024) #32
        self.up1 = UpSampleBlock(in_feat=1024,out_feat=512, k_size=3, stride=1) #64
        self.up2 = UpSampleBlock(in_feat=512,out_feat=256, k_size=3, stride=1)# 128
        self.up3 = UpSampleBlock(in_feat=256,out_feat=128, k_size=3, stride=1) #256
        self.up4 = UpSampleBlock(in_feat=128,out_feat=64, k_size=3, stride=1) #512
        self.out = nn.Sequential(nn.Conv2d(64,1,kernel_size=1),
                                 nn.Sigmoid())

    def forward(self,x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        x5 = self.d5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, k_size=3, stride=1, padding=1, pool=False, batch_norm=True, act=nn.ReLU(), pooling_layer=nn.MaxPool2d(2)):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size=k_size, stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_feat) if batch_norm else nn.Identity()
        self.act = act
        self.pool = pooling_layer if pool else nn.Identity()

        self.skip = nn.Identity()
        if in_feat != out_feat or stride != 1:
            self.skip = nn.Sequential(nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride),
                                      self.bn)
            

    def forward(self,x):
        x_initial = self.skip(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        if x.shape == x_initial.shape:
            x += x_initial

        return x

    