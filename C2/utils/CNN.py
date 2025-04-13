import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class ConvBLock(nn.Module):

    def __init__(self, in_feat, out_feat, k_size, stride=1, padding=True, pool=False, batch_norm=True, pooling_layer=nn.MaxPool2d(kernel_size=2), act=nn.ReLU()):
        super(ConvBLock, self).__init__()
        pad_cond = 1 if padding else 0
        # if padding:
        #     pad_cond = 1
        # else:
        #     pad_cond = 0
        
        self.conv1  = nn.Conv2d(in_feat, out_feat, k_size, stride, padding=pad_cond)
        self.bn1 = nn.BatchNorm2d(out_feat) if batch_norm else nn.Identity()
        # self.act = nn.ReLU() if act=="relu" else nn.LeakyReLU(negative_slope=0.01)
        self.act=act
        self.pool = pooling_layer if pool else nn.Identity()

    def forward(self,x):
        x_ini = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        return x
    
class Classifier(nn.Module):
    def __init__(self,in_feat, out_feat, act=nn.ReLU()):
        super(Classifier, self).__init__()
        self.fc1= nn.Linear(in_feat, out_feat)

    def forward(self,x):
        return self.fc1(x)
    
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1,self).__init__()
        self.conv1 = ConvBLock(in_feat=3, out_feat=32, k_size=3,padding=1, act=nn.ReLU(), pool=True)
        self.conv2 = ConvBLock(in_feat=32, out_feat=64, k_size=3,padding=1, act=nn.ReLU(), pool=True)
        self.conv3 = ConvBLock(in_feat=64, out_feat=128, k_size=3,padding=1, act=nn.ReLU(), pool=True)
        self.conv4 = ConvBLock(in_feat=128, out_feat=256, k_size=3,padding=1, act=nn.ReLU(), pool=True, pooling_layer=nn.AdaptiveAvgPool2d((1,1)))
        self.fc = Classifier(in_feat=256, out_feat=15)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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

    