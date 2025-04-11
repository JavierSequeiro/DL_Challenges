import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class ConvBLock(nn.Module):

    def __init__(self, in_feat, out_feat, k_size, stride=1,padding=True, pool=False,act=nn.ReLU()):
        super(ConvBLock, self).__init__()
        pad_cond = 1 if padding else 0
        # if padding:
        #     pad_cond = 1
        # else:
        #     pad_cond = 0
        
        self.conv1  = nn.Conv2D(in_feat, out_feat, k_size, stride, padding=1)
        self.bn1 = nn.BatchNorm2D(out_feat)
        # self.act = nn.ReLU() if act=="relu" else nn.LeakyReLU(negative_slope=0.01)
        self.act=act
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        return x
    
class Classifier(nn.Module):
    def __init__(self,in_feat, out_feat, act=nn.ReLU()):
        super(Classifier, self).__init()
        self.fc1= nn.Linear(in_feat, out_feat)

    def forward(self,x):
        return self.fc1(x)
    
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1,self).__init__()
        self.conv1 = ConvBLock(in_feat=1, out_feat=32, k_size=3,padding=1, act=nn.ReLU(), pool=True)
        self.conv2 = ConvBLock(in_feat=32, out_feat=64, k_size=3,padding=1, act=nn.ReLU(), pool=True)
        self.conv3 = ConvBLock(in_feat=64, out_feat=128, k_size=3,padding=1, act=nn.ReLU(), pool=True)
        self.conv4 = ConvBLock(in_feat=128, out_feat=256, k_size=3,padding=1, act=nn.AdaptiveAvgPool2d(1), pool=True)
        self.fc = Classifier(in_feat=256, out_feat=15)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

    