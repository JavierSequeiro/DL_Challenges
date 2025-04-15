from .CNN import UNet1, ResUNet1
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models

class Config:
    model = ResUNet1()
    # model = models.resnet18(pretrained=True)
    loss = nn.BCEWithLogitsLoss()
    # loss = nn.BCELoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3,weight_decay=1e-5)

    num_epochs = 20
    batch_size = 4
    train_size = 0.8
    val_size = (1 - train_size)/2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes= 2
    threshold = 0.8