from .CNN import UNet1
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import models

class Config:
    model = UNet1()
    # model = models.resnet18(pretrained=True)
    loss = nn.BCELoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-4,weight_decay=1e-5)

    num_epochs = 10
    batch_size = 8
    train_size = 0.8
    val_size = (1 - train_size)/2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes= 2