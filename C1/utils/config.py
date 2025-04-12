from .CNN import CNN1
import torch.nn as nn
import torch
import torch.optim as optim

class Config:
    model = CNN1()
    loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3,weight_decay=1e-5)

    num_epochs = 10
    batch_size = 32
    train_size = 0.9
    val_size = (1 - train_size)/2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes= 15