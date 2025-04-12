
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
# from .utils.config import *
# from .utils.config import Config
from utils.config import Config
from train import train_fn
from test_ import test_fn 
# from .train import *
# from .test import * 

# data_path = "./DL_Challenges/C1/data/15-Scene"
data_path = "./data/15-Scene"
cfg = Config()
scene_dict = {0:"bedroom",
              1:"suburb",
              2:"industrial",
              3:"kitchen",
              4:"living room",
              5:"coast",
              6:"forest",
              7:"highway",
              8:"inside city",
              9:"mountain",
              10:"open country",
              11:"street",
              12:"tall building",
              13:"office",
              14:"store"}
print(sorted(os.listdir(data_path)))
im_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

# train_size, val_size, test_size = [0.8, 0.1, 0.1]
dataset = datasets.ImageFolder(root=data_path, transform=im_transform)

dataset_size = len(dataset)
train_size = int(cfg.train_size*dataset_size)
val_size = int(cfg.val_size*dataset_size)
test_size = dataset_size - train_size - val_size

train_set, val_set, test_set = random_split(dataset,[train_size, val_size, test_size],generator=torch.Generator().manual_seed(14))

imgs, labels = val_set[70]
print(imgs.shape)
print(f'Train Set: {len(train_set)} Samples')
print(f'Val Set: {len(val_set)} Samples')
print(f'Test Set: {len(test_set)} Samples')

train_loader = DataLoader(train_set, batch_size=cfg.batch_size,shuffle=True)
val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)


model, train_loss, val_loss, train_metrics, val_metrics = train_fn(train_loader=train_loader,
                                                                   val_loader=val_loader,
                                                                   cfg=cfg)

test_metrics = test_fn(model=model,
                       test_loader=test_loader,
                       cfg=cfg)

# imgs, labels = val_set[70]
# imgs = imgs.permute(1,2,0)
# plt.imshow(imgs)
# plt.title(scene_dict[labels])
# plt.show()
