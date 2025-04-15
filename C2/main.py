
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from utils.dataprocessor import DataProcessor
# from .utils.config import *
# from .utils.config import Config
from utils.config import Config
from train import train_fn
from test_ import test_fn, test_fn_with_tta
from utils.xAI import gradcam
import cv2
# from .train import *
# from .test import * 

cfg = Config() # Experiment Config
print(f'Device: {cfg.device}')

root_path = "./DL_Challenges/C2/data/Montgomery/MontgomerySet"
# root_path = "./data/Montgomery/MontgomerySet"
dataprocessor = DataProcessor(root_path=root_path)

img_folder = "CXR_png"
mask_folder = "ManualMask"
resolution = (512,512)

imgs = dataprocessor.image_processor(img_folder=img_folder,
                                     img_resolution=resolution)
print(f'Number of Images: {len(imgs)}')

masks = dataprocessor.mask_processor(mask_folder=mask_folder,
                                     img_resolution=resolution)

dataset = dataprocessor.get_torch_dataset(imgs, masks)
train_loader, val_loader, test_loader = dataprocessor.get_torch_dataloaders(dataset=dataset, cfg=cfg)

print(f'Train Dataloader Size: {len(train_loader)}')
print(f'Validation Dataloader Size: {len(val_loader)}')
print(f'Test Dataloader Size: {len(test_loader)}')

# dataprocessor.visualize_imgs_masks(imgs=imgs, masks=masks)

# Train the model
model, train_loss, val_loss, train_metrics, val_metrics = train_fn(train_loader=train_loader,
                                                                   val_loader=val_loader,
                                                                   cfg=cfg)

# Test the model
print("Entering Testing Phase...")
test_metrics = test_fn(model=model,
                       test_loader=test_loader,
                       cfg=cfg)

# test_metrics_tta = test_fn_with_tta(model=model,
#                        test_loader=test_loader,
#                        cfg=cfg)

# gradcam(model=model,
#         test_loader=test_loader,
#         classes_names=scene_dict,
#         cfg=cfg)

# Plotting if needed

# imgs, labels = val_set[70]
# imgs = imgs.permute(1,2,0)
# plt.imshow(imgs)
# plt.title(scene_dict[labels])
# plt.show()
