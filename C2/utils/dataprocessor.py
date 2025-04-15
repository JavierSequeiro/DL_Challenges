import os
import glob
import cv2
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import random

class DataProcessor():
    def __init__(self, root_path):
        self.root_path = root_path


    def image_processor(self, img_folder, img_resolution):
        imgs_path = os.path.join(self.root_path, img_folder)
        imgs_dirs = sorted(os.listdir(imgs_path)[:100])
        print(imgs_dirs)
        # Load Image --> Convert to RGB --> Convert to PIL Image
        # imgs = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(imgs_path,img_dir), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)) for img_dir in imgs_dirs]
        imgs = [Image.fromarray(cv2.imread(os.path.join(imgs_path,img_dir), cv2.IMREAD_GRAYSCALE)) for img_dir in imgs_dirs]

        img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(img_resolution)])
        # Convert to Tensor + Resize
        imgs = [img_transform(img) for img in imgs]

        return imgs

    def mask_processor(self, mask_folder, img_resolution):
        masks_path = os.path.join(self.root_path, mask_folder)
        masks_dirs_l = sorted(os.listdir(os.path.join(masks_path, "leftMask"))[:100])
        masks_dirs_r = sorted(os.listdir(os.path.join(masks_path, "rightMask"))[:100])
        print(masks_dirs_l)
        print(masks_dirs_r)

        mask_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(img_resolution)])

        masks_l = [mask_transform(Image.fromarray(cv2.imread(os.path.join(masks_path, "leftMask", mask_dir), cv2.IMREAD_GRAYSCALE))) for mask_dir in masks_dirs_l]
        masks_r = [mask_transform(Image.fromarray(cv2.imread(os.path.join(masks_path, "rightMask", mask_dir), cv2.IMREAD_GRAYSCALE))) for mask_dir in masks_dirs_r]
        masks = [masks_l[i] + masks_r[i] for i in range(len(masks_l))]
        return masks

    def get_torch_dataset(self, imgs, masks):
        return SegmentationDataset(imgs=imgs, masks=masks)

    def get_torch_dataloaders(self, dataset, cfg):
        
        train_len = int(cfg.train_size*len(dataset))
        val_len = int(cfg.val_size*len(dataset))
        test_len = len(dataset) - train_len - val_len

        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(14))
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def visualize_imgs_masks(self, imgs, masks):
        n = random.randint(0,len(imgs)-1)
        print(f"Showing Sample {n}")

        merged = 0.7*imgs[n] + 0.3*masks[n]
        plt.subplot(1,3,1)
        plt.imshow(imgs[n].permute(1,2,0).numpy(), cmap="gray")
        plt.title("XRay")
        plt.subplot(1,3,2)
        plt.imshow(masks[n].permute(1,2,0).numpy(), cmap="gray")
        plt.title("Segmentation Mask")
        plt.subplot(1,3,3)
        plt.imshow(merged.permute(1,2,0).numpy(),cmap="gray")
        plt.title("Fused Mask + XRay")
        plt.show()


class SegmentationDataset(Dataset):
    def __init__(self, imgs, masks):
        super().__init__()
        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index], self.masks[index]
        