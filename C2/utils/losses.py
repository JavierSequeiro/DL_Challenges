import torch
import torch.nn.functional as F
import numpy as np

def dice_loss(preds, targets, smooth=1e-6):
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        prediction = preds.float().view(-1)
        label = targets.float().view(-1)
        
        intersection = (prediction*label).sum()
        total = prediction.sum() + label.sum()
        
        dice_score = (2. * intersection + smooth) / (total + smooth)
        return 1 - dice_score