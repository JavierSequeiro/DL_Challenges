import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

class Metrics():
    
    def __init__(self, y_pred, y_target, num_classes):
        self.pred = y_pred
        self.target = y_target
        self.num_classes = num_classes


    def accuracy(self):
        return accuracy_score(self.target, self.pred)
    
    def precision(self):
        return precision_score(self.target, self.pred, average="macro", zero_division=0)
    
    def recall(self):
        return recall_score(self.target, self.pred,  average="macro", zero_division=0)
    
    def f1score(self):
        return f1_score(self.target, self.pred, average="macro", zero_division=0)
    
    # def dice_coeff(self):
    #     epsilon = 1e-6

    #     sum_dim = (-1, -2) if self.pred.ndim() == 2 else (-1, -2, -3)

    #     inter = 2 * (self.pred * self.target).sum(dim=sum_dim)
    #     sets_sum = self.pred.sum(dim=sum_dim) + self.target.sum(dim=sum_dim)
    #     sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    #     dice = (inter + epsilon) / (sets_sum + epsilon)
    #     return dice.mean()
    
    
    def dice_coeff(self, smooth=1e-5):
        if isinstance(self.pred, np.ndarray):
            self.pred = torch.from_numpy(self.pred)
        if isinstance(self.target, np.ndarray):
            self.target = torch.from_numpy(self.target)

        prediction = self.pred.float().view(-1)
        label = self.target.float().view(-1)
        
        intersection = (prediction*label).sum()
        total = prediction.sum() + label.sum()
        
        dice_score = (2. * intersection + smooth) / (total + smooth)
        return dice_score.item()

    # def IoU(self):
    #     epsilon = 1e-6
    #     intersection = (self.pred * self.target).sum(dim=(1,2))
    #     union = (self.pred + self.target - self.pred*self.target).sum(dim=(1,2))
    #     iou = (intersection + epsilon)/(union + epsilon)
    #     return iou

    def IoU(self, smooth=1e-5):
        if isinstance(self.pred, np.ndarray):
            self.pred = torch.from_numpy(self.pred)
        if isinstance(self.target, np.ndarray):
            self.target = torch.from_numpy(self.target)

        prediction = self.pred.float().view(-1)
        label = self.target.float().view(-1)
        
        intersection = (prediction*label).sum()
        total = prediction.sum() + label.sum()
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()

def compute_metrics(preds, targets, num_classes):

    # preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    preds, targets = preds.cpu(), targets.cpu()
    metrics = Metrics(y_pred=preds, y_target=targets, num_classes=num_classes)

    dice = metrics.dice_coeff()
    iou = metrics.IoU()
    return {"dice": dice,
            "iou": iou}