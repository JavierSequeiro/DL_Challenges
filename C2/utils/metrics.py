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
    
    def dice_coeff(self):
        epsilon = 1e-6

        sum_dim = (-1, -2) if self.pred.dim() == 2 else (-1, -2, -3)

        inter = 2 * (self.pred * self.target).sum(dim=sum_dim)
        sets_sum = self.pred.sum(dim=sum_dim) + self.target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()
    
    def IoU(self):
        epsilon = 1e-6
        intersection = (self.pred * self.target).sum(dim=(1,2))
        union = (self.pred + self.target - self.pred*self.target).sum(dim=(1,2))
        iou = (intersection + epsilon)/(union + epsilon)
        return iou

def compute_metrics(preds, targets, num_classes):

    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    metrics = Metrics(y_pred=preds, y_target=targets, num_classes=num_classes)
    # acc = metrics.accuracy()
    # precision = metrics.precision()
    # recall = metrics.recall()
    # f1score = metrics.f1score()
    # conf_matrix = metrics.confusion_matrix_()
    dice = metrics.dice_coeff()
    iou = metrics.IoU()
    return {"dice": dice,
            "iou": iou}