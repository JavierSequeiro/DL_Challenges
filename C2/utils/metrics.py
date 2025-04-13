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
    
    def confusion_matrix_(self):
    #     # _, preds = torch.max(outputs, 1)  # Shape: (B,)
    
    #     # Initialize confusion matrix
    #     confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        
    #     # Populate confusion matrix
    #     for t, p in zip(self.target.view(-1), self.preds.view(-1)):
    #         confusion_matrix[t.long(), p.long()] += 1

    #     self.print_confusion_matrix(confusion_matrix)
        
    #     return confusion_matrix


    # def print_confusion_matrix(self, confusion_matrix):
    #     # Print confusion matrix in a readable format
    #     print("Confusion Matrix:")
    #     for row in confusion_matrix:
    #         print("\t".join(str(x.item()) for x in row))
        return confusion_matrix(self.target, self.pred)


def compute_confusion_matrix(outputs, labels, num_classes):
    # Get predicted class labels from outputs
    _, preds = torch.max(outputs, 1)  # Shape: (B,)
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    # Populate confusion matrix
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix

def print_confusion_matrix(confusion_matrix):
        # Print confusion matrix in a readable format
        print("Confusion Matrix:")
        for row in confusion_matrix:
            print("\t".join(str(x.item()) for x in row))

def compute_metrics(preds, targets, num_classes):

    preds, targets = preds.cpu().numpy(), targets.cpu().numpy()
    metrics = Metrics(y_pred=preds, y_target=targets, num_classes=num_classes)
    acc = metrics.accuracy()
    precision = metrics.precision()
    recall = metrics.recall()
    f1score = metrics.f1score()
    conf_matrix = metrics.confusion_matrix_()

    return {"accuracy": acc,
            "precision": precision,
            "recall":recall,
            "f1 score":f1score,
            "confusion matrix":conf_matrix}