# import os
# from torchvision import datasets, transforms
# from .utils.config import *
# from .utils.metrics import *
from utils.config import *
from utils.metrics import *

def train_fn(train_loader, val_loader, cfg):
    
    print("Starting training process...")
    model = cfg.model.to(cfg.device)
    optimizer, criterion = cfg.optimizer, cfg.loss
    ovall_train_loss = []
    ovall_val_loss = []
    th = cfg.threshold

    for epoch in range(cfg.num_epochs):
        epoch_train_loss, epoch_val_loss = 0.0, 0.0
        print(f"Epoch {epoch+1}/{cfg.num_epochs}")
        model.train()

        train_metrics = {"dice":0.0,
                       "iou":0.0}
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(cfg.device).float(), labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            # outputs = torch.argmax(outputs, dim=1)
            labels = (labels >th).float()
            outputs = (outputs >th).float()

            all_metrics = compute_metrics(outputs, labels, num_classes=cfg.num_classes)
            for met in train_metrics:
                    train_metrics[met] += all_metrics[met]
            epoch_train_loss += loss.item()

        train_metrics = {key: train_metrics[key] / len(train_loader) for key in train_metrics}

        val_metrics = {"dice":0.0,
                       "iou":0.0}
        
        with torch.no_grad():
            model.eval()
            for j, (images, labels) in enumerate(val_loader):
                images, labels = images.to(cfg.device).float(), labels.to(cfg.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)

                # outputs = torch.argmax(outputs, dim=1)
                labels = (labels >th).float()
                outputs = (outputs >th).float()
                all_metrics = compute_metrics(outputs, labels, num_classes=cfg.num_classes)

                for met in val_metrics:
                    val_metrics[met] += all_metrics[met]

            epoch_val_loss += loss.item()

            val_metrics = {key: val_metrics[key] / len(val_loader) for key in val_metrics}

            

        ovall_train_loss.append(epoch_train_loss/len(train_loader))
        ovall_val_loss.append(epoch_val_loss/len(val_loader))

        print(f"Train Loss: {epoch_train_loss/len(train_loader)}|| Metrics {train_metrics}")
        print(f"Validation Loss: {epoch_val_loss/len(val_loader)} || Metrics {val_metrics}")

    return model, ovall_train_loss, ovall_val_loss, train_metrics, val_metrics