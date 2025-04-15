import torch
# from .utils.config import *
# from .utils.metrics import *
from utils.config import *
from utils.metrics import *
import torchvision.transforms.functional as TF
import torchvision.transforms
import matplotlib.pyplot as plt
  
def test_fn(model, test_loader, cfg):
    test_metrics = {"dice":0.0,
                    "iou":0.0}
    # criterion = cfg.loss
    model.eval()
    th=cfg.threshold
    with torch.no_grad():
        # confusion_matrix_overall = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.int64)
        for j, (images, labels) in enumerate(test_loader):
            images, labels = images.to(cfg.device).float(), labels.to(cfg.device)
            
            outputs_logits = model(images)
            # loss = criterion(outputs_logits, labels)

            # outputs = torch.argmax(outputs_logits, dim=1)
            labels = (labels >th).float()
            outputs = (outputs_logits >th).float()
            all_metrics = compute_metrics(outputs, labels, num_classes=cfg.num_classes)

            for met in test_metrics:
                test_metrics[met] += all_metrics[met]

        #     conf_mat = compute_confusion_matrix(outputs_logits, labels, cfg.num_classes)
        #     confusion_matrix_overall += conf_mat

        # print_confusion_matrix(confusion_matrix_overall)
        test_metrics = {key: test_metrics[key] / len(test_loader) for key in test_metrics}

        # img, label, pred = images[0], labels[0], outputs[0]
        # img, label, pred = img.cpu().numpy
        img = images[0].cpu().squeeze().numpy()
        label = labels[0].cpu().squeeze().numpy()
        pred = outputs[0].cpu().squeeze().numpy()
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(img, cmap="gray")
        plt.title("Input Image")
        plt.subplot(1,3,2)
        plt.imshow(label, cmap="gray")
        plt.title("GT Seg.") 
        plt.subplot(1,3,3)
        plt.imshow(pred, cmap="gray")
        plt.title("Pred Seg.")
        plt.show()    

        print(f"Test Metrics: {test_metrics}")

    return test_metrics


def test_fn_with_tta(model, test_loader, cfg):
    test_metrics = {"dice":0.0,
                    "iou":0.0}
    # criterion = cfg.loss



    model.eval()
    with torch.no_grad():
        # confusion_matrix_overall = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.int64)
        for j, (images, labels) in enumerate(test_loader):
            images, labels = images.to(cfg.device).float(), labels.to(cfg.device)
            augmented_imgs = [images, TF.hflip(images), TF.rotate(images, 10)]

            combined_logits = torch.stack([model(img_aug) for img_aug in augmented_imgs])
            # outputs_logits = model(images)
            avg_logits = torch.mean(combined_logits, dim=0)
            outputs = torch.argmax(avg_logits, dim=1)

            all_metrics = compute_metrics(outputs, labels, num_classes=cfg.num_classes)

            for met in test_metrics:
                test_metrics[met] += all_metrics[met]

        #     conf_mat = compute_confusion_matrix(avg_logits, labels, cfg.num_classes)
        #     confusion_matrix_overall += conf_mat

        # print_confusion_matrix(confusion_matrix_overall)
        test_metrics = {key: test_metrics[key] / len(test_loader) for key in test_metrics}

        print(f"Test Metrics: {test_metrics}")

    return test_metrics