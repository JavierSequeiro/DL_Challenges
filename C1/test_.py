import torch
# from .utils.config import *
# from .utils.metrics import *
from utils.config import *
from utils.metrics import *
  
def test_fn(model, test_loader, cfg):
    test_metrics = {"accuracy":0.0,
                    "recall":0.0,
                    "precision":0.0,
                    "f1 score": 0.0}
    criterion = cfg.loss
    model.eval()
    with torch.no_grad():
        confusion_matrix_overall = torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.int64)
        for j, (images, labels) in enumerate(test_loader):
            images, labels = images.to(cfg.device).float(), labels.to(cfg.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            outputs = torch.argmax(outputs, dim=1)
            all_metrics = compute_metrics(outputs, labels)

            for met in test_metrics:
                test_metrics[met] += all_metrics[met]

            conf_mat = compute_confusion_matrix(outputs, labels, cfg.num_classes)
            confusion_matrix_overall += conf_mat

        print_confusion_matrix(confusion_matrix_overall)
        test_metrics = {key: test_metrics[key] / len(test_loader) for key in test_metrics}

        print(f"Test Metrics: {test_metrics}")

    return test_metrics