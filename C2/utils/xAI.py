import torch
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

def gradcam(model, test_loader, classes_names, cfg):

    model.eval()
    cam_extractor = GradCAM(model=model,
                            target_layer="layer1")
    
    # images, labels = test_loader
    for j, (images, labels) in enumerate(test_loader):
        images, labels = images.to(cfg.device).float(), labels.to(cfg.device)
        image = images[0].unsqueeze(0)
        outputs_logits = model(image)
        
        output_class = outputs_logits.squeeze().argmax().item()
        print(output_class)

        # activation_map = cam_extractor(class_idx=output_class, scores=outputs_logits)[0]
        # activation_map = F.interpolate(activation_map[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze()

        activation_map = cam_extractor(class_idx=output_class, scores=outputs_logits)[0]

        # If activation_map has shape [1, H, W], squeeze to [H, W]
        if activation_map.dim() == 3:
            activation_map = activation_map.squeeze(0)

        # Now interpolate: [1, 1, H, W] -> upsample -> [H_out, W_out]
        activation_map = F.interpolate(
            activation_map.unsqueeze(0).unsqueeze(0),  # shape: [1, 1, H, W]
            size=image.shape[-2:],                    # target size: (H, W)
            mode="bilinear",
            align_corners=True
        ).squeeze()  # shape: [H, W]

        image_pil = to_pil_image(image.squeeze().cpu())
        act_map_pil = to_pil_image(activation_map.cpu())
        # activation_map = activation_map.cpu()
        # activation_map -= activation_map.min()
        # activation_map /= activation_map.max()

        # Convert to numpy and apply colormap (e.g., 'jet')
        # import matplotlib.cm as cm
        # activation_map_colored = cm.jet(activation_map.numpy())[..., :3]  # Drop alpha channel

        # Convert to PIL image
        # act_map_pil = to_pil_image((activation_map_colored * 255).astype("uint8"))

        result = overlay_mask(image_pil, act_map_pil, alpha=0.5)
        plt.imshow(result)
        plt.axis("off")
        plt.title(f"Activation Map (GradCAM) for class {classes_names[output_class]}")
        plt.show()
        break