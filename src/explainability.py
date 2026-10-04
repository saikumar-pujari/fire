import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) Visual XAI Engine.
    Computes class activation heatmaps overlaying input images.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.target_layer.register_backward_hook(backward_hook)

    def _disable_inplace(self):
        for module in self.model.modules():
            if hasattr(module, "inplace"):
                module.inplace = False

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap for a given input tensor and target class.
        """
        self.model.eval()
        self._disable_inplace()
        input_tensor = input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients[0] # [C, H, W]
        activations = self.activations[0] # [C, H, W]

        # Global Average Pooling of Gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # [C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0) # [H, W]
        cam = F.relu(cam) # Apply ReLU to keep positive influence only

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # Normalize 0-1
        return cam, target_class


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ Visual XAI Engine.
    Provides superior localization accuracy and handles multi-instance objects (flames, smoke plumes).
    """
    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        self._disable_inplace()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients[0] # [C, H, W]
        activations = self.activations[0] # [C, H, W]

        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        # Alpha weights for Grad-CAM++
        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = grad_2 / alpha_denom

        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


def find_target_layer(model):
    """
    Automatically locates the last convolutional layer of a given model architecture.
    """
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    return target_layer


def overlay_heatmap_on_image(image_np, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays heatmaps onto original RGB image array.
    """
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
        
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def generate_xai_visualizations(model, sample_tensor, sample_img_np, class_names, model_name="resnet18", output_dir="outputs/xai"):
    """
    Generates side-by-side visual explainability report comparing Grad-CAM and Grad-CAM++.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_layer = find_target_layer(model)
    
    if target_layer is None:
        print(f"[XAI] Target convolutional layer not found for model '{model_name}'. Skipping Grad-CAM.")
        return

    gcam = GradCAM(model, target_layer)
    gcam_pp = GradCAMPlusPlus(model, target_layer)

    heatmap_cam, target_cls = gcam.generate_heatmap(sample_tensor)
    heatmap_pp, _ = gcam_pp.generate_heatmap(sample_tensor)

    overlay_cam = overlay_heatmap_on_image(sample_img_np, heatmap_cam)
    overlay_pp = overlay_heatmap_on_image(sample_img_np, heatmap_pp)

    cls_label = class_names[target_cls] if target_cls < len(class_names) else f"Class_{target_cls}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sample_img_np)
    axes[0].set_title(f"Original Input\n(Predicted: {cls_label})")
    axes[0].axis("off")

    axes[1].imshow(overlay_cam)
    axes[1].set_title(f"Grad-CAM Heatmap\n[{model_name}]")
    axes[1].axis("off")

    axes[2].imshow(overlay_pp)
    axes[2].set_title(f"Grad-CAM++ Heatmap\n[{model_name}]")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"xai_{model_name}_gradcam.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[XAI Engine] Saved Grad-CAM visual heatmap for '{model_name}' to: {save_path}")
