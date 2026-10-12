import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, class_idx=None):
        # Zero grads
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
        
        # Backward pass
        score = output[0, class_idx]
        score.backward()
        
        # Generate heatmap
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.data.cpu().numpy()[0, 0]

    @staticmethod
    def overlay_heatmap(image_pil, heatmap, alpha=0.5, colormap='jet'):
        # Resize heatmap to image size
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = heatmap.resize(image_pil.size, resample=Image.BILINEAR)
        
        # Apply colormap
        heatmap_np = np.array(heatmap)
        cmap = plt.get_cmap(colormap)
        colored_heatmap = cmap(heatmap_np / 255.0)  # shape (H, W, 4)
        
        # Convert to RGB (discard alpha from cmap if present, though typically we use it for blending)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        colored_heatmap_pil = Image.fromarray(colored_heatmap)
        
        # Blend
        result = Image.blend(image_pil.convert("RGB"), colored_heatmap_pil, alpha=alpha)
        return result
