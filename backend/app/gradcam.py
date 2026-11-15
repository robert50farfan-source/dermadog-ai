import threading

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Protege el estado compartido (hooks, activaciones, gradientes) si dos hilos
        # piden un mapa de calor a la vez (H4)
        self._lock = threading.Lock()

    def _register_hooks(self):
        # Los hooks solo existen mientras dura generate_heatmap: así las predicciones
        # normales no escriben en self.activations y no quedan hooks colgados (H5)
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # register_full_backward_hook reemplaza a register_backward_hook (obsoleto, H5).
        # Firma: (module, grad_input, grad_output); grad_output[0] es el gradiente
        # respecto a la salida de la capa objetivo.
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        return [
            self.target_layer.register_forward_hook(forward_hook),
            self.target_layer.register_full_backward_hook(backward_hook),
        ]

    def generate_heatmap(self, input_image, class_idx=None):
        with self._lock:
            handles = self._register_hooks()
            try:
                self.model.zero_grad(set_to_none=True)

                # Pasada hacia delante (con gradientes)
                with torch.enable_grad():
                    output = self.model(input_image)

                    if class_idx is None:
                        class_idx = torch.argmax(output)

                    # Pasada hacia atrás desde la puntuación de la clase pedida
                    score = output[0, class_idx]
                    score.backward()

                gradients = self.gradients
                activations = self.activations

                # Promedio global de los gradientes: peso de cada canal
                weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

                # Combinación ponderada de las activaciones
                cam = torch.sum(weights * activations, dim=1, keepdim=True)

                # ReLU
                cam = F.relu(cam)

                # Normaliza a [0, 1]
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-7)

                return cam.cpu().numpy()[0, 0]
            finally:
                for handle in handles:
                    handle.remove()
                # Libera memoria y no deja estado de esta petición para la siguiente
                self.gradients = None
                self.activations = None
                self.model.zero_grad(set_to_none=True)

    @staticmethod
    def overlay_heatmap(image_pil, heatmap, alpha=0.5, colormap='jet'):
        # Redimensiona el mapa de calor al tamaño de la imagen
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = heatmap.resize(image_pil.size, resample=Image.BILINEAR)

        # Aplica el mapa de colores
        heatmap_np = np.array(heatmap)
        cmap = plt.get_cmap(colormap)
        colored_heatmap = cmap(heatmap_np / 255.0)  # forma (H, W, 4)

        # Pasa a RGB (descarta el canal alfa del mapa de colores)
        colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
        colored_heatmap_pil = Image.fromarray(colored_heatmap)

        # Mezcla
        result = Image.blend(image_pil.convert("RGB"), colored_heatmap_pil, alpha=alpha)
        return result
