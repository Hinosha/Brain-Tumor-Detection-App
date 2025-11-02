
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOGradCAM:
    """
    Simple Grad-CAM visualiser for YOLOv8/v11 models.
    Produces a class-agnostic heatmap overlay highlighting
    the most activated spatial regions in the feature maps.
    """

    def __init__(self, model_path):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.model.eval()

        # Select a convolutional layer near the end of the backbone
        # (adjust index depending on model depth)
        self.target_layer = self.model.model.model[8]

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_path):
        """Generate Grad-CAM heatmap overlay for a given image."""
        # --- 1️⃣ Load & preprocess image ---
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0          # HWC, 0-1
        img_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)                                         # BCHW
            .to(next(self.model.model.parameters()).device)
        )
        img_tensor.requires_grad = True

        # --- 2️⃣ Forward pass ---
        outputs = self.model.model(img_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        # --- 3️⃣ Select a representative score to backprop ---
        if outputs.ndim == 2 and outputs.size(1) > 4:
            # Typical YOLO detection tensor
            scores = outputs[:, 4]           # objectness
        else:
            # Fallback: mean activation
            scores = outputs.mean(dim=1)
        if scores.numel() == 0:
            raise ValueError("No detections found.")
        score = scores.max()
        score.backward()

        # --- 4️⃣ Compute weighted activations ---
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        act = self.activations.clone()
        for i in range(act.shape[1]):
            act[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(act, dim=1).squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # --- 5️⃣ Create coloured overlay ---
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, 0.5, (img_np * 255).astype(np.uint8), 0.5, 0)

        return Image.fromarray(overlay.astype(np.uint8))
