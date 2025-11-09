import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOGradCAM:
    """
    Grad-CAM for YOLOv8 — works on final convolutional layer.
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.model.eval()

        # Automatically grab last conv layer before detection head
        self.target_layer = self.model.model.model[-2]  # more reliable than fixed index
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_path):
        # --- 1️⃣ Load & resize image ---
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = (
            torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
            .to(next(self.model.model.parameters()).device)
        )
        img_tensor.requires_grad = True
    
        # ✅ Correct warm-up using image path, not tensor
        _ = self.model.predict(image_path, verbose=False)
        # --- 3️⃣ Forward pass to get raw outputs ---
        with torch.enable_grad():
            outputs = self.model.model(img_tensor)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if outputs.ndim == 2 and outputs.size(1) > 4:
                scores = outputs[:, 4]
            else:
                scores = outputs.mean(dim=1)

            if scores.numel() == 0:
                raise ValueError("No detections.")
            score = scores.max()
            score.backward()

        # --- 4️⃣ Grad-CAM computation ---
        pooled_grad = torch.mean(self.gradients, dim=(0, 2, 3))
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_grad[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (512, 512))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # --- 5️⃣ Overlay ---
        overlay = cv2.addWeighted(heatmap, 0.5, (img_np * 255).astype(np.uint8), 0.5, 0)
        return Image.fromarray(overlay)
