# xai_utils/gradcam_yolo.py
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOGradCAM:
    """
    Grad-CAM for YOLOv8 (feature-map based).
    This implementation focuses on producing a heatmap from a target conv layer.

    IMPORTANT:
    - YOLO is a detector, so a perfect "box-specific" CAM requires mapping the chosen box
      back into the model's internal tensor output.
    - In practice, the most stable approach is ROI-cropping in the app, then Grad-CAM on ROI.

    This class returns:
      - cam01: raw CAM normalised to [0,1]
      - overlay: optional heatmap overlay on the resized input
    """

    def __init__(self, model_path: str, input_size: int = 512):
        self.model = YOLO(model_path)
        self.model.model.eval()

        self.input_size = input_size

        # Target layer: last conv block before detect head (your original -2 choice)
        self.target_layer = self.model.model.model[-2]

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _preprocess(self, image_path: str):
        img = Image.open(image_path).convert("RGB").resize((self.input_size, self.input_size))
        img_np = np.array(img).astype(np.float32) / 255.0  # RGB [0,1]
        img_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)     # HWC -> CHW
            .unsqueeze(0)         # add batch
            .float()
            .to(next(self.model.model.parameters()).device)
        )
        img_tensor.requires_grad_(True)
        return img, img_np, img_tensor

    @staticmethod
    def _postprocess_cam(cam01: np.ndarray, thr: float = 0.55, keep_largest: bool = True):
        """
        Optional cleaning: threshold + keep largest connected component.
        Helps remove scattered highlights.
        """
        cam = cam01.copy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        mask = (cam >= thr).astype(np.uint8)

        if keep_largest:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == largest).astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cam_clean = cam * mask
        cam_clean = (cam_clean - cam_clean.min()) / (cam_clean.max() - cam_clean.min() + 1e-8)
        return cam_clean

    def generate_cam(self, image_path: str, clean: bool = True, thr: float = 0.55):
        """
        Returns raw CAM normalised to [0,1] (single channel), size = input_size x input_size.
        """
        _, _, img_tensor = self._preprocess(image_path)

        # Forward/backward
        with torch.enable_grad():
            outputs = self.model.model(img_tensor)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Fallback scoring: push gradient through something meaningful
            # This is NOT box-specific; ROI-cropping in the app makes it "effectively box-specific".
            if outputs.ndim == 2 and outputs.size(1) > 4:
                scores = outputs[:, 4]  # objectness-like
            else:
                scores = outputs.mean(dim=1)

            if scores.numel() == 0:
                raise ValueError("No detections / no scores found for CAM.")

            score = scores.max()
            self.model.model.zero_grad(set_to_none=True)
            score.backward()

        # Grad-CAM
        pooled_grad = torch.mean(self.gradients, dim=(0, 2, 3))  # [C]
        activations = self.activations[0]                        # [C,H,W]

        weighted = activations * pooled_grad[:, None, None]
        heatmap = torch.mean(weighted, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        heatmap = cv2.resize(heatmap, (self.input_size, self.input_size))
        cam01 = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        if clean:
            cam01 = self._postprocess_cam(cam01, thr=thr, keep_largest=True)

        return cam01

    def overlay(self, image_path: str, cam01: np.ndarray, alpha: float = 0.5, colormap=cv2.COLORMAP_JET):
        """
        Create an overlay image (RGB uint8) from a raw CAM [0,1].
        Overlay is produced on the internally resized image (input_size x input_size).
        """
        img, img_np, _ = self._preprocess(image_path)  # resized internally
        base = (img_np * 255).astype(np.uint8)          # RGB uint8

        heat = (cam01 * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, colormap)        # BGR
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)    # RGB

        out = cv2.addWeighted(heat, alpha, base, 1 - alpha, 0)
        return out
