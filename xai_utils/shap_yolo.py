import shap
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


class YOLOSHAPExplainer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.model.eval()

    def _preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((512, 512))
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )
        return img_tensor, img_np

    def explain(self, image_path, background_path):
        image_tensor, image_np = self._preprocess(image_path)
        background_tensor, _ = self._preprocess(background_path)

        background = background_tensor.numpy()

        # SHAP KernelExplainer expects a model that takes numpy and returns numpy
        def wrapped_model(x):
            x_tensor = torch.from_numpy(x).float()
            with torch.no_grad():
                preds = self.model.model(x_tensor)[0]
                scores = preds[:, 4] if preds.shape[1] > 4 else preds.mean(dim=1)
            return scores.detach().numpy()

        explainer = shap.KernelExplainer(wrapped_model, background)
        shap_values = explainer.shap_values(image_tensor.numpy(), nsamples=100)

        return image_np, shap_values[0]
