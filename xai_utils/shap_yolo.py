import torch
import shap
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOSHAPExplainer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.model.eval()

    def _preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]
        img_np = img_np.transpose(2, 0, 1)  # HWC to CHW
        return img_np

    def explain(self, image_path, background_path):
        # Load and preprocess images
        instance = self._preprocess(image_path)
        background = self._preprocess(background_path)

        instance = instance.reshape(1, -1)       # Shape: [1, features]
        background = background.reshape(1, -1)   # Shape: [1, features]

        # Define prediction function
        def yolo_predict(x):
            x = x.reshape(-1, 3, 224, 224)
            x = torch.tensor(x).float().to(self.model.device)
            results = self.model(x, verbose=False)
            preds = []
            for r in results:
                # Return objectness confidence score (max per image)
                if len(r.boxes) > 0:
                    scores = r.boxes.conf.cpu().numpy()
                    preds.append([scores.max()])
                else:
                    preds.append([0.0])
            return np.array(preds)

        # SHAP KernelExplainer
        explainer = shap.KernelExplainer(yolo_predict, background)
        shap_values = explainer.shap_values(instance, nsamples=100)

        # Reshape back to image
        image_np = instance.reshape(224, 224, 3)
        shap_mask = shap_values[0].reshape(224, 224, 3)

        return image_np, shap_mask
