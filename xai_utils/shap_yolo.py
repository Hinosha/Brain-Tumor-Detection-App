import shap
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO

class YOLOSHAPExplainer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = YOLO(model_path)
        self.model.model.eval().to(self.device)
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
        ])

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    def prediction_function(self, inputs):
        """
        SHAP expects a function that takes in a batch of tensors and returns a batch of outputs (Nx1).
        For YOLO, we extract the max objectness score for each input.
        """
        outputs = []
        for i in range(inputs.shape[0]):
            x = inputs[i:i+1]
            with torch.no_grad():
                preds = self.model.model(x)[0]
            if preds.ndim == 2 and preds.shape[1] > 4:
                objectness = preds[:, 4]
                score = objectness.max().item() if objectness.numel() > 0 else 0.0
            else:
                score = 0.0
            outputs.append([score])
        return torch.tensor(outputs, dtype=torch.float32)

    def explain(self, image_path, background_path):
        """
        image_path: image you want to explain
        background_path: background image(s) for SHAP baseline (can also pass a list)
        """
        image = self.preprocess(image_path)
        background = self.preprocess(background_path)

        explainer = shap.DeepExplainer(self.prediction_function, background)
        shap_values = explainer.shap_values(image)

        # Convert image back to numpy format for visualization
        image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        shap_numpy = shap_values[0].squeeze().permute(1, 2, 0).cpu().numpy()

        return image_np, shap_numpy

