
import torch

import tempfile
import torch
import shap

# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt

from xai_utils.gradcam_yolo import YOLOGradCAM
from xai_utils.shap_yolo import YOLOSHAPExplainer


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Brain Tumor Detection with Explainability", layout="centered")
st.title("🧠 Brain Tumor Detection with Grad-CAM Explainability")
st.markdown("Upload an MRI image to detect the **tumor type** and visualise the activation region.")


# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"
    yolo_model = YOLO(model_path)
    gradcam = YOLOGradCAM(model_path, input_size=512)
    return yolo_model, gradcam


yolo_model, gradcam = load_model()


# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)  # RGB

    st.image(img_np, caption="Uploaded MRI", use_container_width=True)

    # -------------------------------
    # Step 1: YOLO Prediction
    # -------------------------------
    st.write("🔍 Detecting tumour type...")
    results = yolo_model.predict(img_np, conf=0.25, verbose=False)
    res = results[0]

    if res.boxes is None or len(res.boxes) == 0:
        st.warning("⚠️ No tumour detected. Try another image.")
        st.stop()

    # Choose BEST detection (highest conf)
    confs = [float(b.conf) for b in res.boxes]
    best_i = int(np.argmax(confs))
    best_box = res.boxes[best_i]

    cls_id = int(best_box.cls)
    confidence = float(best_box.conf)
    tumor_type = yolo_model.names[cls_id]

    st.success(f"✅ **Detected Tumour Type:** {tumor_type.capitalize()} ({confidence:.2f} confidence)")

    # Show YOLO annotated image
    annotated_bgr = res.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, caption="YOLO Detection (bounding box)", use_container_width=True)

    # -------------------------------
    # Step 2: ROI-based (box-specific) Grad-CAM
    # -------------------------------
    st.write("🧩 Generating Grad-CAM explainability map (focused on detected box)...")

    alpha = st.slider(
        "Adjust Explainability Intensity (Grad-CAM opacity)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )

    margin_pct = st.slider(
        "ROI margin around box (%)",
        min_value=0, max_value=40, value=15, step=5
    ) / 100.0

    thr = st.slider(
        "CAM cleaning threshold (higher = tighter focus)",
        min_value=0.30, max_value=0.80, value=0.55, step=0.05
    )

    # Get xyxy and crop ROI (+ margin)
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    h, w = img_np.shape[:2]

    mx = int(margin_pct * (x2 - x1))
    my = int(margin_pct * (y2 - y1))

    x1m = max(0, x1 - mx); y1m = max(0, y1 - my)
    x2m = min(w, x2 + mx); y2m = min(h, y2 + my)

    roi = img_np[y1m:y2m, x1m:x2m]
    roi_pil = Image.fromarray(roi)

    # Save ROI to temp file for Grad-CAM
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        roi_pil.save(tmp.name)
        roi_path = tmp.name

    # Generate raw CAM on ROI (this is what makes it box-focused)
    cam01 = gradcam.generate_cam(roi_path, clean=True, thr=thr)

    # Create overlay on ROI resized inside GradCAM class, then resize back to ROI
    overlay_roi_resized = gradcam.overlay(roi_path, cam01, alpha=alpha)  # RGB uint8 @512x512
    overlay_roi = cv2.resize(overlay_roi_resized, (roi.shape[1], roi.shape[0]))

    # Paste overlay back into original image
    final = img_np.copy()
    final[y1m:y2m, x1m:x2m] = overlay_roi

    st.image(final, caption=f"Grad-CAM (focused ROI) — Opacity {alpha:.2f}", use_container_width=True)

    # -------------------------------
    # Step 3 (Optional): SHAP Explanation
    # -------------------------------
    if st.checkbox("🔍 Show SHAP Explanation"):
        st.write("📊 Generating SHAP explanation...")

        background_file = st.file_uploader(
            "📎 Upload a background (healthy) MRI image for SHAP",
            type=["jpg", "jpeg", "png"]
        )

        if background_file is not None:
            # Save main and background images as temp files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as bg_tmp:

                image.save(img_tmp.name)
                Image.open(background_file).convert("RGB").save(bg_tmp.name)

                img_path = img_tmp.name
                bg_path = bg_tmp.name

            with st.spinner("Explaining with SHAP..."):
                explainer = YOLOSHAPExplainer("best.pt")
                image_np_out, shap_mask = explainer.explain(
                    image_path=img_path,
                    background_path=bg_path
                )

            # Convert to numpy
            if not isinstance(image_np_out, np.ndarray):
                image_np_out = np.array(image_np_out)
            if not isinstance(shap_mask, np.ndarray):
                shap_mask = np.array(shap_mask)

            # Ensure 3 channels
            if shap_mask.ndim == 2:
                shap_mask = np.repeat(shap_mask[:, :, np.newaxis], 3, axis=2)

            # Normalise to [0,1]
            if shap_mask.max() > 1:
                shap_mask = shap_mask / 255.0
            if image_np_out.max() > 1:
                image_np_out = image_np_out / 255.0

            import matplotlib.cm as cm

            shap_gray = shap_mask.mean(axis=2)
            shap_norm = (shap_gray - shap_gray.min()) / (shap_gray.max() - shap_gray.min() + 1e-8)
            shap_norm = np.power(shap_norm, 0.5)  # contrast

            shap_colormap = cm.jet(shap_norm)[:, :, :3]

            if image_np_out.shape[:2] != shap_colormap.shape[:2]:
                shap_colormap = cv2.resize(shap_colormap, (image_np_out.shape[1], image_np_out.shape[0]))

            overlay = 0.5 * image_np_out + 0.5 * shap_colormap
            overlay = np.clip(overlay, 0, 1)

            fig, ax = plt.subplots()
            ax.imshow(overlay)
            ax.axis("off")
            st.pyplot(fig)
