import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from xai_utils.gradcam_yolo import YOLOGradCAM

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Brain Tumor Detection with Explainability", layout="centered")
st.title("🧠 Brain Tumor Detection with Grad-CAM Explainability")
st.markdown("Upload an MRI image to detect the **tumor type** and visualize the activation region.")

# -------------------------------
# Load YOLO + GradCAM models (cached for speed)
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "best.pt"
    yolo_model = YOLO(model_path)
    gradcam = YOLOGradCAM(model_path)
    return yolo_model, gradcam

yolo_model, gradcam = load_model()

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

    # Convert to PIL + NumPy
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # -------------------------------
    # Step 1: YOLO Prediction
    # -------------------------------
    # -------------------------------
# Step 1: YOLO Prediction
# -------------------------------
st.write("🔍 Detecting tumor type...")
results = yolo_model.predict(img_np, conf=0.25, verbose=False)
res = results[0]

# Extract detection info
detections = []
for box in res.boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    label = yolo_model.names[cls_id]
    detections.append((label, conf))

if detections:
    tumor_type, confidence = detections[0]
    st.success(f"✅ **Detected Tumor Type:** {tumor_type.capitalize()} ({confidence:.2f} confidence)")

    # 🖼️ Get YOLO annotated image (like results[0].show())
    annotated_img = res.plot()  # this draws bounding boxes on the image
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Display YOLO output with bounding box
    st.image(annotated_img_rgb, caption=f"YOLO Detection — {tumor_type.capitalize()} ({confidence:.2f})",
             use_column_width=True)
else:
    st.warning("⚠️ No tumor detected. Try another image.")
    st.stop()


    # -------------------------------
    # Step 2: Grad-CAM Visualization
    # -------------------------------
    st.write("🧩 Generating Grad-CAM explainability map...")
    gradcam_img = gradcam.generate(uploaded_file)

    # Convert PIL image to RGB for display
    gradcam_np = np.array(gradcam_img)

    st.image(gradcam_np, caption=f"Grad-CAM Heatmap for {tumor_type.capitalize()}", use_column_width=True)
