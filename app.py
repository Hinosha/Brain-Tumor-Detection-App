import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from xai_utils.gradcam_yolo import YOLOGradCAM

import tempfile
import torch
import shap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from xai_utils.gradcam_yolo import YOLOGradCAM
from xai_utils.shap_yolo import YOLOSHAPExplainer


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
    st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)

    # Convert to PIL + NumPy
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

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
        annotated_img = res.plot()  # draws bounding boxes
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Display YOLO output with bounding box
        st.image(annotated_img_rgb,
                 caption=f"YOLO Detection — {tumor_type.capitalize()} ({confidence:.2f})",
                 use_container_width=True)
    else:
        st.warning("⚠️ No tumor detected. Try another image.")
        st.stop()

    # -------------------------------
    # Step 2: Grad-CAM Visualization
    # -------------------------------
    # -------------------------------
    # Step 2: Grad-CAM Visualization
    # -------------------------------
    st.write("🧩 Generating Grad-CAM explainability map...")

    # Add a slider to control transparency
    alpha = st.slider("Adjust Explainability Intensity (Grad-CAM opacity)",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)           # Save PIL image to a real file
        temp_image_path = tmp.name
    
    gradcam_img = gradcam.generate(temp_image_path)


    # Generate Grad-CAM
   # gradcam_img = gradcam.generate(uploaded_file)

    # Blend Grad-CAM with original image using alpha
    gradcam_np = np.array(gradcam_img).astype(np.float32)
    original_np = np.array(Image.open(uploaded_file).convert("RGB")).astype(np.float32)

    # Ensure same size
    gradcam_np = cv2.resize(gradcam_np, (original_np.shape[1], original_np.shape[0]))

    blended = cv2.addWeighted(gradcam_np / 255.0, alpha, original_np / 255.0, 1 - alpha, 0)
    blended = np.uint8(blended * 255)

    # Display adjustable Grad-CAM
    st.image(blended, caption=f"Grad-CAM (Opacity: {alpha:.2f})", use_container_width=True)

    from xai_utils.shap_yolo import YOLOSHAPExplainer
    import matplotlib.pyplot as plt
    import shap

    if st.checkbox("🔍 Show SHAP Explanation"):
        st.write("📊 Generating SHAP explanation...")

        background_file = st.file_uploader("📎 Upload a background (healthy) MRI image for SHAP", type=["jpg", "jpeg", "png"])

        if background_file is not None:
            # ✅ Load image again
            image = Image.open(uploaded_file).convert("RGB")
        
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as bg_tmp:
        
                image.save(img_tmp.name)
                Image.open(background_file).convert("RGB").save(bg_tmp.name)
        
                temp_image_path = img_tmp.name
                bg_image_path = bg_tmp.name
        
            with st.spinner("Explaining with SHAP..."):
                explainer = YOLOSHAPExplainer("best.pt")
                image_np, shap_mask = explainer.explain(
                    image_path=temp_image_path,
                    background_path=bg_image_path
                )
        
            # Ensure they're numpy arrays and normalized
            if not isinstance(image_np, np.ndarray):
                image_np = np.array(image_np)
            if not isinstance(shap_mask, np.ndarray):
                shap_mask = np.array(shap_mask)
        
            if shap_mask.ndim == 2:
                shap_mask = np.repeat(shap_mask[:, :, np.newaxis], 3, axis=2)
        
            if shap_mask.max() > 1:
                shap_mask = shap_mask / 255.0
            if image_np.max() > 1:
                image_np = image_np / 255.0
        
            # ✅ PLOT correctly
            import matplotlib.cm as cm

            # Normalize SHAP mask for display
            shap_gray = shap_mask.mean(axis=2)  # Convert to grayscale importance
            shap_norm = (shap_gray - shap_gray.min()) / (shap_gray.max() - shap_gray.min() + 1e-8)
            
            # Apply color map (jet, inferno, plasma, etc.)
            shap_colormap = cm.jet(shap_norm)[:, :, :3]  # Drop alpha channel
            
            # Resize image_np if needed to match shap_colormap
            if image_np.shape[:2] != shap_colormap.shape[:2]:
                shap_colormap = cv2.resize(shap_colormap, (image_np.shape[1], image_np.shape[0]))
            
            # Blend heatmap with original image
            overlay = 0.5 * image_np + 0.5 * shap_colormap
            overlay = np.clip(overlay, 0, 1)
            
            # Show final plot
            fig, ax = plt.subplots()
            ax.imshow(overlay)
            ax.axis("off")
            st.pyplot(fig)
