"""
Brain Tumor Detection + Explainability (Streamlit App)
-----------------------------------------------------

This program builds an interactive Streamlit web app that:

1) Loads a trained YOLO model (best.pt) to detect brain tumour type(s) in an MRI image.
2) Displays the YOLO bounding-box prediction and confidence.
3) Generates a Grad-CAM style explainability heatmap (via a custom YOLOGradCAM wrapper)
   and overlays it on the original image with a user-controlled opacity slider.
4) Optionally generates a SHAP-based explanation (via a custom YOLOSHAPExplainer) using
   a user-uploaded background (healthy) MRI image and displays a blended SHAP overlay.

Notes:
- YOLO is used for object detection (localising tumours with bounding boxes + tumour class).
- Grad-CAM provides post-hoc visual explanation by highlighting areas that most influenced
  the detection/classification.
- SHAP provides an alternative explanation approach and requires a background reference image.
"""

# -------------------------------
# Imports (some are duplicated in your original; kept minimal here)
# -------------------------------
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt

# Custom explainability utilities (your own modules)
from xai_utils.gradcam_yolo import YOLOGradCAM
from xai_utils.shap_yolo import YOLOSHAPExplainer


# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection with Explainability",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection with Grad-CAM Explainability")
st.markdown(
    "Upload an MRI image to detect the **tumor type** and visualise the activation region."
)


# -------------------------------
# Load YOLO + Grad-CAM model objects (cached)
# -------------------------------
@st.cache_resource
def load_model():
    """
    Loads the YOLO detection model and the Grad-CAM wrapper once,
    then caches them so Streamlit doesn't reload on every UI interaction.
    """
    model_path = "best.pt"
    yolo_model = YOLO(model_path)
    gradcam = YOLOGradCAM(model_path)
    return yolo_model, gradcam


yolo_model, gradcam = load_model()


# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an MRI image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

# Only run the pipeline if a file is provided
if uploaded_file:

    # -------------------------------
    # Display uploaded image
    # -------------------------------
    st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)

    # Convert uploaded file to PIL Image and NumPy array
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)  # shape: (H, W, 3), RGB

    # -------------------------------
    # Step 1: YOLO Prediction
    # -------------------------------
    st.write("🔍 Detecting tumor type...")

    # Run YOLO prediction on the numpy image
    # conf=0.25 means detections below 0.25 confidence are filtered out.
    results = yolo_model.predict(img_np, conf=0.25, verbose=False)

    # YOLO returns a list; we take the first result (single image)
    res = results[0]

    # Extract detection info (label, confidence)
    detections = []
    for box in res.boxes:
        cls_id = int(box.cls)                 # class index predicted by YOLO
        conf = float(box.conf)                # confidence score
        label = yolo_model.names[cls_id]      # class name from model
        detections.append((label, conf))

    # If detections exist, show best detection and annotated image
    if detections:
        # In your current code you choose detections[0] (first detection).
        # This is not always the "best" detection unless YOLO sorts them.
        tumor_type, confidence = detections[0]

        st.success(
            f"✅ **Detected Tumor Type:** {tumor_type.capitalize()} "
            f"({confidence:.2f} confidence)"
        )

        # Create an annotated image with bounding boxes
        annotated_img = res.plot()  # returns BGR image (OpenCV style)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(
            annotated_img_rgb,
            caption=f"YOLO Detection — {tumor_type.capitalize()} ({confidence:.2f})",
            use_container_width=True
        )

    else:
        # Stop app flow if no tumours detected
        st.warning("⚠️ No tumor detected. Try another image.")
        st.stop()

    # -------------------------------
    # Step 2: Grad-CAM Visualisation
    # -------------------------------
    st.write("🧩 Generating Grad-CAM explainability map...")

    # User controls the overlay intensity/opacity of the heatmap
    alpha = st.slider(
        "Adjust Explainability Intensity (Grad-CAM opacity)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # Many Grad-CAM implementations expect a file path.
    # So we save the uploaded MRI to a temporary file on disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_image_path = tmp.name

    # Generate Grad-CAM output image using your custom wrapper
    gradcam_img = gradcam.generate(temp_image_path)

    # Convert both Grad-CAM and original image to arrays for blending
    gradcam_np = np.array(gradcam_img).astype(np.float32)
    original_np = np.array(image).astype(np.float32)

    # Ensure Grad-CAM output matches original size
    gradcam_np = cv2.resize(
        gradcam_np,
        (original_np.shape[1], original_np.shape[0])
    )

    # Blend Grad-CAM overlay with original image
    # alpha controls contribution of Grad-CAM vs original
    blended = cv2.addWeighted(
        gradcam_np / 255.0, alpha,
        original_np / 255.0, 1 - alpha,
        0
    )
    blended = np.uint8(blended * 255)

    # Show Grad-CAM blended image
    st.image(
        blended,
        caption=f"Grad-CAM (Opacity: {alpha:.2f})",
        use_container_width=True
    )

    # -------------------------------
    # Step 3 (Optional): SHAP Explanation
    # -------------------------------
    if st.checkbox("🔍 Show SHAP Explanation"):
        st.write("📊 Generating SHAP explanation...")

        # SHAP typically needs a background/reference sample.
        # Here you ask the user to upload a "healthy MRI" as background.
        background_file = st.file_uploader(
            "📎 Upload a background (healthy) MRI image for SHAP",
            type=["jpg", "jpeg", "png"]
        )

        if background_file is not None:
            # Re-load the main uploaded image (already loaded above, but kept explicit)
            image_main = Image.open(uploaded_file).convert("RGB")

            # Save both main image and background to temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as bg_tmp:

                image_main.save(img_tmp.name)
                Image.open(background_file).convert("RGB").save(bg_tmp.name)

                temp_image_path = img_tmp.name
                bg_image_path = bg_tmp.name

            # Run your custom SHAP explainer
            with st.spinner("Explaining with SHAP..."):
                explainer = YOLOSHAPExplainer("best.pt")
                image_np_out, shap_mask = explainer.explain(
                    image_path=temp_image_path,
                    background_path=bg_image_path
                )

            # Convert outputs to numpy arrays if needed
            if not isinstance(image_np_out, np.ndarray):
                image_np_out = np.array(image_np_out)
            if not isinstance(shap_mask, np.ndarray):
                shap_mask = np.array(shap_mask)

            # If SHAP mask is grayscale, expand to 3 channels for overlay
            if shap_mask.ndim == 2:
                shap_mask = np.repeat(shap_mask[:, :, np.newaxis], 3, axis=2)

            # Normalise ranges to [0,1]
            if shap_mask.max() > 1:
                shap_mask = shap_mask / 255.0
            if image_np_out.max() > 1:
                image_np_out = image_np_out / 255.0

            # Create a coloured heatmap from SHAP mask for visualisation
            import matplotlib.cm as cm

            # Convert to grayscale importance by averaging channels
            shap_gray = shap_mask.mean(axis=2)

            # Normalise to [0,1] safely
            shap_norm = (shap_gray - shap_gray.min()) / (shap_gray.max() - shap_gray.min() + 1e-8)

            # Increase contrast (gamma correction)
            shap_norm = np.power(shap_norm, 0.5)

            # Apply colour map (jet)
            shap_colormap = cm.jet(shap_norm)[:, :, :3]  # remove alpha channel

            # Resize heatmap if size mismatch
            if image_np_out.shape[:2] != shap_colormap.shape[:2]:
                shap_colormap = cv2.resize(
                    shap_colormap,
                    (image_np_out.shape[1], image_np_out.shape[0])
                )

            # Blend SHAP heatmap with original image
            overlay = 0.5 * image_np_out + 0.5 * shap_colormap
            overlay = np.clip(overlay, 0, 1)

            # Display using matplotlib inside Streamlit
            fig, ax = plt.subplots()
            ax.imshow(overlay)
            ax.axis("off")
            st.pyplot(fig)
