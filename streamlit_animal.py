import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown
import os
import platform
import pathlib

# ------------------------------
# Model download and setup
# ------------------------------

MODEL_PATH = "animal_classifier.pkl"
GOOGLE_DRIVE_ID = "15aWYj_T7vg-xQlJ10C4okUhJKaJWGmmW"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}",
        MODEL_PATH,
        quiet=False
    )

# Cross-platform path fix (only for Windows)
if platform.system() == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# Load the FastAI learner
if os.path.exists(MODEL_PATH):
    learn = load_learner(MODEL_PATH)
else:
    st.error("Failed to load model. Please check the Google Drive link.")

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Animal Image Classifier", layout="centered")

# Sidebar info
with st.sidebar:
    st.title("About the Project")
    st.markdown("""
**Animal Classifier App**  
Built with FastAI (ResNet34) and Streamlit.

This model classifies images into one of 10 animal categories:  
Dog, Cat, Horse, Elephant, Butterfly, Chicken, Cow, Sheep, Spider, Squirrel.
""")
    st.markdown("---")
    st.markdown("""
**Dataset:**  
Animals-10 (Kaggle Dataset)  
**Model:** Transfer learning using ResNet34 pretrained on ImageNet
""")
    st.markdown("---")
    st.markdown("💡 Tip: Upload a clear animal image for best predictions")

# Main page
st.title("Animal Image Classifier")
st.write("Upload an animal image to predict its type")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=["jpg", "jpeg", "png", "bmp", "gif", "webp"]
)

# Prediction
if uploaded_file is not None:
    # Convert uploaded file to PIL image
    img = Image.open(uploaded_file)
    img = PILImage.create(img)  # Convert to FastAI image

    st.image(img.to_thumb(512, 512), caption="📸 Uploaded Image", use_container_width=True)

    if st.button("Predict Animal", help="Click to predict the animal type", type="primary"):
        with st.spinner("Analyzing the image... 🧠"):
            pred, pred_idx, probs = learn.predict(img)

        st.success(f"### Prediction: {pred.capitalize()}")
        st.write(f"**Confidence:** {probs[pred_idx]*100:.2f}%")
else:
    st.info("Upload an animal image file to start classification.")
