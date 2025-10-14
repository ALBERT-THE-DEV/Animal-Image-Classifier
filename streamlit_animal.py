import streamlit as st
from fastai.vision.all import *
from PIL import Image
import pathlib
import gdown
import os

# ------------------------------
# Download & load model
# ------------------------------
model_path = "animal_classifier_inference.pkl"  # use inference-ready pkl
if not os.path.exists(model_path):
    gdown.download(
        "https://drive.google.com/uc?id=15aWYj_T7vg-xQlJ10C4okUhJKaJWGmmW",
        model_path,
        quiet=False
    )

# Fix Linux → Windows path compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the FastAI learner
learn = load_learner(model_path)

# ------------------------------
# Streamlit page setup
# ------------------------------
st.set_page_config(
    page_title="Animal Image Classifier",
    layout="centered"
)

# Sidebar info
with st.sidebar:
    st.title("About the Project")
    st.markdown("""
Animal Classifier App  
Built with FastAI (ResNet34) and Streamlit.

This model classifies images into one of 10 animal categories:  
Dog, Cat, Horse, Elephant, Butterfly, Chicken, Cow, Sheep, Spider, Squirrel.
""")
    st.markdown("---")
    st.markdown("""
**Dataset:**  
*Animals-10* (Kaggle Dataset)  

**Model:**  
*Transfer learning using ResNet34 pretrained on ImageNet.*
""")
    st.markdown("---")
    st.markdown("💡 Tip: Upload a clear animal image for best predictions")

# ------------------------------
# Main page
# ------------------------------
st.title("Animal Image Classifier")
st.write("Upload an animal image to predict its type")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=["jpg", "jpeg", "png", "bmp", "gif", "webp"]
)

# Prediction logic
if uploaded_file is not None:
    # Load uploaded image with PIL
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    # Convert to FastAI PILImage for prediction
    fastai_img = PILImage.create(img)

    if st.button("Predict Animal", help="Click to predict the animal type", type="primary"):
        with st.spinner("Analyzing the image... 🧠"):
            pred, pred_idx, probs = learn.predict(fastai_img)

        st.success(f"### Prediction: {pred.capitalize()}")
        st.write(f"**Confidence:** {probs[pred_idx]*100:.2f}%")

else:
    st.info("Upload an animal image file to start classification.")
