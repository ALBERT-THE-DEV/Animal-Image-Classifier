import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown
import os

# ------------------------------
# 1️⃣ Download & Load Model
# ------------------------------
MODEL_PATH = "animal_classifier_inference.pkl"
GOOGLE_DRIVE_ID = "15aWYj_T7vg-xQlJ10C4okUhJKaJWGmmW"  # replace with your file ID

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(
        f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}",
        MODEL_PATH,
        quiet=False
    )

# Load the FastAI learner
learn = load_learner(MODEL_PATH)

# ------------------------------
# 2️⃣ Streamlit Page Setup
# ------------------------------
st.set_page_config(
    page_title="Animal Image Classifier",
    page_icon="🐾",
    layout="centered"
)

# Sidebar info
with st.sidebar:
    st.title("About the Project")
    st.markdown("""
Animal Classifier App  
Built with FastAI (ResNet34) and Streamlit.

Classifies images into one of 10 animal categories:  
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
# 3️⃣ Main Page
# ------------------------------
st.title("🐾 Animal Image Classifier")
st.write("Upload an animal image to predict its type")
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=["jpg", "jpeg", "png", "bmp", "gif", "webp"]
)

# ------------------------------
# 4️⃣ Prediction Logic
# ------------------------------
if uploaded_file is not None:
    # Load image with PIL
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    # Convert to FastAI image
    fastai_img = PILImage.create(img)

    if st.button("Predict Animal", help="Click to predict the animal type", type="primary"):
        with st.spinner("Analyzing the image... 🧠"):
            pred, pred_idx, probs = learn.predict(fastai_img)

        st.success(f"### Prediction: {pred.capitalize()}")
        st.write(f"**Confidence:** {probs[pred_idx]*100:.2f}%")

else:
    st.info("Upload an animal image file to start classification.")

