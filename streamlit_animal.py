import streamlit as st
from fastai.vision.all import *
import pathlib

# Fixes Linux→Windows model compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn = load_learner('animal_classifier.pkl')

st.set_page_config(
    page_title="Animal Image Classifier",
    layout="centered"
)

with st.sidebar:
    st.title("About the Project")
    st.markdown(
        """
        Animal Classifier App  
        Built with FastAI (ResNet34) and Streamlit.
        
        This model classifies images into one of 10 animal categories:
         Dog, Cat, Horse, Elephant, Butterfly, Chicken, Cow, Sheep, Spider, Squirrel.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        **Dataset:**  
        *Animals-10* (Kaggle Dataset)
        
        **Model:**  
        *Transfer learning using ResNet34 pretrained on ImageNet.*

        """
    )

    st.markdown("---")
    st.markdown("Tip: Upload a clear animal image for better predictions")

#  Main Page 
st.title("Animal Image Classifier")
st.write("Upload an animal image to predict its type")

st.markdown("---")

# Uploading image file
uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=["jpg", "jpeg", "png", "bmp", "gif", "webp"]
)

# Prediction Logic
if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img.to_thumb(512, 512), caption="📸 Uploaded Image", use_container_width=True)

    if st.button("Predict Animal",help="Click to predict the animal type", type="primary"):
        with st.spinner("Analyzing the image... "):
            pred, pred_idx, probs = learn.predict(img)

        st.success(f"###Prediction: {pred.capitalize()}")
        st.write(f"**Confidence:** {probs[pred_idx]*100:.2f}%")

else:
    st.info("Upload an animal image file to start classification.")



