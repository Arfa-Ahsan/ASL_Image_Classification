import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load model
MODEL_PATH = os.path.join('Model', 'my_model.keras')
model = load_model(MODEL_PATH)

# Class names 
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

st.set_page_config(page_title="ASL Image Classification App", layout="centered")

# Centered title using markdown and HTML
st.markdown(
    "<h1 style='text-align: center; color: #ffff;'>ASL IMAGE CLASSIFICATION APP</h1>",
    unsafe_allow_html=True
)
st.write("Upload an image of an ASL hand sign or select a sample image from the sidebar. The app will predict the letter.")

# --- Sample images selection in sidebar ---
sample_dir = os.path.join('Images')  # Place your sample images here, named as A.jpg, B.jpg, ..., space.jpg
sample_options = [c for c in class_names if c != 'del']
sample_choice = st.sidebar.selectbox("Select a sample image:", ["None"] + sample_options)

sample_image = None
if sample_choice != "None":
    sample_path = os.path.join(sample_dir, f"{sample_choice}.jpg")
    if os.path.exists(sample_path):
        sample_image = Image.open(sample_path).convert('RGB')
    else:
        st.sidebar.warning(f"Sample image for {sample_choice} not found.")

uploaded_file = st.file_uploader("Or upload your own ASL image...", type=["jpg", "jpeg", "png"])

# --- Use uploaded image or sample image ---
image = None
caption = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    caption = 'Uploaded Image'
elif sample_image is not None:
    image = sample_image
    caption = f'Sample Image: {sample_choice}'

if image is not None:
    st.image(image, caption=caption, use_container_width=True)

    # Preprocess image for model (resize to 64x64 for prediction, but display original)
    img_for_model = image.resize((64, 64))
    img_array = np.array(img_for_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]

    st.markdown(f"<h3 style='text-align: center;'>Predicted Letter: <span style='color: #0066cc'>{predicted_label}</span></h3>", unsafe_allow_html=True)

    # --- Show prediction probabilities as a bar graph ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(class_names, prediction[0], color='skyblue')
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)
