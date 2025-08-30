import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Load model
MODEL_PATH = os.path.join('Model', 'my_model.keras')
model = load_model(MODEL_PATH)

# Class names
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

st.title("🤟 ASL Live Prediction with Webcam")

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.empty()

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("⚠️ Could not access webcam")
        break
    
    # Flip horizontally (mirror effect for webcam)
    frame = cv2.flip(frame, 1)

    # Preprocess for prediction
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (64, 64)) / 255.0
    img_array = np.expand_dims(resized, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_names[predicted_class]

    # Overlay prediction text on frame
    cv2.putText(frame, f"Prediction: {predicted_label}", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert BGR → RGB for Streamlit display
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

cap.release()
