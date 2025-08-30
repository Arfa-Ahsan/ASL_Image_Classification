# ASL Image Classification App

## Problem Statement

Recognizing American Sign Language (ASL) hand signs from images is a challenging computer vision task. This project aims to build an interactive web application that can classify ASL alphabet signs (A-Z, space, nothing, del) from uploaded or sample images.

## Solution

- A custom Convolutional Neural Network (CNN) was trained on the ASL Alphabet dataset.
- The model achieves **~99% accuracy** on the validation set.
- The app is built with **Streamlit** and allows users to:
  - Upload their own ASL hand sign images.
  - Select sample images from the sidebar.
  - View the predicted letter and a probability bar graph for all classes.

## Model

- The model is a custom CNN (see `Notebook/ASL_Classification.ipynb` for architecture and training).
- Trained on 64x64 RGB images.
- Achieves high accuracy (~99%) on validation data.

## How to Use

1. **Clone or download this repository.**
2. **Install dependencies:**
   ```
   pip install streamlit tensorflow pillow matplotlib numpy
   ```
3. **Add your trained model:**
   - Place your Keras model file (`my_model.keras`) in the `Model` folder.
4. **Add sample images:**
   - Place sample images named `A.jpg`, `B.jpg`, ..., `space.jpg`, `nothing.jpg`, `del.jpg` in the `Images` folder.
5. **Run the app:**
   ```
   streamlit run app.py
   ```
6. **Open the app in your browser** and either upload an image or select a sample from the sidebar.

## Project Structure

```
ASL_Classification/
│
├── app.py
├── Model/
│   └── my_model.keras
├── Images/
│   ├── A.jpg
│   ├── B.jpg
│   └── ... (other sample images)
├── README.md
├── .gitignore
└── Notebook/
    └── ASL_Classification.ipynb
```

