import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import torch

st.set_page_config(page_title="WildCat Detector", layout="wide")

st.title("🐆 WildCat Image Detection Dashboard")

# Load model
yolo_model = YOLO("Model/Sandrina Nur Amanda_Laporan 4.pt")
tf_model = tf.keras.models.load_model("Model/best_model_transfer_amanda.h5")

uploaded_file = st.file_uploader("Upload gambar hewan (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar diunggah", use_column_width=True)

    # Convert image to array
    img_array = np.array(image)

    try:
        # Prediksi dengan YOLO
        results = yolo_model(img_array)
        st.subheader("🔍 Deteksi YOLO")
        st.image(results[0].plot(), use_column_width=True)

        # Prediksi dengan TensorFlow
        img_resized = cv2.resize(img_array, (224, 224))
        img_resized = np.expand_dims(img_resized / 255.0, axis=0)
        tf_pred = tf_model.predict(img_resized)
        st.subheader("🧠 Prediksi TensorFlow:")
        st.write(tf_pred)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
