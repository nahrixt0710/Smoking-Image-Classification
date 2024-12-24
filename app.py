import streamlit as st
import joblib
import pickle
import cv2
import numpy as np
from src.utils import *
import json

# config
st.set_page_config(page_title="Streamlit App", page_icon=":shark:")
st.write("Upload an image and choose a model to make a prediction.")

# choose model
model_type = st.sidebar.selectbox("Choose Model", ["SVM", "KNN", "RF"])
# model_type = "SVM"

st.title(f"Image Prediction using {model_type}")

if model_type == "SVM":
    st.write("Support Vector Machine (SVM) is a powerful supervised learning model.")
elif model_type == "KNN":
    st.write(
        "K-Nearest Neighbors (KNN) predicts based on the closest training examples."
    )
elif model_type == "RF":
    st.write("Random Forest is a powerful supervised learning model.")

# upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # read and display image using OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        use_container_width=True,
    )

    # predict with selected model
    if model_type == "SVM":
        with open(r"./src/checkpoint/svm_params_latest.json", "r") as f:
            params = json.load(f)

        img_size = params["img_size"]
        bin_size = params["bin"]

        model = joblib.load(r"./src/checkpoint/svm_best_model_latest.pkl")

        img = preprocess_hsv(img, img_size=img_size, bin=bin_size)
        pred = model.predict([img])[0]

    elif model_type == "KNN":
        with open(r"./src/checkpoint/knn_params_latest.json", "r") as f:
            params = json.load(f)

        img_size = params["img_size"]
        bin_size = params["bin"]

        model = joblib.load(r"./src/checkpoint/knn_best_model_latest.pkl")

        img = preprocess_hsv(img, img_size=img_size, bin=bin_size)
        pred = model.predict([img])[0]

    elif model_type == "RF":
        with open(r"./src/checkpoint/rf_params_latest.json", "r") as f:
            params = json.load(f)

        img_size = params["img_size"]
        bin_size = params["bin"]

        model = joblib.load(r"./src/checkpoint/rf_best_model_latest.pkl")

        img = preprocess_hsv(img, img_size=img_size, bin=bin_size)
        pred = model.predict([img])[0]

    if pred == 0:
        prediction = "Not smoking"
    else:
        prediction = "Smoking"

    color = "green" if prediction == "Not smoking" else "red"

    st.markdown(
        f"### Prediction ({model_type}): **:<span style='color:{color}'>{prediction}</span>**",
        unsafe_allow_html=True,
    )
