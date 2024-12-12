import streamlit as st
import joblib
import pickle
import cv2
import numpy as np
from src.utils import *

# config
st.set_page_config(page_title="Streamlit App", page_icon=":shark:")
st.write("Upload an image and choose a model to make a prediction.")

# choose model
model_type = st.sidebar.selectbox("Choose Model", ["SVM", "KNN"])

st.title(f"Image Prediction using {model_type}")

if model_type == "SVM":
    st.write("Support Vector Machine (SVM) is a powerful supervised learning model.")
else:
    st.write(
        "K-Nearest Neighbors (KNN) predicts based on the closest training examples."
    )
# # KNN
# if model_type == "KNN":
#     bin_size = st.sidebar.slider("Bin Size", 1, 64, 16)  # Giá trị mặc định là 16
#     image_size = st.sidebar.slider(
#         "Image Size", 32, 256, 128
#     )  # Giá trị mặc định là 128

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
        model = joblib.load(r"./src/checkpoint/svm_best_model.pkl")
        pred = predict(img, model)
    elif model_type == "KNN":
        try:
            model = joblib.load(r"./src/checkpoint/knn_custom_best.pkl")
        except:
            model = pickle.load(open(r"./src/checkpoint/knn_best_model.pkl", "rb"))

        pred = knn_predict(img, model)

    # Hiển thị kết quả dự đoán
    color = "green" if pred == "Not smoking" else "red"
    st.markdown(
        f"### Prediction ({model_type}): **:<span style='color:{color}'>{pred}</span>**",
        unsafe_allow_html=True,
    )
