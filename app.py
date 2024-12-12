import streamlit as st
from src.utils import *
import joblib


st.title("Image Prediction using SVM")
st.write("Upload an image to make a prediction.")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Đọc và hiển thị ảnh bằng OpenCV
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

    model = joblib.load("./src/checkpoint/svm_best_model.pkl")
    pred = predict(img, model)

    st.write(f"Prediction: {pred}")