import streamlit as st
import joblib
import cv2
import numpy as np
import os
import zipfile
from src.utils import *
import json

# Config
st.set_page_config(page_title="Streamlit App", page_icon=":shark:")
st.write("Upload an image or a zip file containing images and choose a model to make a prediction.")

# Choose model
model_type = st.sidebar.selectbox("Choose Model", ["SVM", "KNN", "RF"])

st.title(f"Image Prediction using {model_type}")

if model_type == "SVM":
    st.write("Support Vector Machine (SVM) is a powerful supervised learning model.")
elif model_type == "KNN":
    st.write("K-Nearest Neighbors (KNN) predicts based on the closest training examples.")
elif model_type == "RF":
    st.write("Random Forest is a powerful supervised learning model.")

# Choose single image upload or zip folder processing
upload_option = st.radio("Choose upload option:", ("Single Image", "Zip Folder"))

if upload_option == "Single Image":
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Read and display image using OpenCV
        img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Load model parameters and model
        if model_type == "SVM":
            with open(r"./src/checkpoint/svm_params_latest.json", "r") as f:
                params = json.load(f)
            model = joblib.load(r"./src/checkpoint/svm_best_model_latest.pkl")
        elif model_type == "KNN":
            with open(r"./src/checkpoint/knn_params_latest.json", "r") as f:
                params = json.load(f)
            model = joblib.load(r"./src/checkpoint/knn_best_model_latest.pkl")
        elif model_type == "RF":
            with open(r"./src/checkpoint/rf_params_latest.json", "r") as f:
                params = json.load(f)
            model = joblib.load(r"./src/checkpoint/rf_best_model_latest.pkl")

        # Preprocess image and predict
        img_size = params["img_size"]
        bin_size = params["bin"]
        img_preprocessed = preprocess_hsv(img, img_size=img_size, bin=bin_size)
        pred = model.predict([img_preprocessed])[0]

        if pred == 0:
            prediction = "Not smoking"
        else:
            prediction = "Smoking"

        color = "green" if prediction == "Not smoking" else "red"

        st.markdown(
            f"### Prediction ({model_type}): **:<span style='color:{color}'>{prediction}</span>**",
            unsafe_allow_html=True,
        )
        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            use_container_width=True,
        )

elif upload_option == "Zip Folder":
    uploaded_zip = st.file_uploader("Choose a zip file containing images", type=["zip"])

    if uploaded_zip is not None:
        # Confirm zip file upload
        st.write("Uploaded zip file:", uploaded_zip.name)

        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall("uploads")

        def get_all_images(folder_path):
            image_files = []
            seen_files = set()  # To keep track of files that have been added
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg')):
                        if file not in seen_files:
                            image_files.append(os.path.join(root, file))
                            seen_files.add(file)
            return image_files

        extracted_folder = "uploads"
        image_files = get_all_images(extracted_folder)

        # List image files for debugging
        # st.write("Image files:", image_files)

        cols = st.columns(5)  # Create 5 columns
        for i, image_file in enumerate(image_files):
            col = cols[i % 5]  # Select the correct column for this image
            with col:
                # Read and display image using OpenCV
                img = cv2.imread(image_file)
                if img is None:
                    st.write(f"Error loading image: {image_file}")
                    continue

                

                # Load model parameters and model
                if model_type == "SVM":
                    with open(r"./src/checkpoint/svm_params_latest.json", "r") as f:
                        params = json.load(f)
                    model = joblib.load(r"./src/checkpoint/svm_best_model_latest.pkl")
                elif model_type == "KNN":
                    with open(r"./src/checkpoint/knn_params_latest.json", "r") as f:
                        params = json.load(f)
                    model = joblib.load(r"./src/checkpoint/knn_best_model_latest.pkl")
                elif model_type == "RF":
                    with open(r"./src/checkpoint/rf_params_latest.json", "r") as f:
                        params = json.load(f)
                    model = joblib.load(r"./src/checkpoint/rf_best_model_latest.pkl")

                # Preprocess image and predict
                img_size = params["img_size"]
                bin_size = params["bin"]
                img_preprocessed = preprocess_hsv(img, img_size=img_size, bin=bin_size)
                pred = model.predict([img_preprocessed])[0]

                if pred == 0:
                    prediction = "Not smoking"
                else:
                    prediction = "Smoking"

                color = "green" if prediction == "Not smoking" else "red"

                st.markdown(
                    f"<p style='font-size:16px'><span style='color:{color}'>{prediction}</span></p>",
                    unsafe_allow_html=True,
                )

                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    caption=f"Image: {os.path.basename(image_file)}",
                    use_container_width=True,
                )
