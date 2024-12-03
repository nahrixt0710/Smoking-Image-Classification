import os
import cv2
import numpy as np
from features import *


def normalize_data(data):

    # Tính norm (độ dài) của từng vector
    norms = np.linalg.norm(data, axis=1, keepdims=True)

    # Tránh chia cho 0 bằng cách thay norm = 0 thành 1
    norms[norms == 0] = 1

    # Chuẩn hóa dữ liệu
    normalized_data = data / norms
    return normalized_data


def preprocess(image):

    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    feature = extract_color_histogram(image)
    # feature = extract_hog(image)

    # feature = normalize_data(np.array([feature]))[0]

    return feature


def load_data_from_folder(folder_path):
    data = []
    labels = []

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)

        if image is not None:
            features = preprocess(image)
            data.append(features)
            if img_name.lower().startswith("notsmoking"):
                labels.append(0)
            elif img_name.lower().startswith("smoking"):
                labels.append(1)
        else:
            print(f"Can't load image: {img_name}")

    return np.array(data), np.array(labels)


def predict(image, model):

    features = preprocess(image)
    prediction = model.predict([features])

    return "Not smoking" if prediction[0] == 0 else "Smoking"
