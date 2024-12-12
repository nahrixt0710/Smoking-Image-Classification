import os
import cv2
import numpy as np
from skimage import feature, io, color

BIN_SIZE = 16

def extract_color_histogram(image, bins=(BIN_SIZE, BIN_SIZE, BIN_SIZE)):
    hist = []
    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [bins[i]], [0, 256])
        channel_hist = cv2.normalize(channel_hist, channel_hist).flatten()
        hist.extend(channel_hist)
    return np.array(hist)

def preprocess(image):

    image = cv2.resize(image, (64, 64)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    feature = extract_color_histogram(image)

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

