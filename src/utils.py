import os
import cv2
import numpy as np
from skimage import feature, io, color

BIN_SIZE = 8


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


def preprocess2(image, img_size, bin):

    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    feature = extract_color_histogram(image, (bin, bin, bin))

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


# KNN predict func in app
def knn_predict(image, model, bin_size=16, image_size=128):
    # Preprocess image
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = extract_color_histogram(image, bins=(bin_size, bin_size, bin_size))

    prediction = model.predict([features])
    return "Not smoking" if prediction[0] == 0 else "Smoking"


# utils to optimize bin size and image size


def load_raw_data(folder_path):
    data = []
    labels = []

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)

        if image is not None:
            data.append(image)
            if img_name.lower().startswith("notsmoking"):
                labels.append(0)
            elif img_name.lower().startswith("smoking"):
                labels.append(1)
        else:
            print(f"Can't load image: {img_name}")

    return np.array(data), np.array(labels)


def calc_hist(image, bin):
    bins = (bin, bin, bin)
    hist = []
    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [bins[i]], [0, 256])
        channel_hist = cv2.normalize(channel_hist, channel_hist).flatten()
        hist.extend(channel_hist)
    return np.array(hist)


class CustomFeatureExtractor:
    def __init__(self, bins=16, image_size=128):
        self.bins = bins
        self.image_size = image_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed = []
        for image in X:
            if self.image_size:
                image = cv2.resize(image, (self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            hist = calc_hist(image, self.bins)
            transformed.append(hist)
        return np.array(transformed)

    def set_params(self, **params):
        # Cập nhật tham số
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        # Lấy tham số
        return {"bins": self.bins, "image_size": self.image_size}
