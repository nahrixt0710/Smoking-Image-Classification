from dataset_loader import load_data_from_folder
from model import *
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import json
import numpy as np
import cv2
import pandas as pd

BIN_SIZE = 32


def extract_color_histogram(
    image, bins=(BIN_SIZE, BIN_SIZE, BIN_SIZE), color_space="RGB"
):
    hist = []
    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [bins[i]], [0, 256])
        channel_hist = cv2.normalize(channel_hist, channel_hist).flatten()
        hist.extend(channel_hist)
    return np.array(hist)


def preprocess(image, bin_size, img_size):

    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    feature = extract_color_histogram(image, bins=(bin_size, bin_size, bin_size))
    return feature


def load_data_from_folder(folder_path, bin_size, img_size):
    data = []
    labels = []

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)

        if image is not None:
            features = preprocess(image, bin_size, img_size)
            data.append(features)
            if img_name.lower().startswith("notsmoking"):
                labels.append(0)
            elif img_name.lower().startswith("smoking"):
                labels.append(1)
        else:
            print(f"Can't load image: {img_name}")

    return np.array(data), np.array(labels)


def train_and_evaluate(k, bin_size, img_size):
    joblib.dump(model, "src/checkpoint/color_histogram.pkl")
    # print("Classification Report:\n", classification_report(y_val, y_pred))


if __name__ == "__main__":

    knn_results = []
    img_sizes = [32, 64, 128]
    bins = [8, 16, 32, 64, 128]
    K = [1, 3, 5, 7, 9]
    for bin_size in bins:
        BIN_SIZE = bin_size
        for img_size in img_sizes:
            train_dir = "data/Training/images"
            val_dir = "data/Validation/images"
            X_train, y_train = load_data_from_folder(train_dir, bin_size, img_size)
            X_val, y_val = load_data_from_folder(val_dir, bin_size, img_size)

            for k in K:
                print(f"\n\nK: {k}, Bin Size: {bin_size}, Image Size: {img_size}")
                model = create_knn(n_neighbors=k)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                print(f"\tAccuracy: {accuracy:.2f}")
                knn_results.append(
                    {
                        "k_neighbors": k,
                        "bin_size": bin_size,
                        "img_size": img_size,
                        "accuracy": round(accuracy, 3),
                    }
                )

    with open("knn_results.json", "w") as f:
        json.dump(knn_results, f)

    df = pd.DataFrame(knn_results)
    df.to_excel("knn_results.xlsx", index=True)
