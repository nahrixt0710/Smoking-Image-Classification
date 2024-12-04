# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json
import cv2

BIN_SIZE = 32


def extract_color_histogram(image, bins=(BIN_SIZE, BIN_SIZE, BIN_SIZE), color_space="RGB"):
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


def create_model(n_estimators):
    return RandomForestClassifier(n_estimators=n_estimators,random_state=42)
    # return SVC(kernel=kernel, C=C, gamma=gamma)
    # return KNeighborsClassifier(n_neighbors=k, metric=distance_metric)


if __name__ == "__main__":
    rf_results = []
    img_sizes = [32, 64, 128]
    bins = [8, 16, 32, 64, 128]
    n_estimators_list = [10, 50, 100, 200]
    # C_list = [0.001, 0.01, 0.1, 1, 10]
    # gamma_list = [0.001, 0.01, 0.1, 1, 10]
    # k_list = [1, 3, 5, 7, 9]  
    # distance_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    # kernels = ["linear", "rbf"]

    img_size = 32
    for bin_size in bins:
        BIN_SIZE = bin_size
        train_dir = "C:\\study\\ComputerVision\\NhapMonCV_SmokingDetection\\NhapMonCV-Smoking-Image-Classification\\data\\Training\\images"  # Update this path if needed
        val_dir = "C:\\study\\ComputerVision\\NhapMonCV_SmokingDetection\\NhapMonCV-Smoking-Image-Classification\\data\\Validation\\images"  # Update this path if needed
        X_train, y_train = load_data_from_folder(train_dir, bin_size, img_size)
        X_val, y_val = load_data_from_folder(val_dir, bin_size, img_size)
        for n_estimators in n_estimators_list:
            # print(f"\n\nC: {C}, Bin Size: {bin_size}, Image Size: {img_size}")
            model = create_model(n_estimators)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            # print(f"\tAccuracy: {accuracy:.3f}")
            rf_results.append(
                {
                    "bin_size": bin_size,
                    "img_size": img_size,
                    "n_estimators": n_estimators,
                    "accuracy": round(accuracy, 4),
                }
            )

    # # Create the output directory if it does not exist
    output_dir = ""
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #     print(f"Created directory: {output_dir}")

    # Save results to JSON and Excel files in the specified directory
    json_path = os.path.join(output_dir, "rf_results.json")
    with open(json_path, "w") as f:
        json.dump(rf_results, f)
        print(f"Saved results to {json_path}")

    excel_path = os.path.join(output_dir, "rf_results.xlsx")
    df = pd.DataFrame(rf_results)
    df.to_excel(excel_path, index=True)
    print(f"Saved results to {excel_path}")

    # Verify if files are created
    if os.path.exists(json_path):
        print(f"JSON file successfully created at {json_path}")
    else:
        print("Failed to create JSON file")

    if os.path.exists(excel_path):
        print(f"Excel file successfully created at {excel_path}")
    else:
        print("Failed to create Excel file")
