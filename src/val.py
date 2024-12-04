# cd src to run this file

import os 
import joblib

from sklearn.metrics import classification_report, accuracy_score
from dataset_loader import load_data_from_folder

# model_name = "knn.pkl" # BIN_SIZE = 64
model_name = "svm.pkl" # BIN_SIZE = 16
# model_name = "rf.pkl" # BIN_SIZE = 16

model_path = os.path.join("../src/checkpoint/", model_name)
model = joblib.load(model_path)

val_dir = "../data/Validation/images"

X_val, y_val = load_data_from_folder(val_dir)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy for {model_name}: {accuracy}")
print("Classification Report:\n", classification_report(y_val, y_pred))