import joblib
import cv2
from dataset_loader import *


def predict(image, model):
    features = preprocess(image)
    prediction = model.predict([features])
    return "Not smoking" if prediction[0] == 0 else "Smoking"


if __name__ == "__main__":

    model_path = "src/svm_model.pkl"
    model = joblib.load("src/svm_model.pkl")

    root = "C:\\study\\ComputerVision\\NhapMonCV_SmokingDetection\\NhapMonCV-Smoking-Image-Classification\\data\\Testing\\images"
    image_path = os.path.join(root,"notsmoking_0534.jpg")

    image = cv2.imread(image_path)
    print(predict(image, model))
