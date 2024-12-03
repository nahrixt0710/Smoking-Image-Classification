from dataset_loader import load_data_from_folder
from model import create_model
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_and_evaluate(train_dir, val_dir, model_name):

    X_train, y_train = load_data_from_folder(train_dir)
    X_val, y_val = load_data_from_folder(val_dir)

    model = create_model(model_name)
    model.fit(X_train, y_train)

    file_path = f"src/checkpoint/{model_name}.pkl"
    joblib.dump(model, file_path)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy for {model_name}: {accuracy}")
    print("Classification Report:\n", classification_report(y_val, y_pred))


if __name__ == "__main__":

    train_dir = "data/Training/images"
    val_dir = "data/Validation/images"

    model_name = "svm"

    train_and_evaluate(train_dir, val_dir, model_name)
