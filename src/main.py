from dataset_loader import load_data_from_folder
from model import create_model
from sklearn.metrics import classification_report, accuracy_score


def train_and_evaluate(train_dir, val_dir):
    # Load dữ liệu huấn luyện và kiểm thử
    X_train, y_train = load_data_from_folder(train_dir)
    X_val, y_val = load_data_from_folder(val_dir)

    # Tạo mô hình
    model = create_model()

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán và đánh giá
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_val, y_pred))


if __name__ == "__main__":
    # Đường dẫn đến các thư mục hình ảnh
    train_dir = "data/Training/images"
    val_dir = "data/Validation/images"

    # Gọi hàm huấn luyện và đánh giá
    train_and_evaluate(train_dir, val_dir)
