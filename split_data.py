import os
from sklearn.model_selection import train_test_split
import shutil

# Đường dẫn đến thư mục ảnh gốc
source_folder = "./data/images"
train_folder = "./data/train"
test_folder = "./data/test"

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Lấy danh sách các tệp ảnh
image_files = [
    f
    for f in os.listdir(source_folder)
    if os.path.isfile(os.path.join(source_folder, f))
]

# Chia dữ liệu thành train và test
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Di chuyển các tệp vào thư mục tương ứng
for file in train_files:
    shutil.copy2(os.path.join(source_folder, file), os.path.join(train_folder, file))

for file in test_files:
    shutil.copy2(os.path.join(source_folder, file), os.path.join(test_folder, file))

print("Done splitting images!")

notsmoking = 0
smoking = 0
for file_name in os.listdir(train_folder):
    if file_name.lower().startswith("notsmoking"):
        notsmoking += 1
    elif file_name.lower().startswith("smoking"):
        smoking += 1
print(f"Train folder: {notsmoking} not smoking, {smoking} smoking")

notsmoking = 0
smoking = 0
for file_name in os.listdir(test_folder):
    if file_name.lower().startswith("notsmoking"):
        notsmoking += 1
    elif file_name.lower().startswith("smoking"):
        smoking += 1
print(f"Test folder: {notsmoking} not smoking, {smoking} smoking")
