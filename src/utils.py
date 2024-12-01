import numpy as np

def normalize_data(data):

    # Tính norm (độ dài) của từng vector
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    
    # Tránh chia cho 0 bằng cách thay norm = 0 thành 1
    norms[norms == 0] = 1
    
    # Chuẩn hóa dữ liệu
    normalized_data = data / norms
    return normalized_data
