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

import cv2
import numpy as np

def extract_color_histogram_hsv(image, bins_h=180, bins_s=256, bins_v=256):
    
    # Tính histogram cho từng kênh HSV
    hist_hue = cv2.calcHist([image], [0], None, [bins_h], [0, 180])
    hist_saturation = cv2.calcHist([image], [1], None, [bins_s], [0, 256])
    hist_value = cv2.calcHist([image], [2], None, [bins_v], [0, 256])
    
    # Chuẩn hóa histogram về khoảng [0, 1]
    hist_hue = cv2.normalize(hist_hue, hist_hue, 0, 1, cv2.NORM_MINMAX)
    hist_saturation = cv2.normalize(hist_saturation, hist_saturation, 0, 1, cv2.NORM_MINMAX)
    hist_value = cv2.normalize(hist_value, hist_value, 0, 1, cv2.NORM_MINMAX)
    
    # Ghép các histogram lại thành một vector duy nhất
    hist_vector = np.concatenate((hist_hue.flatten(),
                                   hist_saturation.flatten(),
                                   hist_value.flatten()))
    
    return hist_vector

# not used
def preprocess_hsv(image, img_size=250, bins_h=180, bins_s=256, bins_v=256):

    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # feature = extract_color_histogram(image)
    feature = extract_color_histogram_hsv(image, bins_h, bins_s, bins_v)

    return feature


# using
def preprocess2(image, img_size=250, bin=16):

    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    feature = extract_color_histogram(image, (bin, bin, bin))

    return feature


# not used
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

def load_raw_data(folder_path):
    data = []
    labels = []

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        # change to 256x256 to have the same array size
        image = cv2.resize(image, (256, 256))

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
