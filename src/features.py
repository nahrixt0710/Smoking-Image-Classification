import cv2
import numpy as np
from skimage import feature, io, color

BIN_SIZE = 16


def extract_color_histogram(image, bins=(BIN_SIZE, BIN_SIZE, BIN_SIZE)):
    hist = []
    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [bins[i]], [0, 256])
        channel_hist = cv2.normalize(channel_hist, channel_hist).flatten()
        hist.extend(channel_hist)
    return np.array(hist)


# def extract_hog(image):

#     image = color.rgb2gray(image)

#     hog_features = feature.hog(
#         image,
#         pixels_per_cell=(8, 8),
#         cells_per_block=(2, 2),
#         block_norm="L2-Hys",
#         visualize=False,
#     )
#     return hog_features
