import cv2
import numpy as np

BIN_SIZE = 64


def extract_color_histogram(
    image, bins=(BIN_SIZE, BIN_SIZE, BIN_SIZE), color_space="RGB"
):
    hist = []
    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [bins[i]], [0, 256])
        channel_hist = cv2.normalize(channel_hist, channel_hist).flatten()
        hist.extend(channel_hist)
    return np.array(hist)
