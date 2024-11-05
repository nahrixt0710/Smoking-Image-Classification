from sklearn.svm import SVC


def create_model():
    return SVC(kernel="linear", C=0.1, gamma=0.001)
