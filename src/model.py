from sklearn.svm import SVC


def create_model():
    return SVC(kernel="linear", C=0.1) # linear không cần gamma
    # return SVC(kernel="rbf", C=1, gamma=1)
