from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def create_model():
    return SVC(kernel="linear", C=0.1, gamma=0.001)


def create_knn(n_neighbors=3):
    return KNeighborsClassifier(n_neighbors=3)
