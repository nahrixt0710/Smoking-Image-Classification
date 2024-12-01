from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_model(model_name="svm"):
    if model_name == "svm":
        return SVC(
            kernel="linear", C=0.01
        )  # linear không cần gamma, best accuracy : 0.7611111111111111

    elif model_name == "knn":
        return KNeighborsClassifier(n_neighbors=5)

    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=100)

    # return SVC(kernel="rbf", C=1, gamma=1)
