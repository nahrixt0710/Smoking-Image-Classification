from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def create_model(model_name="svm"):
    if model_name == "svm":
        return SVC(kernel="linear", C=0.001)  # linear không cần gamma, best accuracy : 0.7777777777777778

    elif model_name == "knn":
        return KNeighborsClassifier(n_neighbors=5, metric="manhattan")

    elif model_name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42)
