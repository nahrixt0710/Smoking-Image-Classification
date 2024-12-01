from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from dataset_loader import load_data_from_folder

# Base models
"""
base_models = [
    # ('svm_rbf', SVC(kernel='rbf')),
    ('svm_rbf', SVC(kernel='rbf', C=1, gamma=1)),
    # ('svm_linear', SVC(kernel="linear", C=0.1)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    # ('rf', RandomForestClassifier(n_estimators=100))
]
meta_model = LogisticRegression()

# Stacking Classifier
stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

Accuracy: 0.7722222222222223
"""

base_models = [
    # ('svm_rbf', SVC(kernel='rbf')),
    ("svm_rbf", SVC(kernel="rbf", C=1, gamma=1)),
    # ('svm_linear', SVC(kernel="linear", C=0.1)),
    ("knn", KNeighborsClassifier(n_neighbors=1)),
    # ('rf', RandomForestClassifier(n_estimators=50))
]

# Meta model
meta_model = LogisticRegression()

# Stacking Classifier
stacked_model = StackingClassifier(
    estimators=base_models, final_estimator=meta_model, cv=5
)

train_dir = "data/Training/images"
val_dir = "data/Validation/images"

X_train, y_train = load_data_from_folder(train_dir)
X_val, y_val = load_data_from_folder(val_dir)

# Train and evaluate
stacked_model.fit(X_train, y_train)
accuracy = stacked_model.score(X_val, y_val)
print("Accuracy:", accuracy)


"""
base_model = SVC(kernel='rbf', C=1, gamma=0.1)
bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)

train_dir = "data/Training/images"
val_dir = "data/Validation/images"

X_train, y_train = load_data_from_folder(train_dir)
X_val, y_val = load_data_from_folder(val_dir)

# Train and evaluate
bagging_model.fit(X_train, y_train)
accuracy = bagging_model.score(X_val, y_val)
print("Accuracy (Bagging):", accuracy)

Accuracy (Bagging): 0.7666666666666667
"""
