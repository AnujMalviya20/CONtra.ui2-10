# ML Model Training & Testing for Small and Large Datasets

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -----------------------------
# Models Dictionary
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# -----------------------------
# Datasets
# -----------------------------

datasets = {
    "Small Dataset (Iris)": load_iris(),
    "Large Dataset (Digits)": load_digits()
}

# -----------------------------
# Loop Through Datasets
# -----------------------------

for dataset_name, dataset in datasets.items():

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    X = dataset.data
    y = dataset.target

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # Loop Through Models
    # -----------------------------

    for model_name, model in models.items():

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{model_name:<25} Accuracy: {accuracy:.4f}")
