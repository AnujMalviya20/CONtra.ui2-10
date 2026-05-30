import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("dataset.csv")

print("="*70)
print("DATASET INFORMATION")
print("="*70)

print(df.head())
print(df.info())
print(df.describe())

# ---------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------

df.fillna(df.median(numeric_only=True), inplace=True)

# ---------------------------------------------------
# SPLIT FEATURES & TARGET
# ---------------------------------------------------

TARGET_COLUMN = "target"

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier(),

    "Extra Trees":
        ExtraTreesClassifier(),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(),

    "Naive Bayes":
        GaussianNB()
}

# ---------------------------------------------------
# RESULTS STORAGE
# ---------------------------------------------------

results = []

# ---------------------------------------------------
# MODEL LOOP
# ---------------------------------------------------

for model_name, model in models.items():

    print("\n")
    print("="*70)
    print(f"TRAINING: {model_name}")
    print("="*70)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Training
    pipeline.fit(X_train, y_train)

    # Prediction
    predictions = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    # Cross Validation
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5
    )

    cv_mean = cv_scores.mean()

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"CV Score : {cv_mean:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "CV Score": cv_mean
    })

# ---------------------------------------------------
# RESULTS DATAFRAME
# ---------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*70)
print("FINAL MODEL COMPARISON")
print("="*70)

print(results_df)

# ---------------------------------------------------
# BEST MODEL
# ---------------------------------------------------

best_model = results_df.iloc[0]

print("\n")
print("="*70)
print("BEST MODEL")
print("="*70)

print(best_model)

# ---------------------------------------------------
# HYPERPARAMETER TUNING
# ---------------------------------------------------

print("\n")
print("="*70)
print("HYPERPARAMETER TUNING")
print("="*70)

param_grid = {

    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:")
print(grid.best_params_)

print("\nBest CV Score:")
print(grid.best_score_)

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

best_rf = grid.best_estimator_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("="*70)
print("TOP FEATURES")
print("="*70)

print(feature_importance.head(20))

# ---------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("dataset.csv")

print("="*70)
print("DATASET INFORMATION")
print("="*70)

print(df.head())
print(df.info())
print(df.describe())

# ---------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------

df.fillna(df.median(numeric_only=True), inplace=True)

# ---------------------------------------------------
# SPLIT FEATURES & TARGET
# ---------------------------------------------------

TARGET_COLUMN = "target"

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUM

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=2000),

    "Decision Tree":
        DecisionTreeClassifier(),

    "Random Forest":
        RandomForestClassifier(),

    "Extra Trees":
        ExtraTreesClassifier(),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(),

    "Naive Bayes":
        GaussianNB()
}

# ---------------------------------------------------
# RESULTS STORAGE
# ---------------------------------------------------

results = []

# ---------------------------------------------------
# MODEL LOOP
# ---------------------------------------------------

for model_name, model in models.items():

    print("\n")
    print("="*70)
    print(f"TRAINING: {model_name}")
    print("="*70)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # Training
    pipeline.fit(X_train, y_train)

    # Prediction
    predictions = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    # Cross Validation
    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5
    )

    cv_mean = cv_scores.mean()

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"CV Score : {cv_mean:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "CV Score": cv_mean
    })

# ---------------------------------------------------
# RESULTS DATAFRAME
# ---------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n")
print("="*70)
print("FINAL MODEL COMPARISON")
print("="*70)

print(results_df)

# ---------------------------------------------------
# BEST MODEL
# ---------------------------------------------------

best_model = results_df.iloc[0]

print("\n")
print("="*70)
print("BEST MODEL")
print("="*70)

print(best_model)

# ---------------------------------------------------
# HYPERPARAMETER TUNING
# ---------------------------------------------------

print("\n")
print("="*70)
print("HYPERPARAMETER TUNING")
print("="*70)

param_grid = {

    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:")
print(grid.best_params_)

print("\nBest CV Score:")
print(grid.best_score_)

# ---------------------------------------------------
# 
# ---------------------------------------------------

best_rf = grid.best_estimator_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
print("\nResults Saved Successfully 🚀")
