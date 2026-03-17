"""
Machine Learning models for the AI Job Market Insights project.

Three tasks are addressed:
1. Classification  – predict Automation_Risk (Low / Medium / High)
2. Regression      – predict Salary_USD
3. Clustering      – discover natural job groupings with K-Means
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> dict:
    """Train multiple classification models and return them in a dict."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=random_state),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_classifiers(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate classifiers and return a summary DataFrame."""
    records = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        records.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            }
        )
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"  Accuracy: {records[-1]['Accuracy']:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
    return pd.DataFrame(records)


def cross_validate_classifier(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """Return cross-validation accuracy scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {"mean": scores.mean(), "std": scores.std(), "scores": scores}


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def train_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> dict:
    """Train regression models and return them."""
    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_regressors(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate regressors and return a summary DataFrame."""
    records = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        records.append({"Model": name, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R²": round(r2, 4)})
        print(f"\n  {name}: RMSE={rmse:,.0f}  MAE={mae:,.0f}  R²={r2:.4f}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def compute_elbow(X: np.ndarray, k_range: range = range(2, 11)) -> list[float]:
    """Compute within-cluster SSE for each k."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)
    return inertias


def train_kmeans(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> KMeans:
    """Fit a K-Means model and return it."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(X)
    return km
