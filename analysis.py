"""
AI Job Market Insights – End-to-End Analysis
=============================================
This script loads the AI Job Market Insights dataset, performs exploratory
data analysis (EDA), trains multiple machine learning models, and saves all
visualizations to the `plots/` directory.

Usage:
    python analysis.py [--data PATH_TO_CSV]

If --data is not provided, the default path
    data/ai_job_market_insights.csv
is used.  If the file does not exist, a synthetic dataset is generated
automatically.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure the project root is on the Python path when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import (
    build_feature_matrix,
    clean_data,
    load_data,
    scale_features,
    encode_features,
    ORDINAL_MAPS,
)
from src.ml_models import (
    compute_elbow,
    evaluate_classifiers,
    evaluate_regressors,
    train_classifiers,
    train_kmeans,
    train_regressors,
    cross_validate_classifier,
)
from src.visualization import (
    plot_ai_impact_distribution,
    plot_automation_risk_by_industry,
    plot_automation_risk_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_elbow_curve,
    plot_feature_importance,
    plot_job_growth_projection,
    plot_pca_clusters,
    plot_regression_actual_vs_predicted,
    plot_remote_vs_salary,
    plot_roc_curves,
    plot_salary_by_ai_adoption,
    plot_top_skills,
)


DEFAULT_DATA_PATH = os.path.join("data", "ai_job_market_insights.csv")
DATASET_GENERATOR = os.path.join("data", "generate_dataset.py")
PLOTS_DIR = "plots"


def _ensure_dataset(path: str) -> None:
    """Generate the dataset if the CSV file is not found."""
    if not os.path.exists(path):
        print(f"Dataset not found at '{path}'. Generating synthetic data …")
        import subprocess
        subprocess.run([sys.executable, DATASET_GENERATOR], check=True)
        # Generator saves to cwd; move if needed
        generated = "ai_job_market_insights.csv"
        if os.path.exists(generated) and not os.path.exists(path):
            os.replace(generated, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not create dataset at {path}")


def main(data_path: str = DEFAULT_DATA_PATH) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & clean data
    # ------------------------------------------------------------------
    _ensure_dataset(data_path)
    print(f"\n{'='*60}")
    print("  AI Job Market Insights – Analysis")
    print(f"{'='*60}")
    print(f"\n[1/5] Loading data from '{data_path}' …")
    df = load_data(data_path)
    df = clean_data(df)
    print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")
    print("\n  Value counts – Automation_Risk:")
    print(df["Automation_Risk"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 2. EDA Visualizations
    # ------------------------------------------------------------------
    print("\n[2/5] Generating EDA visualizations …")
    plot_automation_risk_distribution(df)
    plot_automation_risk_by_industry(df)
    plot_salary_by_ai_adoption(df)
    plot_job_growth_projection(df)
    plot_correlation_heatmap(df)
    plot_top_skills(df)
    plot_ai_impact_distribution(df)
    plot_remote_vs_salary(df)

    # ------------------------------------------------------------------
    # 3. Classification – Predict Automation_Risk
    # ------------------------------------------------------------------
    print("\n[3/5] Classification – Predicting Automation Risk …")
    X_cls, y_cls = build_feature_matrix(df, target="Automation_Risk")
    X_tr_cls, X_te_cls, y_tr_cls, y_te_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    X_tr_sc, X_te_sc, scaler_cls = scale_features(X_tr_cls, X_te_cls)

    classifiers = train_classifiers(X_tr_sc, y_tr_cls)
    label_names = ["Low", "Medium", "High"]
    clf_results = evaluate_classifiers(classifiers, X_te_sc, y_te_cls, label_names)
    print("\n  Classification results summary:")
    print(clf_results.to_string(index=False))

    # Cross-validation for best classifier (Random Forest)
    rf_clf = classifiers["Random Forest"]
    cv_result = cross_validate_classifier(rf_clf, X_tr_sc, y_tr_cls)
    print(f"\n  Random Forest 5-fold CV accuracy: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}")

    # Feature importance
    feat_names = list(X_cls.columns)
    plot_feature_importance(
        feat_names,
        rf_clf.feature_importances_,
        title="Random Forest – Feature Importances (Automation Risk)",
        filename="09_classification_feature_importance.png",
    )

    # Confusion matrix for best model
    y_pred_rf = rf_clf.predict(X_te_sc)
    plot_confusion_matrix(
        y_te_cls, y_pred_rf, label_names,
        title="Confusion Matrix – Random Forest (Automation Risk)",
    )

    # ROC curves (binarise: High vs rest)
    high_idx = ORDINAL_MAPS["Automation_Risk"]["High"]
    y_te_bin = (y_te_cls == high_idx).astype(int)
    plot_roc_curves(classifiers, X_te_sc, y_te_bin, high_class_index=high_idx)

    # ------------------------------------------------------------------
    # 4. Regression – Predict Salary_USD
    # ------------------------------------------------------------------
    print("\n[4/5] Regression – Predicting Salary_USD …")
    # Build features without using Salary_USD as a target-encoding proxy
    X_reg, _ = build_feature_matrix(df, target="Automation_Risk", include_skills=True)
    # Drop Salary_USD from features (it is our regression target)
    X_reg = X_reg.drop(columns=["Salary_USD"], errors="ignore")
    y_reg = df["Salary_USD"]

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    X_tr_r_sc, X_te_r_sc, scaler_reg = scale_features(X_tr_r, X_te_r)

    regressors = train_regressors(X_tr_r_sc, y_tr_r)
    reg_results = evaluate_regressors(regressors, X_te_r_sc, y_te_r)
    print("\n  Regression results summary:")
    print(reg_results.to_string(index=False))

    # Best regressor feature importance + actual vs predicted
    rf_reg = regressors["Random Forest Regressor"]
    plot_feature_importance(
        list(X_reg.columns),
        rf_reg.feature_importances_,
        title="Random Forest – Feature Importances (Salary Prediction)",
        filename="09b_regression_feature_importance.png",
    )
    y_pred_reg = rf_reg.predict(X_te_r_sc)
    plot_regression_actual_vs_predicted(y_te_r.values, y_pred_reg)

    # ------------------------------------------------------------------
    # 5. Clustering – K-Means
    # ------------------------------------------------------------------
    print("\n[5/5] Clustering – K-Means on numeric features …")
    numeric_cols = ["Salary_USD", "AI_Impact_Score", "Tasks_Automated_Pct"]
    df_enc = encode_features(df)
    ordinal_enc_cols = ["AI_Adoption_Level_enc", "Automation_Risk_enc", "Company_Size_enc"]
    cluster_cols = numeric_cols + [c for c in ordinal_enc_cols if c in df_enc.columns]
    X_clust = df_enc[cluster_cols].dropna().values

    from sklearn.preprocessing import StandardScaler
    X_clust_sc = StandardScaler().fit_transform(X_clust)

    # Elbow method
    k_range = range(2, 9)
    inertias = compute_elbow(X_clust_sc, k_range)
    plot_elbow_curve(inertias, k_range)

    # Fit final model with k=4
    km = train_kmeans(X_clust_sc, n_clusters=4)
    print(f"  K-Means (k=4) inertia: {km.inertia_:.1f}")
    plot_pca_clusters(X_clust_sc, km.labels_)

    # Cluster profiles
    df_cluster = df_enc.loc[df_enc[cluster_cols].notna().all(axis=1)].copy()
    df_cluster["Cluster"] = km.labels_
    cluster_profile = df_cluster.groupby("Cluster")[numeric_cols].mean().round(1)
    print("\n  Cluster profiles (mean of numeric features):")
    print(cluster_profile.to_string())

    print(f"\n{'='*60}")
    print(f"  Analysis complete. All plots saved to '{PLOTS_DIR}/'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Job Market Insights Analysis")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the dataset CSV (default: {DEFAULT_DATA_PATH})",
    )
    args = parser.parse_args()
    main(data_path=args.data)
