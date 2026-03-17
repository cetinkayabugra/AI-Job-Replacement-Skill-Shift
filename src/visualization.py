"""
Visualization functions for the AI Job Market Insights analysis.
All plots are saved as PNG files and also returned as Figure objects.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

PALETTE = "viridis"
FIGURE_DIR = "plots"

os.makedirs(FIGURE_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_automation_risk_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of Automation_Risk counts."""
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Low", "Medium", "High"]
    counts = df["Automation_Risk"].value_counts().reindex(order)
    sns.barplot(x=counts.index, y=counts.values, palette="RdYlGn_r", ax=ax)
    ax.set_title("Distribution of Automation Risk Levels")
    ax.set_xlabel("Automation Risk")
    ax.set_ylabel("Number of Jobs")
    for bar in ax.patches:
        ax.annotate(
            f"{int(bar.get_height())}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    _save(fig, "01_automation_risk_distribution.png")
    return fig


def plot_automation_risk_by_industry(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of automation risk per industry."""
    order = ["Low", "Medium", "High"]
    pivot = (
        df.groupby(["Industry", "Automation_Risk"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order)
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, colormap="RdYlGn_r", ax=ax)
    ax.set_title("Automation Risk by Industry")
    ax.set_xlabel("Industry")
    ax.set_ylabel("Number of Jobs")
    ax.legend(title="Risk Level", loc="upper right")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    _save(fig, "02_automation_risk_by_industry.png")
    return fig


def plot_salary_by_ai_adoption(df: pd.DataFrame) -> plt.Figure:
    """Box plot of salary grouped by AI adoption level."""
    order = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x="AI_Adoption_Level", y="Salary_USD",
        order=order, palette="Blues", ax=ax,
    )
    ax.set_title("Salary Distribution by AI Adoption Level")
    ax.set_xlabel("AI Adoption Level")
    ax.set_ylabel("Salary (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    _save(fig, "03_salary_by_ai_adoption.png")
    return fig


def plot_job_growth_projection(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of job growth projection vs automation risk."""
    order_risk = ["Low", "Medium", "High"]
    order_growth = ["Decline", "Stable", "Growth"]
    pivot = (
        df.groupby(["Automation_Risk", "Job_Growth_Projection"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=order_risk, columns=order_growth)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, colormap="RdYlGn")
    ax.set_title("Job Growth Projection by Automation Risk Level")
    ax.set_xlabel("Automation Risk Level")
    ax.set_ylabel("Number of Jobs")
    ax.legend(title="Growth Projection")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save(fig, "04_job_growth_by_risk.png")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of numeric feature correlations."""
    numeric_cols = ["Salary_USD", "AI_Impact_Score", "Tasks_Automated_Pct"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    _save(fig, "05_correlation_heatmap.png")
    return fig


def plot_top_skills(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Horizontal bar chart of the most frequently required skills."""
    all_skills: list[str] = []
    for entry in df["Required_Skills"].dropna():
        all_skills.extend([s.strip() for s in entry.split(",")])
    skill_counts = pd.Series(all_skills).value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=skill_counts.values, y=skill_counts.index, palette=PALETTE, ax=ax)
    ax.set_title(f"Top {top_n} Most Required Skills")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Skill")
    plt.tight_layout()
    _save(fig, "06_top_required_skills.png")
    return fig


def plot_ai_impact_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram of AI Impact Score with KDE."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["AI_Impact_Score"], bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of AI Impact Score")
    ax.set_xlabel("AI Impact Score (0-100)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _save(fig, "07_ai_impact_score_distribution.png")
    return fig


def plot_remote_vs_salary(df: pd.DataFrame) -> plt.Figure:
    """Violin plot comparing salary for remote vs non-remote jobs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=df, x="Remote_Friendly", y="Salary_USD",
        palette={"Yes": "skyblue", "No": "salmon"}, ax=ax,
    )
    ax.set_title("Salary Distribution: Remote vs Non-Remote Jobs")
    ax.set_xlabel("Remote Friendly")
    ax.set_ylabel("Salary (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    _save(fig, "08_remote_vs_salary.png")
    return fig


# ---------------------------------------------------------------------------
# ML result plots
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "Feature Importances",
    filename: str = "09_feature_importance.png",
    top_n: int = 15,
) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
    )
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save(fig, filename)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str = "Confusion Matrix",
    filename: str = "10_confusion_matrix.png",
) -> plt.Figure:
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, filename)
    return fig


def plot_roc_curves(
    models: dict,
    X_test: np.ndarray,
    y_test_bin: np.ndarray,
    high_class_index: int = 2,
    filename: str = "11_roc_curves.png",
) -> plt.Figure:
    """
    Overlay ROC curves for multiclass models evaluated in a
    one-vs-rest manner for the 'High' automation-risk class.

    Parameters
    ----------
    models : dict
        {name: fitted sklearn estimator with predict_proba}
    X_test : array-like
        Test feature matrix.
    y_test_bin : array-like
        Binarised target (1 = High risk, 0 = Low/Medium).
    high_class_index : int
        Index of the 'High' class in model.classes_.
    filename : str
        Output PNG filename saved under FIGURE_DIR.
    """
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, high_class_index]
            RocCurveDisplay.from_predictions(
                y_test_bin, proba, name=name, ax=ax
            )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves – Automation Risk (High vs Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, filename)
    return fig


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title: str = "K-Means Clusters (PCA Projection)",
    filename: str = "12_kmeans_clusters_pca.png",
) -> plt.Figure:
    """Scatter plot of K-Means cluster assignments projected onto 2 PCA components."""
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        components[:, 0], components[:, 1],
        c=labels, cmap="tab10", alpha=0.7, s=30,
    )
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    _save(fig, filename)
    return fig


def plot_regression_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Salary",
    filename: str = "13_salary_actual_vs_predicted.png",
) -> plt.Figure:
    """Scatter plot comparing actual vs predicted salary values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="steelblue", s=25)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect fit")
    ax.set_title(title)
    ax.set_xlabel("Actual Salary (USD)")
    ax.set_ylabel("Predicted Salary (USD)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)
    return fig


def plot_elbow_curve(
    inertias: list[float],
    k_range: range,
    filename: str = "14_elbow_curve.png",
) -> plt.Figure:
    """
    Elbow method plot for selecting optimal K in K-Means.

    Parameters
    ----------
    inertias : list[float]
        Within-cluster SSE for each value of k.
    k_range : range
        The range of k values evaluated (e.g. range(2, 9)).
    filename : str
        Output PNG filename saved under FIGURE_DIR.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, "bo-")
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (Within-cluster SSE)")
    plt.tight_layout()
    _save(fig, filename)
    return fig
