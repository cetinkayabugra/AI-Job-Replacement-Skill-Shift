"""
Data loading, cleaning, and feature engineering for the
AI Job Market Insights dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


ORDINAL_MAPS = {
    "AI_Adoption_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Automation_Risk": {"Low": 0, "Medium": 1, "High": 2},
    "Company_Size": {"Small": 0, "Medium": 1, "Large": 2},
    "Job_Growth_Projection": {"Decline": 0, "Stable": 1, "Growth": 2},
    "Remote_Friendly": {"No": 0, "Yes": 1},
}


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV dataset and return a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and handle missing values."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["Salary_USD", "Automation_Risk", "Job_Growth_Projection"])
    df["Salary_USD"] = df["Salary_USD"].clip(lower=0)
    return df.reset_index(drop=True)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns:
    - Ordinal columns are mapped to integers.
    - Nominal columns (Job_Title, Industry, Location) are label-encoded.
    Returns a new DataFrame with encoded columns added (suffix _enc).
    """
    df = df.copy()

    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(mapping)

    for col in ["Job_Title", "Industry", "Location"]:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))

    return df


def extract_skill_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the top-N most common individual skills from
    the Required_Skills column.
    """
    df = df.copy()
    all_skills: list[str] = []
    for entry in df["Required_Skills"].dropna():
        all_skills.extend([s.strip() for s in entry.split(",")])

    skill_counts = pd.Series(all_skills).value_counts()
    top_skills = skill_counts.head(15).index.tolist()

    for skill in top_skills:
        col_name = f"skill_{skill.replace(' ', '_').replace('/', '_')}"
        df[col_name] = df["Required_Skills"].apply(
            lambda x: int(skill in str(x))
        )

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target: str,
    include_skills: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix X and target series y for modelling.
    Returns (X, y).
    """
    df = encode_features(df)
    if include_skills:
        df = extract_skill_features(df)

    numeric_cols = ["Salary_USD", "AI_Impact_Score", "Tasks_Automated_Pct"]
    ordinal_enc_cols = [f"{c}_enc" for c in ORDINAL_MAPS if f"{c}_enc" in df.columns and c != target]
    nominal_enc_cols = [c for c in ["Job_Title_enc", "Industry_enc", "Location_enc"] if c in df.columns]
    skill_cols = [c for c in df.columns if c.startswith("skill_")]

    feature_cols = numeric_cols + ordinal_enc_cols + nominal_enc_cols + skill_cols
    # Remove target-derived column if present
    target_enc = f"{target}_enc"
    feature_cols = [c for c in feature_cols if c != target_enc and c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_enc] if target_enc in df.columns else df[target]

    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on training data and transform both splits."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler
