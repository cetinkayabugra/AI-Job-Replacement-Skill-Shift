# AI-Job-Replacement-Skill-Shift

A Python data science project analysing **AI's impact on the job market and skill shifts**, using a structured dataset that mirrors the popular [Kaggle AI-Powered Job Market Insights](https://www.kaggle.com/) dataset.

## Project Overview

The project addresses three machine learning tasks and produces rich visualisations:

| Task | Description | Models used |
|---|---|---|
| **Classification** | Predict Automation Risk (Low / Medium / High) | Random Forest, Logistic Regression, SVM |
| **Regression** | Predict Salary USD | Random Forest Regressor, Gradient Boosting |
| **Clustering** | Discover natural job groupings | K-Means (elbow method for k selection) |

## Dataset Columns

| Column | Type | Description |
|---|---|---|
| `Job_Title` | string | Job role name |
| `Industry` | string | Industry sector |
| `Company_Size` | string | Small / Medium / Large |
| `Location` | string | City or Remote |
| `AI_Adoption_Level` | string | Low / Medium / High |
| `Automation_Risk` | string | Low / Medium / High |
| `Required_Skills` | string | Comma-separated skill list |
| `Salary_USD` | int | Annual salary in USD |
| `Remote_Friendly` | string | Yes / No |
| `Job_Growth_Projection` | string | Decline / Stable / Growth |
| `AI_Impact_Score` | int | 0–100 score of AI impact |
| `Tasks_Automated_Pct` | int | Percentage of tasks automated |

## Project Structure

```
.
├── analysis.py                   # Main end-to-end analysis script
├── requirements.txt              # Python dependencies
├── data/
│   ├── ai_job_market_insights.csv  # Dataset (auto-generated if missing)
│   └── generate_dataset.py         # Synthetic data generator
├── src/
│   ├── data_preprocessing.py     # Loading, cleaning, encoding, feature engineering
│   ├── ml_models.py              # ML model training & evaluation
│   └── visualization.py          # All visualisation functions
└── plots/                        # Generated PNG charts (created at runtime)
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Bring your own dataset

Replace `data/ai_job_market_insights.csv` with your own CSV that follows the column schema above.  
If the file is absent, a realistic synthetic dataset (500 rows) is generated automatically.

### 3. Run the analysis

```bash
python analysis.py
```

To use a custom dataset path:

```bash
python analysis.py --data /path/to/your_dataset.csv
```

All charts are saved to the `plots/` directory.

## Visualisations Produced

| File | Description |
|---|---|
| `01_automation_risk_distribution.png` | Bar chart of automation risk counts |
| `02_automation_risk_by_industry.png` | Stacked bar: risk per industry |
| `03_salary_by_ai_adoption.png` | Box plot: salary vs AI adoption level |
| `04_job_growth_by_risk.png` | Grouped bar: job growth vs automation risk |
| `05_correlation_heatmap.png` | Correlation heatmap of numeric features |
| `06_top_required_skills.png` | Top 15 most demanded skills |
| `07_ai_impact_score_distribution.png` | Histogram + KDE of AI impact scores |
| `08_remote_vs_salary.png` | Violin plot: remote vs non-remote salary |
| `09_classification_feature_importance.png` | Feature importances (Random Forest classifier) |
| `09b_regression_feature_importance.png` | Feature importances (Random Forest regressor) |
| `10_confusion_matrix.png` | Confusion matrix for automation risk prediction |
| `11_roc_curves.png` | ROC curves: High risk vs rest (all classifiers) |
| `12_kmeans_clusters_pca.png` | K-Means clusters projected onto 2 PCA components |
| `13_salary_actual_vs_predicted.png` | Actual vs predicted salary scatter |
| `14_elbow_curve.png` | Elbow method: within-cluster SSE vs k |
