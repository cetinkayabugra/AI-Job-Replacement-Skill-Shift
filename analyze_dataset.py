import os
import pandas as pd

CSV_FILE = "ai_job_replacement_2020_2026_v2.csv"

if not os.path.exists(CSV_FILE):
    print(f"Error: '{CSV_FILE}' not found.")
    print("Make sure the file is in the same directory as this script.")
    print("You can also download it by running: python download_dataset.py")
    raise SystemExit(1)

df = pd.read_csv(CSV_FILE)

print("=== Dataset Overview ===")
print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}\n")

print("=== First 5 Rows ===")
print(df.head().to_string(index=False))
print()

print("=== Basic Statistics ===")
print(df.describe().to_string())
print()

print("=== Automation Risk Distribution ===")
print(df["automation_risk_category"].value_counts().to_string())
print()

print("=== Top 10 Job Roles by Average AI Replacement Score ===")
top_roles = (
    df.groupby("job_role")["ai_replacement_score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
print(top_roles.to_string())
print()

print("=== Average Salary Change by Industry ===")
salary_by_industry = (
    df.groupby("industry")["salary_change_percent"]
    .mean()
    .sort_values()
)
print(salary_by_industry.to_string())
