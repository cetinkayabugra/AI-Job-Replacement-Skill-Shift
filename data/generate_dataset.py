"""
Script to generate a synthetic AI Job Market Insights dataset
that mirrors the structure of the popular Kaggle dataset
'AI-Powered Job Market Insights'.
Run this script once to produce ai_job_market_insights.csv.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 500

JOB_TITLES = [
    "Data Scientist", "Software Engineer", "Marketing Analyst",
    "Financial Analyst", "HR Manager", "Graphic Designer",
    "Supply Chain Manager", "Customer Service Rep", "Cybersecurity Analyst",
    "Accountant", "Product Manager", "UX Designer",
    "Machine Learning Engineer", "DevOps Engineer", "Business Analyst",
    "Sales Manager", "Content Writer", "Legal Advisor",
    "Healthcare Administrator", "Logistics Coordinator",
]

INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Manufacturing",
    "Retail", "Education", "Transportation", "Media & Entertainment",
    "Energy", "Government",
]

COMPANY_SIZES = ["Small", "Medium", "Large"]

LOCATIONS = [
    "New York", "San Francisco", "Chicago", "Austin", "Seattle",
    "Boston", "Los Angeles", "Denver", "Atlanta", "Remote",
]

SKILLS_POOL = [
    "Python", "SQL", "Machine Learning", "Data Analysis", "Communication",
    "Project Management", "Excel", "Java", "Cloud Computing", "Cybersecurity",
    "Leadership", "Marketing Strategy", "Financial Modeling", "UX Research",
    "DevOps", "Agile", "Tableau", "Deep Learning", "NLP", "Power BI",
]

AI_ADOPTION = ["Low", "Medium", "High"]
AUTOMATION_RISK = ["Low", "Medium", "High"]
JOB_GROWTH = ["Decline", "Stable", "Growth"]

# Map automation risk and AI adoption to plausible salary ranges
risk_salary_mean = {"Low": 95_000, "Medium": 75_000, "High": 55_000}
adoption_salary_bonus = {"Low": 0, "Medium": 5_000, "High": 10_000}

rows = []
for _ in range(N):
    title = rng.choice(JOB_TITLES)
    industry = rng.choice(INDUSTRIES)
    company_size = rng.choice(COMPANY_SIZES, p=[0.3, 0.4, 0.3])
    location = rng.choice(LOCATIONS)
    ai_adoption = rng.choice(AI_ADOPTION, p=[0.25, 0.45, 0.30])
    automation_risk = rng.choice(AUTOMATION_RISK, p=[0.30, 0.40, 0.30])

    base_salary = risk_salary_mean[automation_risk] + adoption_salary_bonus[ai_adoption]
    salary = int(rng.normal(base_salary, 12_000))
    salary = max(30_000, salary)

    # Remote-friendly more likely for tech/high AI adoption
    remote_prob = 0.6 if (industry == "Technology" or ai_adoption == "High") else 0.35
    remote_friendly = "Yes" if rng.random() < remote_prob else "No"

    # Job growth: high automation risk -> more likely to decline
    if automation_risk == "High":
        growth_probs = [0.50, 0.35, 0.15]
    elif automation_risk == "Medium":
        growth_probs = [0.20, 0.50, 0.30]
    else:
        growth_probs = [0.05, 0.40, 0.55]
    job_growth = rng.choice(JOB_GROWTH, p=growth_probs)

    # Required skills: 2-5 random skills
    num_skills = rng.integers(2, 6)
    skills = ", ".join(rng.choice(SKILLS_POOL, size=int(num_skills), replace=False))

    # AI impact score (0-100)
    ai_impact_base = {"Low": 25, "Medium": 55, "High": 80}[automation_risk]
    ai_impact = int(np.clip(rng.normal(ai_impact_base, 10), 0, 100))

    # Tasks automated (percentage)
    tasks_automated = int(np.clip(rng.normal(ai_impact - 5, 8), 0, 100))

    rows.append({
        "Job_Title": title,
        "Industry": industry,
        "Company_Size": company_size,
        "Location": location,
        "AI_Adoption_Level": ai_adoption,
        "Automation_Risk": automation_risk,
        "Required_Skills": skills,
        "Salary_USD": salary,
        "Remote_Friendly": remote_friendly,
        "Job_Growth_Projection": job_growth,
        "AI_Impact_Score": ai_impact,
        "Tasks_Automated_Pct": tasks_automated,
    })

df = pd.DataFrame(rows)
out_path = "ai_job_market_insights.csv"
df.to_csv(out_path, index=False)
print(f"Dataset saved to {out_path} ({len(df)} rows, {len(df.columns)} columns)")
print(df.dtypes)
print(df.head(3))
