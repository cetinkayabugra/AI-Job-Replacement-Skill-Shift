# AI-Job-Replacement-Skill-Shift
AI Job Replacement Skill Shift. using Kaggle Dataset

## Dataset

The dataset `ai_job_replacement_2020_2026_v2.csv` is included in this repository. It contains **15,000 rows** and **20 columns** covering AI-driven job replacement and skill shift trends from 2020 to 2026.

| Column | Description |
|---|---|
| `job_id` | Unique identifier for each record |
| `job_role` | Job title |
| `industry` | Industry sector |
| `country` | Country |
| `year` | Year (2020–2026) |
| `automation_risk_percent` | Likelihood of automation (%) |
| `ai_replacement_score` | AI replacement score (0–100) |
| `skill_gap_index` | Skill gap index |
| `salary_before_usd` | Salary before AI disruption (USD) |
| `salary_after_usd` | Salary after AI disruption (USD) |
| `salary_change_percent` | Percentage salary change |
| `skill_demand_growth_percent` | Skill demand growth (%) |
| `remote_feasibility_score` | Remote work feasibility score |
| `ai_adoption_level` | AI adoption level |
| `education_requirement_level` | Education level required |
| `automation_risk_category` | Categorical risk level (Low / Medium / High) |
| `skill_transition_pressure` | Pressure to transition skills |
| `wage_volatility_index` | Wage volatility index |
| `reskilling_urgency_score` | Urgency of reskilling |
| `ai_disruption_intensity` | Overall AI disruption intensity |

The dataset is also available on Kaggle: [AI Job Replacement and Skill Shift Dataset](https://www.kaggle.com/datasets/dmahajanbe23/ai-job-replacement-and-skill-shift-dataset).

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Analyze the Dataset

Run the analysis script to get a quick overview of the local CSV:

```bash
python analyze_dataset.py
```

This prints a summary of the dataset, including basic statistics, automation risk distribution, top job roles by AI replacement score, and average salary change by industry.

## Download Dataset from Kaggle (Optional)

If you want to re-download the dataset from Kaggle, configure your API credentials first:

1. Sign in to [Kaggle](https://www.kaggle.com) and go to **Account → API → Create New Token**.
2. This downloads a `kaggle.json` file containing your credentials.
3. Place `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows).

For more details, see the [Kaggle API authentication docs](https://www.kaggle.com/docs/api#authentication).

Then run:

```bash
python download_dataset.py
```
