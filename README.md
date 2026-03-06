# AI-Job-Replacement-Skill-Shift
AI Job Replacement Skill Shift. using Kaggle Dataset

## Dataset

This project uses the [AI Job Replacement and Skill Shift Dataset](https://www.kaggle.com/datasets/dmahajanbe23/ai-job-replacement-and-skill-shift-dataset) from Kaggle.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Kaggle Authentication

Before downloading the dataset, configure your Kaggle API credentials:

1. Sign in to [Kaggle](https://www.kaggle.com) and go to **Account → API → Create New Token**.
2. This downloads a `kaggle.json` file containing your credentials.
3. Place `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows).

For more details, see the [Kaggle API authentication docs](https://www.kaggle.com/docs/api#authentication).

## Download Dataset

Run the following script to download the dataset:

```bash
python download_dataset.py
```

This will download the dataset to your local Kaggle cache and print the path to the downloaded files.
