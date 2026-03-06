import kagglehub

# Download latest version
try:
    path = kagglehub.dataset_download("dmahajanbe23/ai-job-replacement-and-skill-shift-dataset")
    print("Path to dataset files:", path)
except Exception as e:
    print(f"Failed to download dataset: {e}")
    print("Ensure your Kaggle API credentials are configured.")
    print("See: https://www.kaggle.com/docs/api#authentication")
