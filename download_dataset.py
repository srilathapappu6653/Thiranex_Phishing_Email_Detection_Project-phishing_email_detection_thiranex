from pathlib import Path
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DATASET_URL = "https://zenodo.org/records/13474746/files/Phishing_validation_emails.csv?download=1"
OUTPUT_FILE = DATA_DIR / "Phishing_validation_emails.csv"

def download_dataset():
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 1000:
        print(f"Dataset already exists: {OUTPUT_FILE}")
        return OUTPUT_FILE

    print("Downloading the 2,000-email Zenodo dataset...")
    try:
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
        OUTPUT_FILE.write_bytes(response.content)
        print(f"Saved dataset to: {OUTPUT_FILE}")
        return OUTPUT_FILE
    except Exception as exc:
        print("Dataset download failed:", exc)
        print("The training script will use data/sample_emails.csv instead.")
        return None

if __name__ == "__main__":
    download_dataset()
