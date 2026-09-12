from pathlib import Path
import re
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from download_dataset import download_dataset
from utils import EmailSecurityFeatures

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_data():
    full_dataset = DATA_DIR / "Phishing_validation_emails.csv"

    if not full_dataset.exists():
        download_dataset()

    if full_dataset.exists():
        df = pd.read_csv(full_dataset)
        text_col = next((c for c in df.columns if c.lower().strip() in
                         ["email text", "email_text", "text", "body", "email content"]), None)
        label_col = next((c for c in df.columns if c.lower().strip() in
                          ["email type", "label", "class", "target"]), None)

        if text_col is None or label_col is None:
            raise ValueError(
                f"Could not identify text/label columns. Found: {list(df.columns)}"
            )

        df = df[[text_col, label_col]].rename(
            columns={text_col: "text", label_col: "label"}
        )
    else:
        sample = DATA_DIR / "sample_emails.csv"
        df = pd.read_csv(sample)

    df["text"] = df["text"].fillna("").map(clean_text)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    label_map = {
        "safe email": 0,
        "safe": 0,
        "legitimate": 0,
        "legitimate email": 0,
        "0": 0,
        "phishing email": 1,
        "phishing": 1,
        "1": 1,
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 5]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    if df["label"].nunique() < 2:
        raise ValueError("Dataset must contain both Safe and Phishing emails.")

    return df

def build_model():
    text_features = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=25000,
    )

    security_features = EmailSecurityFeatures()

    features = FeatureUnion([
        ("tfidf", text_features),
        ("security", security_features),
    ])

    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )),
    ])

def main():
    print("\n=== Thiranex Task 3: Phishing Email Detection ===\n")

    df = load_data()
    print(f"Usable emails: {len(df)}")
    print("Class distribution:")
    print(df["label"].map({0: "Safe", 1: "Phishing"}).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    model = build_model()
    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n=== Evaluation ===")
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print(classification_report(
        y_test, predictions,
        target_names=["Safe", "Phishing"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Safe", "Phishing"]
    ).plot(ax=ax, values_format="d")
    ax.set_title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("Thiranex Task 3 - Phishing Email Detection\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset size after cleaning: {len(df)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write("Confusion Matrix [rows=true, columns=predicted]\n")
        f.write(str(cm) + "\n\n")
        f.write(classification_report(
            y_test, predictions,
            target_names=["Safe", "Phishing"],
            zero_division=0
        ))

    model_path = MODEL_DIR / "phishing_model.joblib"
    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Confusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'metrics.txt'}")
    print("\nNext command:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
