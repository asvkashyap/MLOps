"""
Trains a classifier on the Wine dataset and saves the model, scaler, and
metadata to /app/model/ so the serving stage can load them.

Run inside the trainer build stage, or locally:
    python train_wine.py
"""
import json
import os
import time

import joblib
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "200"))
SEED = 42


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED)
    model.fit(X_train_scaled, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    joblib.dump(model, os.path.join(MODEL_DIR, "wine_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    metadata = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_estimators": N_ESTIMATORS,
        "test_accuracy": round(float(accuracy), 4),
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Trained on {metadata['train_rows']} rows, test accuracy {accuracy:.4f}")
    print(f"Saved model, scaler, and metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()
