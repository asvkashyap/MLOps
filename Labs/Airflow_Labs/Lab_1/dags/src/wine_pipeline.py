"""
Steps for the wine_clustering DAG.

Each function reads its inputs from files and writes its outputs to files,
returning only small JSON-safe values (paths and numbers). Airflow stores task
return values (XComs) in its metadata database, so keeping them small is the
recommended practice. The original lab instead passes whole pickled DataFrames
through XCom.

scikit-learn and kneed are imported inside the functions so that Airflow's
DAG parser, which re-reads the DAG file every 30 seconds, never loads them.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "wine.csv")
TEST_CSV = os.path.join(DATA_DIR, "wine_test.csv")

LABEL_COL = "cultivar"
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline",
]
SEED = 42


def validate(path: str, min_rows: int) -> dict:
    """Checks columns, row count, missing values and types. Never raises:
    the DAG's branch task decides what to do with the report."""
    df = pd.read_csv(path)
    problems = []

    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows, need at least {min_rows}")

    present = [c for c in FEATURES if c in df.columns]
    n_null = int(df[present].isna().sum().sum())
    if n_null:
        problems.append(f"{n_null} missing values in feature columns")
    non_numeric = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric feature columns: {non_numeric}")

    return {"rows": int(len(df)), "passed": not problems, "problems": problems}


def preprocess(path: str, run_dir: str) -> dict:
    """Standardizes the 13 features. Saves the scaled matrix and the fitted
    scaler, so prediction later applies exactly the same scaling."""
    from sklearn.preprocessing import StandardScaler

    os.makedirs(run_dir, exist_ok=True)
    df = pd.read_csv(path)
    scaler = StandardScaler().fit(df[FEATURES])

    x_path = os.path.join(run_dir, "X_scaled.npy")
    scaler_path = os.path.join(run_dir, "scaler.pkl")
    np.save(x_path, scaler.transform(df[FEATURES]))
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return {"run_dir": run_dir, "X": x_path, "scaler": scaler_path, "rows": int(len(df))}


def fit_one_k(x_path: str, k: int) -> dict:
    """Fits KMeans for a single k. The DAG runs one of these per k, in parallel."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.load(x_path)
    model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    return {
        "k": int(k),
        "sse": float(model.inertia_),
        "silhouette": float(silhouette_score(X, model.labels_)),
    }


def choose_k(results: list) -> dict:
    """Picks the k with the best silhouette score. The elbow is computed too
    and reported next to it, since on this data the elbow is less clear-cut."""
    from kneed import KneeLocator

    results = sorted(results, key=lambda r: r["k"])
    ks = [r["k"] for r in results]
    sse = [r["sse"] for r in results]

    elbow = None
    if len(ks) >= 3:
        elbow = KneeLocator(ks, sse, curve="convex", direction="decreasing").elbow
    best = max(results, key=lambda r: r["silhouette"])

    return {
        "k": best["k"],
        "silhouette": round(best["silhouette"], 3),
        "elbow_k": int(elbow) if elbow is not None else None,
        "scores": {r["k"]: round(r["silhouette"], 3) for r in results},
    }


def train_final(x_path: str, k: int, run_dir: str) -> str:
    """Refits KMeans at the chosen k and saves it. (The original lab saves
    whichever model came last in its loop, k=49.)"""
    from sklearn.cluster import KMeans

    X = np.load(x_path)
    model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    model_path = os.path.join(run_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path


def evaluate_and_predict(model_path: str, scaler_path: str, run_dir: str, chosen: dict) -> dict:
    """Scores the clusters against the true cultivars, then predicts the
    held-out wines using the saved scaler. (The original lab predicts on
    unscaled data with a model trained on scaled data.)"""
    from sklearn.metrics import adjusted_rand_score

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    train = pd.read_csv(TRAIN_CSV)
    train_clusters = model.predict(scaler.transform(train[FEATURES]))
    ari = adjusted_rand_score(train[LABEL_COL], train_clusters)

    test = pd.read_csv(TEST_CSV)
    test["cluster"] = model.predict(scaler.transform(test[FEATURES]))
    test.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    summary = {
        "chosen_k": chosen["k"],
        "silhouette": chosen["silhouette"],
        "elbow_k": chosen["elbow_k"],
        "ari_vs_cultivar": round(float(ari), 3),
        "test_predictions": test["cluster"].value_counts().sort_index().to_dict(),
    }
    summary["test_predictions"] = {int(k): int(v) for k, v in summary["test_predictions"].items()}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
