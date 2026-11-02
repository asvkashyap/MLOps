"""
Diabetes model comparison with MLflow (extends MLflow Lab 1).

One parent run holds a nested child run for every model configuration
(ElasticNet, Random Forest, Gradient Boosting, each with a small grid).
The best child by test RMSE is registered and gets the "champion" alias,
but only if it beats the current champion. That is a simple promotion gate.

Run:
    python train_diabetes.py
Then:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import itertools

import matplotlib

matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "diabetes-model-comparison"
MODEL_NAME = "diabetes-regressor"
ALIAS = "champion"
SEED = 42

# Each family: the estimator class and its grid
SEARCH_SPACE = {
    "elasticnet": (ElasticNet, {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}),
    "random_forest": (RandomForestRegressor, {"n_estimators": [100, 300], "max_depth": [4, 8]}),
    "gradient_boosting": (GradientBoostingRegressor, {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
}

# MLflow saves sklearn models with skops, which refuses to reload types that a
# malicious file could abuse. Tree models need sklearn's Tree type. We trained
# these models ourselves, so we trust that one type, and only for tree families.
TRUSTED_TYPES = {
    "random_forest": ["sklearn.tree._tree.Tree"],
    "gradient_boosting": ["sklearn.tree._tree.Tree"],
}


def grid(params: dict):
    keys = list(params)
    for values in itertools.product(*params.values()):
        yield dict(zip(keys, values))


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_predictions(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual disease progression")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def current_champion_rmse(client: MlflowClient):
    """RMSE of the model that currently holds the alias, or None."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except Exception:
        return None, None
    return float(version.tags.get("test_rmse", "inf")), version.version


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    data = load_diabetes(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    train_df = X_train.assign(target=y_train)
    dataset = mlflow.data.from_pandas(train_df, name="sklearn-diabetes-train", targets="target")

    results = []
    with mlflow.start_run(run_name="model-sweep") as parent:
        mlflow.log_input(dataset, context="training")
        mlflow.set_tags({"dataset": "sklearn diabetes", "split": "80/20, seed 42"})

        for family, (cls, space) in SEARCH_SPACE.items():
            for params in grid(space):
                run_name = f"{family}-" + "-".join(f"{k}={v}" for k, v in params.items())
                with mlflow.start_run(run_name=run_name, nested=True) as child:
                    model = cls(random_state=SEED, **params).fit(X_train, y_train)
                    metrics = evaluate(y_test, model.predict(X_test))

                    mlflow.set_tag("model_family", family)
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    extra = {}
                    if family in TRUSTED_TYPES:
                        extra["skops_trusted_types"] = TRUSTED_TYPES[family]
                    info = mlflow.sklearn.log_model(
                        model,
                        name="model",
                        signature=infer_signature(X_train, model.predict(X_train)),
                        input_example=X_test.head(3),
                        **extra,
                    )
                    results.append({"run_id": child.info.run_id, "name": run_name,
                                    "model": model, "uri": info.model_uri, **metrics})
                    print(f"{run_name:<55} rmse={metrics['rmse']:.2f}  r2={metrics['r2']:.3f}")

        best = min(results, key=lambda r: r["rmse"])
        mlflow.log_metrics({"best_rmse": best["rmse"], "best_r2": best["r2"]})
        mlflow.set_tag("best_child", best["name"])
        mlflow.log_figure(
            plot_predictions(y_test, best["model"].predict(X_test), best["name"]),
            "best_model_predictions.png",
        )

    print(f"\nBest: {best['name']}  rmse={best['rmse']:.2f}  r2={best['r2']:.3f}")

    # Register the best model, and move the alias only if it beats the champion
    version = mlflow.register_model(best["uri"], MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, version.version, "test_rmse", f"{best['rmse']:.4f}")
    client.set_model_version_tag(MODEL_NAME, version.version, "source_run", best["name"])

    old_rmse, old_version = current_champion_rmse(client)
    if old_rmse is None or best["rmse"] < old_rmse:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
        print(f"Version {version.version} is now '{ALIAS}'"
              + (f" (beat version {old_version}, rmse {old_rmse:.2f})" if old_version else ""))
    else:
        print(f"Version {version.version} registered, but '{ALIAS}' stays on version "
              f"{old_version} (rmse {old_rmse:.2f} is not worse)")


if __name__ == "__main__":
    main()
