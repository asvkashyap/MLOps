# Airflow Lab 1: Wine Clustering Pipeline

This extends the course's Airflow Lab 1. The original DAG (`dags/airflow.py`, kept unchanged for comparison) runs four tasks in a straight line on a credit card dataset. This version adds a new DAG, `wine_clustering`, that uses more of Airflow itself: data validation with branching, dynamic task mapping for parallel model selection, runtime parameters, retries with a failure callback, and a daily schedule.

## What changed from the original lab

| Original lab | This version |
|---|---|
| 4 tasks in a straight line | Validate, branch, preprocess, parallel fit per k, select k, train, evaluate |
| Classic `PythonOperator` | TaskFlow API (`@task`, `@task.branch`) |
| Whole DataFrames pickled through XCom | Only file paths and small numbers through XCom; data stays on disk |
| No validation | `validate_data` checks columns, row count, nulls and types; `check_quality` branches to `report_bad_data` if they fail |
| A loop over k inside one task | Dynamic task mapping: one parallel `fit_k` task per k |
| Saves the last model in the loop (k=49) | Picks k by silhouette score, logs the elbow next to it, and refits at that k |
| Predicts raw test data with a model trained on scaled data | Saves the fitted scaler and applies it to the test data |
| No way to check the result | Scores clusters against the true wine cultivars (adjusted Rand index) |
| No params, retries or schedule | Runtime params (`k_min`, `k_max`, `min_rows`), 2 retries, a failure callback, daily at 06:00 UTC |
| Credit card CSV | scikit-learn Wine dataset (178 wines, 13 chemical measurements) |

Smaller fixes: the setup uses the official Airflow 3 `docker-compose.yaml` (the lab's README links the Airflow 2.5.1 file, which doesn't match the lab's Airflow 3 code), example DAGs are turned off, and `dags/.airflowignore` keeps the scheduler from scanning `src/` for DAGs.

## DAG structure

```
validate_data -> check_quality -+-> report_bad_data          (if validation fails)
                                +-> preprocess -> k_values -> fit_k [one per k] -> select_k -> train_final -> evaluate
```

## Versions

| Component | Version |
|---|---|
| Apache Airflow | 3.3.2 (`apache/airflow:3.3.2` image, CeleryExecutor) |
| Python (inside the image) | 3.13 |
| PostgreSQL / Redis | 16 / 7.2 |
| Docker / Docker Compose | 29.9.1 / v5.6.0 |
| scikit-learn, kneed | Latest at container start (installed through `_PIP_ADDITIONAL_REQUIREMENTS`) |
| Host | Windows 11, Docker Desktop with the WSL 2 backend |

`_PIP_ADDITIONAL_REQUIREMENTS` reinstalls packages every time the containers start, so their versions aren't pinned. That's fine for a lab; a production setup would build a custom image instead.

## Setup

Requires Docker Desktop with at least 4 GB of memory for Docker. Run everything from this folder (`Labs/Airflow_Labs/Lab_1`).

1. Create the folders and the `.env` file (on Windows, use UID 50000):
   ```
   mkdir logs plugins working_data
   echo AIRFLOW_UID=50000> .env
   ```
2. Initialize the database and start Airflow:
   ```
   docker compose up airflow-init
   docker compose up -d
   ```
3. Create the dataset (`dags/data/wine.csv` and `dags/data/wine_test.csv`):
   ```
   docker compose exec airflow-worker python /opt/airflow/dags/src/make_wine_data.py
   ```
4. Check that the DAG loaded (expect `No data found`):
   ```
   docker compose exec airflow-scheduler airflow dags list-import-errors
   ```
5. Open http://localhost:8080 and log in with `airflow` / `airflow`. Turn on `wine_clustering` and trigger it.
6. Stop everything with `docker compose down`.

The `FERNET_KEY` warnings Docker prints are harmless here: the key only encrypts saved connection passwords, which this lab doesn't use.

## What a successful run looks like

- **Graph view:** every task green, except `report_bad_data`, which shows as skipped because the data passed validation. `fit_k` shows `[9]`, one task for each k from 2 to 10.
- **Dataset:** `make_wine_data.py` prints `wine.csv: 166 rows, wine_test.csv: 12 rows`.
- **Run time:** about 17 seconds, compared with about 58 seconds for the original lab's DAG.
- **Outputs**, in `working_data/<run id>/`: `X_scaled.npy`, `scaler.pkl`, `model.pkl`, `predictions.csv`, and `summary.json`.

`summary.json` from a run with the default params:

```json
{
  "chosen_k": 3,
  "silhouette": 0.284,
  "elbow_k": 4,
  "ari_vs_cultivar": 0.89,
  "test_predictions": {"0": 4, "1": 5, "2": 3}
}
```

The chosen k matches the three real grape cultivars, and an adjusted Rand index of 0.89 means the clusters line up closely with them. The cultivar label is used only for this score, never as a feature. The elbow method suggests k=4 on the same data, which is why the DAG reports both rather than relying on the elbow alone, as the original lab does.

For comparison, the original `Airflow_Lab1` DAG runs its 4 tasks in about 58 seconds, and its elbow method picks k=8 on data with no ground truth to check that choice against.

## Trying the other paths

- **Bad data:** trigger with `min_rows` set above 166 (for example 200). `report_bad_data` runs, logs `only 166 rows, need at least 200`, and every training task is skipped.
- **Different k range:** change `k_min` and `k_max` at trigger time. The number of parallel `fit_k` tasks changes to match, with no code changes.
- **Invalid params:** setting `k_max` at or below `k_min` fails `k_values` immediately. That task has retries turned off, since retrying a bad parameter can't help.

## Files

| Path | Purpose |
|---|---|
| `dags/wine_clustering_dag.py` | The new DAG |
| `dags/src/wine_pipeline.py` | The ML steps, called by the DAG's tasks |
| `dags/src/make_wine_data.py` | Creates the dataset from scikit-learn |
| `dags/airflow.py`, `dags/src/lab.py` | The original lab, unchanged |
| `dags/.airflowignore` | Excludes `src/` from DAG scanning |
| `docker-compose.yaml` | Official Airflow 3.3.2 file, with examples off, extra packages, and the `working_data` mount |
