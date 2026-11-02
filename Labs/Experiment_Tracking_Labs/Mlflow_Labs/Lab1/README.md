# MLflow Lab 1: Diabetes Model Comparison

This extends the course's MLflow Lab 1. The original script (`linear_regression.py`, kept unchanged) trains one ElasticNet model on red wine quality and logs its parameters and metrics. This version, `train_diabetes.py`, uses more of MLflow itself: a named experiment, nested runs for a sweep across three model families, dataset and plot logging, the Model Registry with an alias, a promotion gate, and REST serving of the registered model.

## What changed from the original lab

| Original lab | This version |
|---|---|
| One run in the default experiment | Named experiment `diabetes-model-comparison`, with one parent run and 17 nested child runs |
| One model type, hyperparameters from the command line | Grid over three families: ElasticNet (9 configs), Random Forest (4), Gradient Boosting (4) |
| Params and metrics only | Also tags (`model_family`), the training dataset (`mlflow.log_input`), and a predicted vs actual plot for the best model |
| Registers every run's model | Registers only the best run by test RMSE |
| No model selection | Gives the best version the `champion` alias, but only if it beats the current champion (promotion gate) |
| `serving.py` demonstrates pip requirements; its temp-file pattern fails on Windows | Serves `models:/diabetes-regressor@champion` as a REST API; `predict_client.py` calls it |
| Red wine CSV downloaded at runtime | scikit-learn's built-in Diabetes dataset (442 patients, 10 features, no download) |
| Deprecated `artifact_path` argument | Current `name` argument |

Aliases are used instead of registry stages, which MLflow has deprecated.

## Versions

| Component | Version |
|---|---|
| Python | 3.13.5 |
| MLflow | 3.17.2 (SQLite backend store, `mlflow.db`) |
| scikit-learn | 1.9.1 |
| skops | 0.16.0 |
| pandas / numpy / matplotlib | 3.1.0 / 2.5.4 / 3.11.2 |
| OS | Windows 11, Command Prompt |

Pinned in `requirements-diabetes.txt`.

## Setup

From this folder (`Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1`), in Command Prompt:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-diabetes.txt
```

## Running it

1. **Train and register:**
   ```
   python train_diabetes.py
   ```
2. **Browse the results:**
   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   Open http://localhost:5000. Stop with Ctrl+C.
3. **Serve the champion** (keep this terminal open):
   ```
   set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
   mlflow models serve -m "models:/diabetes-regressor@champion" -p 5001 --env-manager local
   ```
4. **Call it** from a second terminal (with the venv activated):
   ```
   python predict_client.py
   ```

## What a successful run looks like

**Training** prints one line per configuration, then the winner:

```
gradient_boosting-learning_rate=0.05-max_depth=2        rmse=52.37  r2=0.482
...
Best: gradient_boosting-learning_rate=0.05-max_depth=2  rmse=52.37  r2=0.482
Successfully registered model 'diabetes-regressor'.
Created version '1' of model 'diabetes-regressor'.
Version 1 is now 'champion'
```

Best result per family on the 20% test split:

| Family | Best config | RMSE | R2 |
|---|---|---|---|
| Gradient Boosting | learning_rate 0.05, max_depth 2 | 52.37 | 0.482 |
| Random Forest | 300 trees, max_depth 4 | 53.06 | 0.469 |
| ElasticNet | alpha 0.01, l1_ratio 0.8 | 54.61 | 0.437 |

**Running training again** registers a new version, but the alias stays put, since the same data and seed give the same RMSE. From this lab's third run:

```
Created version '3' of model 'diabetes-regressor'.
Version 3 registered, but 'champion' stays on version 1 (rmse 52.37 is not worse)
```

**In the MLflow UI:** the experiment shows a `model-sweep` parent run that expands into 17 child runs, which can be sorted or compared by RMSE. The parent run holds the dataset, the `best_child` tag, and `best_model_predictions.png`. Under Models, `diabetes-regressor` version 1 carries the `@champion` alias.

**Serving:** `predict_client.py` sends 5 test patients to the REST endpoint and prints:

```
  actual  predicted
     219      162.0
      70      178.9
     202      171.6
     230      264.0
     111      116.3
```

The target is a disease progression score one year after baseline. An R2 of about 0.48 is typical for this dataset, so individual predictions are rough.

## A note on tree models and skops

MLflow 3 saves scikit-learn models with skops, which refuses to reload object types that a malicious file could abuse. Tree-based models use one such type (`sklearn.tree._tree.Tree`), so saving them failed until it was explicitly trusted. `train_diabetes.py` trusts that single type, and only for the two tree families, since these models are trained locally rather than loaded from an outside source.

## Files

| File | Purpose |
|---|---|
| `train_diabetes.py` | Sweep, logging, registration and promotion gate |
| `predict_client.py` | Calls the served model |
| `requirements-diabetes.txt` | Pinned versions |
| `linear_regression.py`, `serving.py`, notebooks | The original lab, unchanged |
