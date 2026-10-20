"""
### Wine clustering pipeline

Extends Airflow Lab 1. Clusters 166 wines on 13 chemical measurements and
checks the clusters against the three known grape cultivars.

**Flow:** validate the data, branch (bad data stops the run), standardize,
fit KMeans for every k in parallel (dynamic task mapping), pick k by
silhouette score, refit at that k, then score and predict held-out wines.

**Params** (set when triggering):
- `k_min`, `k_max`: the range of k to try
- `min_rows`: validation fails below this. Set it above 166 to see the
  bad-data branch.

Outputs go to `working_data/<run id>/` in the Lab_1 folder.
"""
import logging
import os
import re
from datetime import datetime, timedelta

from airflow.sdk import Param, dag, get_current_context, task

from src import wine_pipeline as wp

log = logging.getLogger(__name__)
WORK_ROOT = "/opt/airflow/working_data"


def notify_failure(context):
    """Runs when a task fails for good (after its retries)."""
    ti = context["task_instance"]
    log.error(
        "FAILED: task %s in run %s after %s tries: %s",
        ti.task_id, context["run_id"], ti.try_number, context.get("exception"),
    )


default_args = {
    "owner": "Sai Kashyap",
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
    "on_failure_callback": notify_failure,
}


@dag(
    dag_id="wine_clustering",
    description="Validated, parallel KMeans model selection on the Wine dataset",
    schedule="0 6 * * *",          # daily at 06:00 UTC
    start_date=datetime(2026, 10, 1),
    catchup=False,                 # don't backfill the days since start_date
    max_active_runs=1,
    default_args=default_args,
    params={
        "k_min": Param(2, type="integer", minimum=2, description="Smallest k to try"),
        "k_max": Param(10, type="integer", minimum=3, maximum=20, description="Largest k to try"),
        "min_rows": Param(100, type="integer", minimum=1,
                          description="Validation fails below this many rows"),
    },
    tags=["lab", "clustering"],
    doc_md=__doc__,
)
def wine_clustering():

    @task
    def validate_data() -> dict:
        params = get_current_context()["params"]
        report = wp.validate(wp.TRAIN_CSV, params["min_rows"])
        log.info("Validation report: %s", report)
        return report

    @task.branch
    def check_quality(report: dict) -> str:
        return "preprocess" if report["passed"] else "report_bad_data"

    @task
    def report_bad_data(report: dict) -> None:
        log.warning("Data failed validation, skipping training: %s", report["problems"])

    @task
    def preprocess() -> dict:
        run_id = get_current_context()["run_id"]
        # Run ids contain ':' and '+', which Windows folders can't hold
        run_dir = os.path.join(WORK_ROOT, re.sub(r"[^A-Za-z0-9_.-]", "_", run_id))
        return wp.preprocess(wp.TRAIN_CSV, run_dir)

    @task(retries=0)               # a bad param won't fix itself on retry
    def k_values() -> list:
        params = get_current_context()["params"]
        if params["k_max"] <= params["k_min"]:
            raise ValueError(f"k_max ({params['k_max']}) must be greater than k_min ({params['k_min']})")
        return list(range(params["k_min"], params["k_max"] + 1))

    @task
    def fit_k(prep: dict, k: int) -> dict:
        result = wp.fit_one_k(prep["X"], k)
        log.info("k=%s  sse=%.1f  silhouette=%.3f", k, result["sse"], result["silhouette"])
        return result

    @task
    def select_k(results) -> dict:
        chosen = wp.choose_k(list(results))
        log.info("Chosen k=%s (silhouette %s), elbow suggested k=%s",
                 chosen["k"], chosen["silhouette"], chosen["elbow_k"])
        return chosen

    @task
    def train_final(prep: dict, chosen: dict) -> str:
        return wp.train_final(prep["X"], chosen["k"], prep["run_dir"])

    @task
    def evaluate(prep: dict, model_path: str, chosen: dict) -> dict:
        summary = wp.evaluate_and_predict(model_path, prep["scaler"], prep["run_dir"], chosen)
        log.info("Summary: %s", summary)
        return summary

    report = validate_data()
    branch = check_quality(report)
    prep = preprocess()
    branch >> [prep, report_bad_data(report)]

    ks = k_values()
    prep >> ks
    results = fit_k.partial(prep=prep).expand(k=ks)   # one task per k
    chosen = select_k(results)
    model_path = train_final(prep, chosen)
    evaluate(prep, model_path, chosen)


wine_clustering()
