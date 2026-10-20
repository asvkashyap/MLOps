"""
Creates the lab's dataset from scikit-learn's built-in Wine data.

  data/wine.csv       166 wines: 13 chemical measurements plus the true
                      cultivar (used only to score the clusters, never as a feature)
  data/wine_test.csv  12 held-out wines, measurements only

Run once from Lab_1:
    docker compose exec airflow-worker python /opt/airflow/dags/src/make_wine_data.py
"""
import os


def main():
    from sklearn.datasets import load_wine

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    df = load_wine(as_frame=True).frame.rename(columns={"target": "cultivar"})
    df.columns = [c.replace("/", "_") for c in df.columns]

    test = df.sample(n=12, random_state=7)
    train = df.drop(test.index)

    train.to_csv(os.path.join(data_dir, "wine.csv"), index=False)
    test.drop(columns="cultivar").to_csv(os.path.join(data_dir, "wine_test.csv"), index=False)
    print(f"wine.csv: {len(train)} rows, wine_test.csv: {len(test)} rows")


if __name__ == "__main__":
    main()
