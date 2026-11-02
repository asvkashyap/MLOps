"""
Sends test rows to the served champion model and compares its predictions
with the true values.

Start the server first (in another terminal):
    set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    mlflow models serve -m "models:/diabetes-regressor@champion" -p 5001 --env-manager local
Then:
    python predict_client.py
"""
import json
import urllib.request

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

URL = "http://127.0.0.1:5001/invocations"

data = load_diabetes(as_frame=True)
_, X_test, _, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
rows = X_test.head(5)

payload = {"dataframe_split": {"columns": list(rows.columns), "data": rows.values.tolist()}}
request = urllib.request.Request(
    URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(request) as response:
    predictions = json.loads(response.read())["predictions"]

print(f"{'actual':>8} {'predicted':>10}")
for actual, predicted in zip(y_test.head(5), predictions):
    print(f"{actual:>8.0f} {predicted:>10.1f}")
