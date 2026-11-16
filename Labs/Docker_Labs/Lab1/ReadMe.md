# Docker Lab 1: Wine Classifier, Multi-Stage Build and Serving

This extends the course's Docker Lab 1. The original (`dockerfile`, `src/main.py`, kept unchanged) trains a RandomForest on Iris inside a single-stage image and exits; the trained model is discarded unless you manually `docker cp` it out, and nothing serves it. This version, `Dockerfile.wine`, trains a model in one build stage and serves it from a second, slimmer stage, with a healthcheck and a non-root user.

## What changed from the original lab

| Original lab | This version |
|---|---|
| Single stage, `FROM python:3.10` (full image) | Multi-stage: `trainer` stage trains the model, `final` stage serves it and never sees the build tools |
| Trains Iris, prints a message, exits; model is lost when the container is removed | Trains Wine (178 samples, 13 features), saves model + scaler + metadata, and serves predictions over HTTP |
| No serving | Flask app: `/health`, `/metadata`, `/predict` |
| `docker run lab1:v1` with no port mapping (nothing to reach anyway) | `docker run -p 8000:8000` with a working `/predict` endpoint |
| Unpinned `requirements.txt` | Pinned versions in `requirements-wine.txt` |
| Runs as root, no healthcheck | Non-root `app` user, `HEALTHCHECK` instruction |
| No `.dockerignore` | `.dockerignore` keeps the local venv and old tar files out of the build context |

## Versions

| Component | Version |
|---|---|
| Docker | 29.9.1 |
| Base image (this version) | `python:3.11-slim` |
| Base image (original lab) | `python:3.10` (full) |
| scikit-learn | 1.5.2 |
| Flask | 3.0.3 |
| joblib | 1.4.2 |
| numpy | 2.1.3 |
| Host | Windows 11, Docker Desktop (WSL 2 backend) |

## Setup

From this folder (`Labs/Docker_Labs/Lab1`):

```
docker build -f Dockerfile.wine -t wine-classifier:v1 .
docker run -d -p 8000:8000 --name wine-api wine-classifier:v1
```

Stop and remove the container with `docker rm -f wine-api`.

## What a successful run looks like

**Build:** two stages run in order. `[trainer 6/6] RUN python train_wine.py` prints `Trained on 142 rows, test accuracy 1.0000` during the build itself, not at container start.

**Image size**, compared with the original lab's single-stage build:

| Image | Disk usage | Content size |
|---|---|---|
| Original (`lab1-original`, single stage, `python:3.10`) | 2.07 GB | 551 MB |
| This version (`wine-classifier:v1`, multi-stage, `python:3.11-slim`) | 583 MB | 136 MB |

About a 72% smaller disk footprint, from dropping the training stage's build tools and pip cache from the final image, and starting from the slim base.

**Running container:**
```
CONTAINER ID   IMAGE                COMMAND             STATUS                   PORTS
dba782b679e8   wine-classifier:v1   "python serve.py"   Up (healthy)             0.0.0.0:8000->8000/tcp
```

**Health check:**
```
curl http://localhost:8000/health
{"model_trained_at":"2026-11-16T03:58:20Z","status":"ok"}
```

**Prediction**, for one real Wine sample:
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" ^
  -d "{\"features\": [13.2, 1.78, 2.14, 11.2, 100, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]}"

{"predicted_class":"class_0","probabilities":{"class_0":0.98,"class_1":0.01,"class_2":0.01}}
```

**Metadata**, confirming the trained model's stats:
```
curl http://localhost:8000/metadata

{"class_names":["class_0","class_1","class_2"],
 "feature_names":["alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
                   "total_phenols","flavanoids","nonflavanoid_phenols",
                   "proanthocyanins","color_intensity","hue",
                   "od280/od315_of_diluted_wines","proline"],
 "n_estimators":200,"test_accuracy":1.0,"test_rows":36,"train_rows":142,
 "trained_at":"2026-11-16T03:58:20Z"}
```

100% test accuracy is expected here: Wine's three classes are easy to separate on quality features. `test_accuracy` is logged specifically so a later retrain with different settings has something concrete to compare against.

## Trying the training knob

`N_ESTIMATORS` is read from the environment during the build's training step. To retrain with a different forest size and rebuild:
```
docker build -f Dockerfile.wine -t wine-classifier:v2 --build-arg N_ESTIMATORS=50 .
```
(Note: this requires adding an `ARG N_ESTIMATORS` line to the Dockerfile's trainer stage and passing it through as an `ENV`, since `--build-arg` and the script's `os.environ.get` aren't connected automatically. Left as a possible follow-up.)

## Files

| Path | Purpose |
|---|---|
| `Dockerfile.wine` | Multi-stage build: train, then serve |
| `src/train_wine.py` | Trains the Wine model, saves model, scaler, and metadata |
| `src/serve.py` | Flask serving app |
| `src/requirements-wine.txt` | Pinned dependencies for both stages |
| `.dockerignore` | Excludes the local venv and old image tarballs from the build context |
| `dockerfile`, `src/main.py`, `ReadMe.md` | The original lab, unchanged |
