# AutoML Studio

AutoML Studio is a full-stack tabular machine-learning workbench. Upload a CSV, select a target, train and compare curated models, inspect experiment results, and serve predictions through a REST API or browser UI.

> **Status:** Production-oriented application, not yet production-hardened. It is suitable for local development, demonstrations, and controlled internal use. Review [Production readiness](#production-readiness) before exposing it to users or sensitive data.

## Highlights

- CSV upload with schema metadata, target suggestions, and task-type inference
- Classification and regression workflows for tabular data
- Dataset checks for row count, target integrity, null-heavy columns, duplicate rows, and zero-variance numeric features
- Fitted scikit-learn preprocessing pipeline with imputation, scaling, and one-hot encoding
- Optuna tuning for Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM
- MLflow tracking, Sweetviz profiles, evaluation metrics, leaderboards, and versioned local artifacts
- FastAPI endpoints for upload, asynchronous training, experiment status, results, feature schemas, and inference
- React/Vite dashboard plus a Streamlit alternative for local use

## Architecture

```text
CSV upload / configured local file
              |
              v
Data ingestion + Sweetviz profile
              |
              v
Validation and data-quality checks
              |
              v
Fitted preprocessing pipeline
              |
              v
Optuna-tuned model training
              |
              v
Evaluation, MLflow tracking, and artifacts
              |
              +--> React dashboard / Streamlit dashboard
              +--> FastAPI prediction endpoint
```

## Technology

| Area | Implementation |
| --- | --- |
| API | FastAPI, Pydantic, Uvicorn |
| Web application | React 18, Vite, Tailwind CSS, React Query, Zustand |
| ML | scikit-learn, XGBoost, LightGBM |
| Optimisation | Optuna |
| Tracking | MLflow |
| Profiling | Sweetviz |
| Persistence | joblib and local filesystem |

## Repository layout

```text
api/                         FastAPI application, validation, job store, inference
automl-studio-frontend/      React dashboard
configs/config.yaml          Pipeline defaults
dashboard/app.py             Streamlit dashboard
src/
  data_ingestion/            CSV loading, task detection, profiling
  data_validation/           Data-quality gate
  preprocessing/             Fitted feature transformations
  model_engine/              Model registry and Optuna training
  evaluation/                Metrics and leaderboard generation
  pipeline/                  End-to-end training orchestrator
  utils/                     Configuration, logging, and artifact helpers
data/raw/                    Uploaded and example datasets
artifacts/                   Per-experiment models, metrics, reports, leaderboards
logs/                        Application and pipeline logs
```

## Prerequisites

- Python 3.10 or later
- Node.js 18 or later and npm 9 or later for the React dashboard
- A CSV with at least 100 rows and a target column containing at least two distinct values

Python packages are pinned in [requirements.txt](requirements.txt). The frontend lockfile is committed at `automl-studio-frontend/package-lock.json`.

## Quick start

Clone the repository and create an isolated environment.

```bash
git clone https://github.com/tanishqqq-hub/AutoML.git
cd AutoML
python -m venv .venv
```

Activate it and install backend dependencies.

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API from the repository root.

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Interactive API documentation: <http://127.0.0.1:8000/docs>

```bash
curl http://127.0.0.1:8000/health
```

In a second terminal, start the React application.

```bash
cd automl-studio-frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. The dashboard currently expects the API at `http://localhost:8000`, configured in `automl-studio-frontend/src/api/client.js`.

## Training workflows

### Web workflow

1. Upload a `.csv` file (maximum 50 MB).
2. Review the schema and choose the target column. Target suggestions are heuristic; verify the selection.
3. Set a 5%–50% test split and 1–50 Optuna trials per model.
4. Start training and monitor the progress/log feed.
5. Review the leaderboard, then submit inference records from the Predict page.

### Command-line workflow

Set `data.dataset_path`, `data.target_column`, and other defaults in [configs/config.yaml](configs/config.yaml), then run:

```bash
python -m src.pipeline.train_pipeline
```

### Streamlit workflow

```bash
streamlit run dashboard/app.py
```

Open <http://localhost:8501>.

## API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness response |
| `POST` | `/upload` | Upload and inspect a CSV |
| `POST` | `/experiments/start` | Start background training |
| `GET` | `/experiments/{id}/status` | Read status, estimated progress, and logs |
| `GET` | `/experiments/{id}/results` | Fetch metrics and leaderboard |
| `GET` | `/experiments/{id}/features` | Get the feature schema required for inference |
| `POST` | `/experiments/{id}/predict` | Predict one record with the experiment’s best model |

Use the `file_path` returned by `/upload` when starting an experiment. Do not construct server-side paths in clients.

```bash
# Upload
curl -F "file=@data/raw/Housing.csv" http://127.0.0.1:8000/upload

# Use the file_path from the response and a verified target column
curl -X POST http://127.0.0.1:8000/experiments/start \
  -H "Content-Type: application/json" \
  -d '{"file_path":"data/raw/Housing.csv","target_column":"price","test_size":0.2,"n_trials":5}'

# After status is complete, use the exact schema returned by /features
curl -X POST http://127.0.0.1:8000/experiments/EXPERIMENT_ID/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"area":7420,"bedrooms":4,"bathrooms":2,"stories":3}}'
```

Prediction fields are dataset-specific. Fetch `/experiments/{id}/features` after training and send every expected feature.

## Configuration and model behaviour

[configs/config.yaml](configs/config.yaml) controls paths, train/test split, random seed, task detection, validation thresholds, preprocessing, enabled models, Optuna trials, MLflow, and logging. The API makes an isolated copy for each run and overrides dataset path, target, test split, and trial count from the request.

With `task_type: auto`, a target having 20 or fewer unique values is classified as a classification task; otherwise it is treated as regression. Override this setting when the heuristic does not match the domain.

Classification reports accuracy, weighted precision, weighted recall, weighted F1, and ROC-AUC where probabilities are available. Regression reports RMSE, MAE, and R². The deployed `best_model.pkl` is chosen by Optuna’s training-time score. The results also expose the test-set winner and flag a discrepancy; define a formal model-selection policy before operational use.

## Outputs and experiment tracking

Each run uses a timestamp ID (`YYYYMMDD_HHMMSS`) and produces:

```text
artifacts/
  models/<experiment-id>/
    preprocessing_pipeline.pkl
    best_model.pkl
    <individual-model>.pkl
  metrics/<experiment-id>/metrics.json
  experiments/<experiment-id>/
    leaderboard.csv
    profile_report.html
```

MLflow defaults to local tracking. Browse it with:

```bash
mlflow ui
```

Then open <http://127.0.0.1:5000>. Use a remote MLflow backend and artifact store for shared or durable environments.

## Testing and quality checks

The current `test_*.py` files are executable pipeline smoke tests, not an isolated unit-test suite. They use `configs/config.yaml`; training-oriented tests may take time and create artifacts.

```bash
python test_ingestor.py
python test_validation.py
python test_preprocessor.py
python test_trainer.py
python test_evaluator.py
```

Build the frontend before release:

```bash
cd automl-studio-frontend
npm ci
npm run build
```

## Production readiness

This repository has application foundations but does **not** yet meet a typical internet-facing production baseline:

- **Identity and access:** no authentication, authorisation, tenancy, or API access control.
- **CORS:** the API currently allows every origin; use a strict production allowlist.
- **Jobs:** training runs in daemon threads and experiment state is in memory. API restart loses state and multiple workers cannot coordinate jobs. Adopt a durable queue and database-backed job store.
- **Storage:** uploads, artifacts, logs, and default MLflow state are local. Use managed object storage, a remote MLflow backend, backups, and retention policies.
- **Deployment:** no Dockerfile, orchestration manifests, reverse proxy, TLS configuration, or readiness policy is currently tracked.
- **Observability:** add structured logging, metrics, tracing, alerts, audit trails, and operational dashboards.
- **Security:** add upload scanning, rate limiting, proxy-level request limits, secret management, dependency scanning, least-privilege execution, and data protection controls.
- **ML governance:** add lineage, approval gates, registry promotion, drift and performance monitoring, rollback, and retraining policies.
- **Assurance:** expand smoke tests into deterministic unit, API integration, frontend, security, and load testing in CI.

Do not upload personal, confidential, regulated, or otherwise sensitive data until these controls and your organisation's data-governance requirements are satisfied.

## Contributing

Keep changes modular and configuration-driven. Preserve the fitted preprocessing artifact with each trained model so inference uses the exact training transformations. Verify relevant smoke tests, run a frontend build for UI changes, and update this README when public behaviour or API contracts change.

## License

Released under the [MIT License](LICENSE).
