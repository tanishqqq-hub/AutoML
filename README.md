# 🤖 AutoML Studio

A production-grade end-to-end automated machine learning platform that automatically trains, evaluates, and deploys machine learning models from raw datasets.

---

## 🚀 What It Does

Upload any CSV dataset and AutoML Studio will automatically:

- ✅ Profile and validate the dataset
- ✅ Handle missing values, encoding, and scaling
- ✅ Train 5 ML models simultaneously
- ✅ Optimize hyperparameters using Optuna
- ✅ Track all experiments using MLflow
- ✅ Generate a model leaderboard
- ✅ Deploy the best model via FastAPI
- ✅ Provide a Streamlit UI for end users

---

## 🏗️ System Architecture

```
Dataset Upload
      ↓
Data Ingestion (Sweetviz profiling)
      ↓
Data Validation (null checks, duplicates, schema)
      ↓
Preprocessing Pipeline (sklearn ColumnTransformer)
      ↓
Model Engine (5 models × Optuna optimization)
      ↓
Evaluation (metrics + leaderboard)
      ↓
FastAPI Inference Service
      ↓
Streamlit Dashboard
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML Models | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna |
| Experiment Tracking | MLflow |
| Preprocessing | scikit-learn Pipeline + ColumnTransformer |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit |
| Config | YAML |
| Serialization | joblib |

---

## 📁 Project Structure

```
automl-studio/
│
├── configs/
│   └── config.yaml           # All pipeline parameters
│
├── src/
│   ├── utils/
│   │   ├── common.py         # Path and timestamp utilities
│   │   ├── config_loader.py  # YAML config loader
│   │   ├── logger.py         # Centralized logging
│   │   └── artifact_manager.py # Model and metric persistence
│   │
│   ├── data_ingestion/
│   │   └── ingestor.py       # Dataset loading and profiling
│   │
│   ├── data_validation/
│   │   └── validator.py      # Data quality checks
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py   # Feature engineering pipeline
│   │
│   ├── model_engine/
│   │   └── trainer.py        # Model training + Optuna
│   │
│   ├── evaluation/
│   │   └── evaluator.py      # Metrics + leaderboard
│   │
│   └── pipeline/
│       └── train_pipeline.py # Main orchestrator
│
├── api/
│   ├── app.py                # FastAPI server
│   ├── schemas.py            # Request/response schemas
│   └── predictor.py          # Inference engine
│
├── dashboard/
│   └── app.py                # Streamlit UI
│
├── data/
│   ├── raw/                  # Upload datasets here
│   └── processed/
│
├── artifacts/
│   ├── models/               # Trained model files
│   ├── metrics/              # Evaluation metrics
│   └── experiments/          # Leaderboards and reports
│
├── logs/                     # Pipeline logs
├── requirements.txt
└── packages.txt
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/tanishqqq-hub/AutoML.git
cd AutoML
```

**2. Create virtual environment**
```bash
python -m venv venv
```

**3. Activate environment**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Pipeline

**Option 1 — Streamlit Dashboard (Recommended)**
```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` and:
1. Upload any CSV dataset
2. Select target column
3. Configure settings
4. Click Train
5. View leaderboard
6. Make predictions

---

**Option 2 — Command Line**

Update `configs/config.yaml` with your dataset path and target column, then:

```bash
python src/pipeline/train_pipeline.py
```

---

**Option 3 — FastAPI**

After training, start the API:
```bash
# Set experiment ID from training output
$env:EXPERIMENT_ID = "YOUR_EXPERIMENT_ID"

uvicorn api.app:app --reload
```

API available at `http://localhost:8000`

Swagger UI at `http://localhost:8000/docs`

**Example prediction request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Pclass": 1,
      "Sex": "female",
      "Age": 29,
      "Fare": 72.5
    }
  }'
```

---

## 📊 Experiment Tracking

Start MLflow UI to visualize all experiments:
```bash
mlflow ui
```

Open `http://localhost:5000`

---

## ⚙️ Configuration

All pipeline parameters are controlled via `configs/config.yaml`:

```yaml
data:
  dataset_path: data/raw/dataset.csv
  target_column: Survived
  test_size: 0.2
  task_type: auto              # auto, classification, regression

models:
  n_trials: 20                 # Optuna trials per model
  enabled:
    - logistic_regression
    - random_forest
    - gradient_boosting
    - xgboost
    - lightgbm

evaluation:
  primary_metric: roc_auc
```

---

## 📈 Supported Models

| Model | Task |
|---|---|
| Logistic Regression | Classification |
| Random Forest | Classification + Regression |
| Gradient Boosting | Classification + Regression |
| XGBoost | Classification + Regression |
| LightGBM | Classification + Regression |

---

## 📐 Evaluation Metrics

**Classification:**
- Accuracy, Precision, Recall, F1, ROC-AUC

**Regression:**
- RMSE, MAE, R²

---

## 🔧 Design Principles

- **Modular Architecture** — each pipeline stage is independent
- **Config-Driven** — no hardcoded values anywhere
- **Zero Data Leakage** — preprocessing fitted on training data only
- **Reproducible** — fixed random seeds + MLflow tracking
- **Production-Ready** — proper logging, error handling, artifact versioning

---

## 📋 Pipeline Results (Titanic Dataset)

```
Best Model  : Logistic Regression
ROC-AUC     : 0.8608
Accuracy    : 0.8101
F1 Score    : 0.8059
Duration    : ~14 seconds
```

---

## 🗺️ Roadmap

- [ ] Docker deployment
- [ ] Data drift monitoring
- [ ] Automatic retraining
- [ ] React enterprise frontend
- [ ] Multi-user support
- [ ] Database backend

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.