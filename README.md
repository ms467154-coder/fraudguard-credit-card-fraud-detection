# FraudGuard — Credit Card Fraud Detection

FraudGuard is a production-oriented credit-card fraud detection application built around an existing machine-learning pipeline. It combines trained scikit-learn and XGBoost artifacts with a validated FastAPI inference layer and a responsive React dashboard.

> **Product direction:** A clean risk-operations workspace for reviewing anonymized transaction feature vectors across multiple trained models.

## Architecture

```text
React + Vite dashboard
        ↓
FastAPI REST API
        ↓
Model registry and inference layer
        ↓
Logistic Regression · Random Forest · XGBoost
        ↓
Existing notebook pipeline and local dataset
```

The original training notebook, feature selection, BorderlineSMOTE workflow, model artifacts, and model configuration are preserved. The new application does not retrain or rewrite the ML pipeline.

## Features

The React dashboard provides a branded cream/off-white and muted-red interface, responsive layouts for desktop and mobile, model registry status, feature-vector input, configurable decision threshold, loading and error states, ensemble fraud probability, per-model predictions, and an accessible risk interpretation.

The FastAPI backend provides validated request schemas, safe error responses, model loading, health status, model registry metadata, OpenAPI documentation, configurable threshold and CORS settings, and a clean separation between UI and model implementation details.

## Project Structure

```text
.
├── api.py                    # FastAPI application and inference endpoints
├── frontend/                 # React + Vite source application
│   ├── index.html
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       └── styles.css
├── static/                   # Production React build served by FastAPI
├── models/
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│   └── model_config.json
├── tests/test_api.py         # Focused API contract tests
├── creditcard.ipynb          # Original ML training and evaluation pipeline
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
├── .env.example              # Runtime configuration template
└── README.md
```

## Local Setup

Install the Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

Install and build the React frontend:

```bash
cd frontend
npm install
npm run build
cd ..
```

Start the full application through FastAPI:

```bash
uvicorn api:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the dashboard. The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

During frontend development, run Vite separately from `frontend/`:

```bash
npm run dev
```

The development server expects the FastAPI service to be available on the same host for API requests.

## API Contract

### Health

```http
GET /health
```

### Model registry

```http
GET /api/models
```

### Prediction

```http
POST /api/predict
Content-Type: application/json
```

Example request:

```json
{
  "features": {
    "V17": 0.0,
    "V14": 0.0,
    "V12": 0.0,
    "V10": 0.0,
    "V16": 0.0,
    "V3": 0.0,
    "V7": 0.0,
    "V11": 0.0,
    "V4": 0.0,
    "V18": 0.0,
    "V1": 0.0,
    "V9": 0.0,
    "V5": 0.0,
    "V2": 0.0,
    "V6": 0.0,
    "V21": 0.0,
    "V19": 0.0,
    "V20": 0.0,
    "V8": 0.0,
    "V27": 0.0
  },
  "threshold": 0.5
}
```

The response includes each model's predicted class, fraud probability, threshold, and interpretation.

## Testing

Run the focused API tests from the project root:

```bash
python -m pytest tests/test_api.py -q
```

The production smoke check should report three loaded model artifacts through `/health` and `/api/models`.

## Configuration

Copy `.env.example` to `.env` when local configuration is needed. Supported settings are `LOG_LEVEL`, `FRAUD_THRESHOLD`, and `CORS_ORIGINS`.

## Docker

```bash
docker build -t fraudguard .
docker run --rm -p 8000:8000 fraudguard
```

## Data and Model Notes

The original dataset contains anonymized PCA-derived transaction features and is used by the notebook training workflow. The local `creditcard.csv` file is intentionally excluded from GitHub because it is a large raw data artifact. Place the dataset in the project root before running the notebook.

The trained models use the 20 features defined in `models/model_config.json`. Fraud detection is evaluated with metrics appropriate for imbalanced classification, including precision, recall, F1 score, ROC-AUC, and confusion matrices. Accuracy should not be treated as the primary success measure for this problem.

## React Migration

The legacy Streamlit application has been removed. The supported user interface is now the React/Vite dashboard in `frontend/`, with its production build emitted to `static/` and served by FastAPI.

## Author

Built by **Mohamed Salem** across AI Engineering, Machine Learning, MLOps, and LLM Engineering.
