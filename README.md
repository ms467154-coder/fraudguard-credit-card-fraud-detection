# Credit Card Fraud Detection System

A comprehensive machine learning pipeline for detecting fraudulent credit card transactions with multiple models, interactive prediction tools, and deployment options.

---

## 📋 Project Overview

This project builds an **end-to-end fraud detection system** that:
- Handles severely imbalanced data (99.8% legitimate, 0.2% fraud)
- Trains multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Applies SMOTE for balanced training
- Provides interactive prediction interfaces
- Offers batch evaluation and API deployment options

---

## 📁 Project Structure

```
credit card fraud detection/
├── creditcard.ipynb                    # Main Jupyter notebook - full pipeline
├── creditcard.csv                      # Dataset (285,000+ transactions)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── api.py                              # FastAPI service for real-time predictions
├── evaluate_on_new_data.py             # CLI script for batch evaluation
├── evaluate_streamlit.py                # Streamlit web app for predictions
│
├── models/                             # Trained model files
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   ├── XGBoost.pkl
│   └── model_config.json              # Feature configuration for each model
│
├── model_results_template.md           # Template for results documentation
└── README_EVALUATION.md                # Details on the evaluation script
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

Explore the full pipeline from data preprocessing to model evaluation:

```bash
jupyter notebook creditcard.ipynb
```

**Key phases:**
- **Phase 1:** Data Exploration & Understanding
- **Phase 2:** Data Preprocessing & Feature Engineering
- **Phase 3:** Handle Class Imbalance (SMOTE)
- **Phase 4:** Train Multiple Models
- **Phase 5:** Evaluate & Compare Models
- **Phase 6:** Deployment & Next Steps

### 3. Make Predictions (Choose One)

#### Option A: Interactive Streamlit App ⭐ (Recommended)

```bash
streamlit run evaluate_streamlit.py
```

- Upload CSV with transactions
- Select features for each model
- View predictions + fraud probabilities
- Download results

#### Option B: FastAPI Service

```bash
uvicorn api:app --reload
```

Access at `http://localhost:8000` and POST to `/predict` with transaction features.

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"V1": 1.0, "V2": -0.5, ..., "Amount": 100, "Time": 45000}'
```

#### Option C: Batch Evaluation Script

```bash
python evaluate_on_new_data.py \
  --input new_transactions.csv \
  --columns V1,V2,V3,Amount,Time \
  --target Class \
  --models-dir ./models \
  --output results.json
```

See `README_EVALUATION.md` for full usage details.

---

## 📊 Dataset

- **Source:** Kaggle Credit Card Fraud Detection Dataset
- **Size:** 284,807 transactions
- **Features:** 28 PCA-transformed features (V1-V28) + Amount + Time
- **Target:** Class (0 = Legitimate, 1 = Fraud)
- **Imbalance:** 99.827% legitimate, 0.173% fraud

---

## 🤖 Models

All models are trained on the **same 20 features** (top correlated with fraud):

```
V17, V14, V12, V10, V16, V3, V7, V11, V4, V18, 
V1, V9, V5, V2, V6, V21, V19, V20, V8, V27
```

### Model Configurations

1. **Logistic Regression**
   - Fast baseline model
   - Interpretable coefficients
   - Expected F1: ~0.75

2. **Random Forest**
   - 200 estimators, max_depth=15
   - Feature importance analysis
   - Expected F1: ~0.80

3. **XGBoost**
   - 300 estimators, max_depth=6
   - Best performance
   - Expected F1: ~0.82

---

## 🛠 Training Pipeline

### Data Preprocessing
- StandardScaler normalization for Amount and Time
- Feature correlation analysis
- Top 20 features selected by correlation with fraud

### Class Imbalance Handling
- **SMOTE** (BorderlineSMOTE) applied to training data
- Balanced synthetic fraud samples generated
- Test set remains imbalanced (realistic distribution)

### Train-Test Split
- **Stratified split:** 80% train, 20% test
- Maintains class distribution in both sets
- Random state: 42 (reproducible)

### Model Training
- All models trained on resampled (balanced) training data
- Evaluated on original (imbalanced) test set
- 5-fold stratified cross-validation for robustness

---

## 📈 Evaluation Metrics

For imbalanced classification, the project uses:

- **Precision:** Of predicted fraud, how many are actual fraud?
- **Recall:** Of actual fraud, how many did we catch?
- **F1 Score:** Harmonic mean (primary metric for imbalanced data)
- **ROC-AUC:** Probability-based metric
- **Confusion Matrix:** TP, TN, FP, FN breakdown

*Note: Accuracy is NOT used (98%+ accuracy possible by predicting "No Fraud" always)*

---

## 🧪 Testing & Validation

The `evaluate_streamlit.py` app allows you to:

✅ Upload any CSV with transaction data
✅ Auto-select features per model (with customization)
✅ See predictions + fraud probabilities
✅ View prediction distribution (fraud vs. legitimate)
✅ Download results as CSV

---

## 🔑 Key Features

### Model-Specific Configuration
Each model's feature set is defined in `models/model_config.json`:
```json
{
  "ModelName": {
    "features": ["V1", "V2", ...],
    "description": "Model description"
  }
}
```

### Probability Predictions
- All models return fraud probability (0-1)
- Threshold can be adjusted for precision/recall trade-off
- Default threshold: 0.5

### Batch Processing
- `evaluate_on_new_data.py` processes entire CSV files
- Computes metrics if ground truth available
- Outputs detailed JSON report

---

## 📦 Dependencies

See `requirements.txt` for full list. Key packages:

- **pandas, numpy:** Data manipulation
- **scikit-learn:** ML algorithms, metrics
- **xgboost:** Gradient boosting
- **imbalanced-learn:** SMOTE oversampling
- **joblib:** Model serialization
- **matplotlib, seaborn:** Visualization
- **streamlit:** Interactive web app
- **fastapi, uvicorn:** REST API
- **jupyter:** Notebook environment

---

## 🚨 Important Notes

### Data Privacy
- Original dataset is anonymized (PCA features)
- Time and Amount are the only interpretable features
- No personally identifiable information

### Model Limitations
- Trained on fixed feature set (V1-V27)
- Requires exact features for prediction
- Missing features will cause errors
- Time series patterns not explicitly modeled

### Imbalanced Data Handling
- SMOTE adds synthetic fraud samples to training
- Test set remains imbalanced for honest evaluation
- F1 score preferred over accuracy
- Precision/recall trade-off can be tuned via threshold

---

## 📚 Further Improvements

1. **Feature Engineering:**
   - Add temporal patterns (hour, day, month)
   - Transaction sequence analysis
   - Merchant category features

2. **Model Tuning:**
   - Hyperparameter optimization (GridSearch, Bayesian)
   - Ensemble methods (stacking, blending)
   - Neural networks (LSTM, GRU for sequences)

3. **Deployment:**
   - Docker containerization
   - Kubernetes orchestration
   - Real-time monitoring & alerting
   - Model drift detection

4. **Data:**
   - Collect more recent data
   - Handle emerging fraud patterns
   - Feedback loop for model retraining

---

## 📖 Usage Examples

### Jupyter Notebook
```python
# Load and explore data
df = pd.read_csv('creditcard.csv')
print(df.describe())

# Run full pipeline
# See creditcard.ipynb for detailed steps
```

### Streamlit App
```bash
streamlit run evaluate_streamlit.py
# Open browser, upload CSV, select features, predict
```

### API Service
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "V1": 1.0, "V2": -0.5, ...,
        "V27": 0.2, "Amount": 100.0, "Time": 45000
    }
)
print(response.json())
# Output: {"LogisticRegression": {"predicted_class": 0}, ...}
```

### Batch Script
```bash
python evaluate_on_new_data.py \
  --input my_data.csv \
  --target Class \
  --models-dir ./models \
  --output my_results.json
```

---

## 🐛 Troubleshooting

**Error: Model not found**
- Check `models/` directory contains `.pkl` files
- Ensure model_config.json exists

**Error: Missing features**
- Verify CSV contains all required columns
- Check sidebar for expected feature names
- Use `evaluate_streamlit.py` for guided feature selection

**Error: Streamlit not starting**
- Install: `pip install streamlit`
- Check port 8501 is available

**Error: Poor predictions**
- Verify features match training features
- Check feature scaling (normalize if needed)
- Consider retraining with new data

---

## 📝 License & Citation

This project uses the Kaggle Credit Card Fraud Detection Dataset:
```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification.
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
```

---

## 👤 Author

Credit Card Fraud Detection System - Built with scikit-learn, XGBoost, and Streamlit

---

## 📞 Contact & Support

For questions or issues, refer to:
- **Notebook:** `creditcard.ipynb` - Full implementation details
- **Evaluation:** `README_EVALUATION.md` - Batch processing guide
- **Model Config:** `models/model_config.json` - Feature specifications

---

**Last Updated:** March 2, 2026


## FraudGuard Production Application

The repository now includes a branded FastAPI application layer and responsive dashboard around the existing trained model artifacts. The application preserves the original feature configuration and loads the available Logistic Regression, Random Forest, and XGBoost artifacts from `models/` without changing the training notebook.

### Run locally

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Open `http://localhost:8000` for the FraudGuard dashboard. API documentation is available at `http://localhost:8000/docs`, with health information at `/health`, registry metadata at `/api/models`, and validated inference at `POST /api/predict`.

The prediction payload is shaped as follows:

```json
{
  "features": {"V17": 0.0, "V14": 0.0, "V12": 0.0},
  "threshold": 0.5
}
```

The API requires the complete feature vector declared in `models/model_config.json`, validates finite numeric values, translates inference failures into safe user-facing errors, and returns per-model probabilities alongside the ensemble-ready result. `FRAUD_THRESHOLD`, `LOG_LEVEL`, and `CORS_ORIGINS` can be configured through environment variables; see `.env.example`.

### Container run

```bash
docker build -t fraudguard .
docker run --rm -p 8000:8000 fraudguard
```

### Verification

Focused API tests live in `tests/test_api.py`. Run them with `python -m pytest tests/test_api.py -q`. The production smoke test confirms that all three model artifacts load and that `/health` reports the service as ready.

### Repository contents

The raw `creditcard.csv` dataset remains available in the local project folder for training and experimentation but is excluded from GitHub because it is a large local data artifact. Download or place the dataset locally before running the notebook-based training workflow. Environment files, caches, logs, and generated outputs are also excluded; `.env.example` documents the supported runtime settings.

### Scope note

The deleted `evaluate_streamlit.py` file appears as a pre-existing working-tree deletion and was intentionally not restored or overwritten. The existing notebook, model artifacts, model configuration, dataset, and requirements were preserved.
