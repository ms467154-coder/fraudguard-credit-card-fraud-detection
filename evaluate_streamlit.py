"""Streamlit application for making predictions with saved classification models on new CSV data.

Usage:
    


The user can upload a CSV file, optionally select feature columns,
then run predictions against all models found in a specified directory (default `models`).
Predictions and probabilities are displayed interactively.
"""

import os
import glob
import json
from typing import Dict

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------

def load_models(models_dir: str) -> Dict[str, object]:
    models = {}
    patterns = ["*.pkl", "*.joblib", "*.sav"]
    for pat in patterns:
        for path in glob.glob(os.path.join(models_dir, pat)):
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed to load model {path}: {e}")
    return models


def load_model_config(models_dir: str) -> Dict[str, dict]:
    """Load model configuration (features for each model)."""
    config_path = os.path.join(models_dir, "model_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load model config from {config_path}: {e}")
    return {}


def predict_with_model(model, X):
    """Make predictions and return class + probability."""
    res = {}
    y_pred = model.predict(X)
    res['predictions'] = y_pred
    
    # Get probability/confidence scores
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X)
            res['probabilities'] = y_proba
        except Exception:
            res['probabilities'] = None
    elif hasattr(model, 'decision_function'):
        try:
            y_score = model.decision_function(X)
            res['decision_scores'] = y_score
        except Exception:
            res['decision_scores'] = None
    else:
        res['probabilities'] = None
    
    return res


# -----------------------------------------------------------------------------
# Streamlit UI (enhanced design)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Fraud Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom styling to give a modern, professional look
CUSTOM_CSS = """
<style>
:root {
  --bg:#0b1220;
  --panel:#0f1724;
  --accent:#00b4d8;
  --accent-2:#7c3aed;
  --muted:#9aa6b2;
  --card:#0f1b2b;
}
html, body, [data-testid='stAppViewContainer'] {
  background: linear-gradient(180deg, #071224 0%, #081726 100%);
  color: #e6eef8;
}
.app-header {
  border-radius: 12px;
  padding: 18px 22px;
  margin-bottom: 12px;
  background: linear-gradient(90deg, rgba(0,180,216,0.12), rgba(124,58,237,0.08));
  box-shadow: 0 6px 18px rgba(7,12,20,0.6);
}
.app-title {font-size:28px; font-weight:700; color: #fff;}
.app-subtitle {color: var(--muted); margin-top:6px;}
.panel {background: rgba(255,255,255,0.02); padding: 14px; border-radius:10px;}
.card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:10px;}
.metric-card {background: linear-gradient(90deg, rgba(124,58,237,0.16), rgba(0,180,216,0.12)); padding:10px; border-radius:10px; text-align:center; color:#fff}
.small {font-size:13px; color:var(--muted)}
.bold {font-weight:700}
.download-btn {background:linear-gradient(90deg,#00b4d8,#7c3aed); color:#fff; padding:8px 14px; border-radius:8px}
a.link {color: #9ad3ff}
.sidebar-header {font-size:16px; font-weight:700; color:#fff}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="app-header">
      <div class="app-title"> Fraud Detection — Prediction Studio</div>
      <div class="app-subtitle">Upload transactions and compare model predictions with confidence scores.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='small'>Designed for fast evaluation of saved classification models.</div>", unsafe_allow_html=True)

# Load models and config upfront for sidebar display
models_dir = "models"
models = load_models(models_dir)
model_config = load_model_config(models_dir)

# Sidebar: model guide with clean styling
with st.sidebar:
    st.markdown("<div class='sidebar-header'> Model Features Guide</div>", unsafe_allow_html=True)
    st.markdown("<div class='small' style='margin-bottom:8px'>Expand a model to view its feature set and description.</div>", unsafe_allow_html=True)
    for model_name in sorted(models.keys()):
        with st.expander(f" {model_name}"):
            if model_name in model_config:
                features = model_config[model_name].get("features", [])
                description = model_config[model_name].get("description", model_name)
                st.markdown(f"**{description}**")
                st.markdown(f"<div class='small'>Total features: {len(features)}</div>", unsafe_allow_html=True)
                # show compact list in two columns
                cols = st.columns(2)
                for idx, feature in enumerate(features):
                    cols[idx % 2].markdown(f"- {feature}")
            else:
                st.warning(f"No configuration found for {model_name}")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"]) 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdownL("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("###  Data Preview")
    st.dataframe(df.head(10))
    st.markdown(f"<div class='small'>Total rows: {len(df)}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Select Features for Each Model")

    # Create input fields for each model to let user select features
    model_feature_selections = {}

    for model_name in sorted(models.keys()):
        with st.expander(f"{model_name} - Select Features", expanded=False):
            if model_name in model_config:
                default_features = model_config[model_name].get("features", [])
                description = model_config[model_name].get("description", model_name)
                st.markdown(f"**{description}**")
                st.markdown(f"<div class='small'>Default: {len(default_features)} features</div>", unsafe_allow_html=True)

                # Get available columns from CSV
                available_cols = list(df.columns)

                # Pre-select default features that exist in the CSV
                preselected = [f for f in default_features if f in available_cols]

                # Let user select features
                selected_features = st.multiselect(
                    f"Features for {model_name}",
                    options=available_cols,
                    default=preselected,
                    key=f"{model_name}_features"
                )
                model_feature_selections[model_name] = selected_features
            else:
                st.warning(f"No configuration found for {model_name}")
                model_feature_selections[model_name] = []

    st.markdown("---")

    run_btn = st.button("Run Predictions")
    if run_btn:
        # Check if features are selected for all models
        models_with_no_features = [name for name, feats in model_feature_selections.items() if not feats]
        if models_with_no_features:
            st.error(f"Please select at least one feature for: {', '.join(models_with_no_features)}")
        else:
            with st.spinner("Making predictions..."):
                current_models = load_models(models_dir)
                current_model_config = load_model_config(models_dir)

            if not current_models:
                st.error("No models found in the directory")
            else:
                st.markdown("###  Prediction Configuration")

                # Display which features each model will use
                config_display = {}
                for model_name, selected_features in model_feature_selections.items():
                    config_display[model_name] = {
                        "Model": model_name,
                        "Feature Count": len(selected_features),
                        "Features": ", ".join(selected_features[:5]) + ("..." if len(selected_features) > 5 else "")
                    }

                st.dataframe(pd.DataFrame(config_display).T, use_container_width=True)

                # Make predictions with each model using selected features
                all_predictions = {}
                for model_name, model in current_models.items():
                    selected_features = model_feature_selections.get(model_name, [])

                    if not selected_features:
                        st.warning(f"No features selected for {model_name}. Skipping...")
                        continue

                    try:
                        X_model = df[selected_features]
                        pred_result = predict_with_model(model, X_model)
                        all_predictions[model_name] = pred_result
                        st.success(f"✅ {model_name} - Predictions made with {len(selected_features)} features")
                    except Exception as e:
                        st.error(f"❌ Error predicting with {model_name}: {e}")

                # Display predictions
                if all_predictions:
                    st.markdown("---")
                    st.success("✅ All predictions complete!")

                    # Create a results dataframe
                    results_data = {'Transaction_ID': range(len(df))}

                    for model_name, pred_result in all_predictions.items():
                        predictions = pred_result['predictions']
                        results_data[f'{model_name}_Prediction'] = predictions

                        # Add probability/confidence if available
                        if pred_result['probabilities'] is not None:
                            fraud_prob = pred_result['probabilities'][:, 1]
                            results_data[f'{model_name}_Fraud_Probability'] = fraud_prob

                    results_df = pd.DataFrame(results_data)

                    st.markdown("### 📈 Predictions Summary")
                    st.dataframe(results_df, use_container_width=True)

                    # Download button (styled via CSS)
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                    # Show distribution of predictions with styled metric cards
                    st.markdown("###  Prediction Distribution")
                    cols = st.columns(len(all_predictions))
                    for idx, model_name in enumerate(all_predictions.keys()):
                        predictions = all_predictions[model_name]['predictions']
                        fraud_count = int((predictions == 1).sum())
                        legit_count = int((predictions == 0).sum())
                        with cols[idx]:
                            st.markdown(f"<div class='metric-card'><div class='bold'>{model_name}</div><div class='small'>Fraud: {fraud_count} &nbsp; | &nbsp; Legit: {legit_count}</div></div>", unsafe_allow_html=True)