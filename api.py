from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("fraudguard")


def load_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = MODEL_DIR / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models: dict[str, Any] = {}
    for model_name in config:
        model_path = MODEL_DIR / f"{model_name}.pkl"
        if model_path.exists():
            try:
                models[model_name] = joblib.load(model_path)
            except Exception:
                logger.exception("Unable to load model %s", model_name)
    return config, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_config, app.state.models = load_registry()
        logger.info("Loaded %d model artifacts", len(app.state.models))
    except Exception as exc:
        app.state.model_config, app.state.models = {}, {}
        app.state.startup_error = str(exc)
        logger.error("Model registry unavailable: %s", exc)
    yield


app = FastAPI(
    title="FraudGuard API",
    version="1.0.0",
    description="Production application API for the credit-card fraud detection models.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")
    features: dict[str, float] = Field(..., description="Transaction feature values keyed by model feature name")
    threshold: float = Field(default=THRESHOLD, ge=0.0, le=1.0)

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("At least one feature is required")
        if not all(np.isfinite(v) for v in value.values()):
            raise ValueError("Feature values must be finite numbers")
        return value


class Prediction(BaseModel):
    model: str
    predicted_class: int
    fraud_probability: float
    threshold: float
    interpretation: str


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    available_models: list[str]
    required_features: list[str]


def required_features(config: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for definition in config.values():
        for feature in definition.get("features", []):
            if feature not in ordered:
                ordered.append(feature)
    return ordered


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    models = getattr(request.app.state, "models", {})
    return {"status": "ok" if models else "degraded", "models_loaded": len(models), "service": "fraudguard-api"}


@app.get("/api/models")
def models(request: Request) -> dict[str, Any]:
    config = getattr(request.app.state, "model_config", {})
    loaded = getattr(request.app.state, "models", {})
    return {"models": [{"name": name, "description": item.get("description", ""), "loaded": name in loaded} for name, item in config.items()], "features": required_features(config), "threshold": THRESHOLD}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: Transaction, request: Request) -> PredictionResponse:
    config = getattr(request.app.state, "model_config", {})
    loaded = getattr(request.app.state, "models", {})
    if not loaded:
        raise HTTPException(status_code=503, detail="Prediction service is temporarily unavailable because no model artifacts are loaded.")

    results: list[Prediction] = []
    for name, model in loaded.items():
        features = config.get(name, {}).get("features", [])
        missing = [feature for feature in features if feature not in payload.features]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing required transaction features: {', '.join(missing)}")
        vector = np.asarray([[payload.features[feature] for feature in features]], dtype=float)
        try:
            probability = float(model.predict_proba(vector)[0][1]) if hasattr(model, "predict_proba") else float(model.decision_function(vector)[0])
            if not 0 <= probability <= 1:
                probability = 1 / (1 + np.exp(-probability))
            fraud = int(probability >= payload.threshold)
        except Exception as exc:
            logger.exception("Prediction failed for %s", name)
            raise HTTPException(status_code=500, detail=f"Model inference failed for {name}.") from exc
        results.append(Prediction(model=name, predicted_class=fraud, fraud_probability=round(probability, 6), threshold=payload.threshold, interpretation="Review transaction" if fraud else "Likely legitimate"))
    return PredictionResponse(predictions=results, available_models=list(loaded), required_features=required_features(config))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")
