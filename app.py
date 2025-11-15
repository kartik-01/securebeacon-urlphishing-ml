"""FastAPI microservice for phishing URL classification."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from features import FeatureExtractor, FeatureExtractorConfig

LOGGER = logging.getLogger("urlscan_app")

ARTIFACT_PATH = Path("artifacts/model.pkl")
TLD_STATS_PATH = Path("artifacts/tld_stats.json")
TOP_DOMAINS_PATH = Path("data/top_domains.txt")

if not ARTIFACT_PATH.exists():  # pragma: no cover - runtime guard
    raise FileNotFoundError("Model artifact not found. Please run train_model.py first.")

MODEL_BUNDLE = joblib.load(ARTIFACT_PATH)
CALIBRATED_MODEL = MODEL_BUNDLE["model"]
FEATURE_COLUMNS: List[str] = MODEL_BUNDLE["feature_columns"]
INTERPRETATION_MODEL = MODEL_BUNDLE.get("interpretation_model")

EXTRACTOR = FeatureExtractor(
    FeatureExtractorConfig(
        tld_stats_path=TLD_STATS_PATH,
        top_domain_cache=TOP_DOMAINS_PATH,
        trusted_domains_path=Path("artifacts/trusted_domains.json"),
    )
)

SHAP_EXPLAINER = shap.TreeExplainer(INTERPRETATION_MODEL) if INTERPRETATION_MODEL is not None else None
BASELINE_VECTOR = np.zeros(len(FEATURE_COLUMNS), dtype=float)
PROBABILITY_TEMPERATURE = 1.5
PROBABILITY_CLIP = 1e-4
PHISHING_THRESHOLD = 0.7
SUSPICIOUS_THRESHOLD = 0.5


class PredictRequest(BaseModel):
    url: str

    @validator("url")
    def _validate_url(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("URL must be a non-empty string")
        return value.strip()


class PredictResponse(BaseModel):
    classification: str
    confidence: float
    probability: float
    top_features: List[str]


app = FastAPI(title="Brand-Aware Phishing Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    try:
        feature_map = EXTRACTOR.extract_features(payload.url)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Feature extraction failed")
        raise HTTPException(status_code=400, detail=f"Unable to extract features: {exc}")

    feature_vector = np.array([[feature_map.get(col, 0.0) for col in FEATURE_COLUMNS]], dtype=float)

    probability = float(CALIBRATED_MODEL.predict_proba(feature_vector)[0][1])
    probability = _calibrate_probability(probability)
    if probability >= PHISHING_THRESHOLD:
        classification = "phishing"
        confidence = probability
    elif probability >= SUSPICIOUS_THRESHOLD:
        classification = "suspicious"
        confidence = probability
    else:
        classification = "legitimate"
        confidence = 1.0 - probability

    top_features = _top_feature_names(feature_vector)

    return PredictResponse(
        classification=classification,
        confidence=round(confidence, 6),
        probability=round(probability, 6),
        top_features=top_features,
    )


def _top_feature_names(feature_vector: np.ndarray, top_k: int = 3) -> List[str]:
    if SHAP_EXPLAINER is not None:
        shap_values = SHAP_EXPLAINER.shap_values(feature_vector)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        contributions = shap_values[0]
        indices = np.argsort(np.abs(contributions))[::-1][:top_k]
        return [FEATURE_COLUMNS[idx] for idx in indices]

    # Fallback heuristic: choose features with largest magnitude
    diffs = np.abs(feature_vector[0] - BASELINE_VECTOR)
    indices = np.argsort(diffs)[::-1][:top_k]
    return [FEATURE_COLUMNS[idx] for idx in indices]


def _calibrate_probability(probability: float) -> float:
    probability = float(np.clip(probability, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
    logit = np.log(probability / (1.0 - probability))
    scaled = 1.0 / (1.0 + np.exp(-logit / PROBABILITY_TEMPERATURE))
    return float(np.clip(scaled, 0.01, 0.99))


@app.get("/health")
async def healthcheck() -> dict:
    return {"status": "ok", "features": len(FEATURE_COLUMNS)}
