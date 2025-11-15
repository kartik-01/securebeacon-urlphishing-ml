"""Training pipeline for the phishing URL classifier."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
import shap
import tldextract
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import FeatureExtractor, FeatureExtractorConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train_model")

DATA_PATH = Path("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")
ARTIFACT_DIR = Path("artifacts")
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.json"
METRICS_PATH = Path("metrics.json")
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SHAP_PLOT_PATH = ARTIFACT_DIR / "shap_summary.png"
TLD_STATS_PATH = ARTIFACT_DIR / "tld_stats.json"
TOP_DOMAINS_PATH = Path("data/top_domains.txt")
TRUSTED_DOMAINS_PATH = ARTIFACT_DIR / "trusted_domains.json"
TOP_DOMAIN_AUGMENT_LIMIT = 200
TRUSTED_TOP_DOMAIN_LIMIT = 500
SAFE_LEGITIMATE_URLS = [
    "https://www.facebook.com/login",
    "https://www.facebook.com/security",
    "https://www.facebook.com/settings",
    "https://accounts.google.com/signin",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://www.google.com/accounts",
    "https://www.paypal.com/signin",
    "https://www.paypal.com/myaccount",
    "https://www.netflix.com/login",
    "https://www.netflix.com/browse",
    "https://www.microsoft.com/account",
    "https://account.microsoft.com/",
    "https://github.com/login",
    "https://login.live.com/login.srf",
    "https://my.university.edu/portal",
    "https://secure.bankofamerica.com/login/sign-in/signOnV2Screen.go",
    "https://www.chase.com/personal/logon",
    "https://fb.com",
    "https://jb.com",
    "https://harrypotter.com",
    "https://primarykeytech.in/",
    "https://maddock.primarykeytech.in/",
    "https://letterboxd.com",
    "https://filmfreeway.com",
    "https://indiewire.com",
    "https://mubi.com",
    "https://artstation.com",
    "https://soundstripe.com",
    "https://goodreads.com",
    "https://unsplash.com",
    "https://itch.io",
    "https://producthunt.com",
]

SAFE_LEGITIMATE_REPEATS = 40

CALIBRATION_CV = 5
RANDOM_STATE = 42


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={col: col.strip().lstrip("\ufeff") for col in df.columns})
    return df


def _compute_tld_legitimacy(df: pd.DataFrame) -> Dict[str, float]:
    if "TLD" not in df.columns:
        return {}
    grouped = df.groupby(df["TLD"].astype(str).str.lower())
    stats: Dict[str, float] = {}
    for tld, group in grouped:
        total = len(group)
        if total == 0:
            continue
        legit_ratio = float((group["label"] == 1).sum() / total)
        stats[tld] = legit_ratio
    return stats


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _canonical_domain(value: str) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if not candidate.startswith("http://") and not candidate.startswith("https://"):
        candidate = f"https://{candidate}"
    parsed = tldextract.extract(candidate)
    domain = parsed.registered_domain or parsed.domain
    return domain.lower() if domain else None


def _build_trusted_domains(curated_urls: Iterable[str], seed_domains: Iterable[str]) -> Set[str]:
    domains: Set[str] = set()
    for url in curated_urls:
        domain = _canonical_domain(url)
        if domain:
            domains.add(domain)
    for seed in seed_domains:
        domain = _canonical_domain(seed)
        if domain:
            domains.add(domain)
    return domains


def _load_dataset(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    df = _clean_columns(df)
    df = df[["URL", "label", "TLD"]].dropna(subset=["URL", "label"])  # keep necessary columns
    df["URL"] = df["URL"].astype(str)
    df["label"] = df["label"].astype(int)
    LOGGER.info("Dataset size after cleaning: %s", df.shape)
    return df


def _augment_with_top_domains(df: pd.DataFrame, top_domains: List[str], limit: int) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    for domain in top_domains[:limit]:
        domain = domain.strip().lower()
        if not domain:
            continue
        parsed = tldextract.extract(domain if "://" in domain else f"https://{domain}")
        tld = parsed.suffix or ""
        registered = parsed.top_domain_under_public_suffix or parsed.registered_domain or domain
        if not registered:
            continue
        url = f"https://{registered}"
        records.append({"URL": url, "label": 1, "TLD": tld})
    if not records:
        return df
    extra_df = pd.DataFrame(records)
    combined = pd.concat([df, extra_df[df.columns]], ignore_index=True)
    LOGGER.info("Augmented dataset with %d top-domain exemplars", len(records))
    return combined


def _augment_with_known_legit(df: pd.DataFrame, urls: List[str], repeats: int) -> pd.DataFrame:
    records = []
    for _ in range(max(1, repeats)):
        for url in urls:
            parsed = tldextract.extract(url)
            tld = parsed.suffix or ""
            records.append({"URL": url, "label": 1, "TLD": tld})
    if not records:
        return df
    extra_df = pd.DataFrame(records)
    combined = pd.concat([df, extra_df[df.columns]], ignore_index=True)
    LOGGER.info("Augmented dataset with %d curated legitimate URLs", len(records))
    return combined


def _extract_features(urls: Iterable[str], extractor: FeatureExtractor, workers: int = 8) -> pd.DataFrame:
    urls = list(urls)
    LOGGER.info("Extracting features for %d URLs", len(urls))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        feature_dicts = list(executor.map(extractor.extract_features, urls))
    feature_frame = pd.DataFrame(feature_dicts)
    LOGGER.info("Feature matrix shape: %s", feature_frame.shape)
    return feature_frame


def _build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.5,
        tree_method="hist",
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        n_jobs=4,
        eval_metric="logloss",
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df_base = _load_dataset(DATA_PATH)

    LOGGER.info("Computing TLD legitimacy statistics")
    tld_stats = _compute_tld_legitimacy(df_base)
    _save_json(TLD_STATS_PATH, tld_stats)

    seed_top_domains = FeatureExtractor._load_top_domains(TOP_DOMAINS_PATH, TRUSTED_TOP_DOMAIN_LIMIT)
    trusted_domains = _build_trusted_domains(SAFE_LEGITIMATE_URLS, seed_top_domains)
    _save_json(TRUSTED_DOMAINS_PATH, {"trusted_domains": sorted(trusted_domains)})

    extractor = FeatureExtractor(
        FeatureExtractorConfig(
            tld_stats_path=TLD_STATS_PATH,
            top_domain_cache=TOP_DOMAINS_PATH,
            trusted_domains_path=TRUSTED_DOMAINS_PATH,
        )
    )

    df = _augment_with_top_domains(df_base, extractor.top_domains, TOP_DOMAIN_AUGMENT_LIMIT)
    df = _augment_with_known_legit(df, SAFE_LEGITIMATE_URLS, SAFE_LEGITIMATE_REPEATS)

    feature_frame = _extract_features(df["URL"], extractor)
    FEATURE_COLUMNS = feature_frame.columns.tolist()
    _save_json(FEATURE_COLUMNS_PATH, {"features": FEATURE_COLUMNS})

    X = feature_frame.values.astype(np.float32)
    y = (df["label"].astype(int) == 0).astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    base_model = _build_model()
    calibrator = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv=CALIBRATION_CV, n_jobs=-1)
    LOGGER.info("Training calibrated model")
    calibrator.fit(X_train, y_train)

    LOGGER.info("Evaluating on hold-out set")
    y_pred = calibrator.predict(X_test)
    y_proba = calibrator.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    LOGGER.info("Metrics: %s", metrics)
    _save_json(METRICS_PATH, metrics)

    LOGGER.info("Fitting interpretation model for feature importance")
    interpretation_model = _build_model()
    interpretation_model.fit(X_train, y_train)
    feature_importances = interpretation_model.feature_importances_
    top_indices = feature_importances.argsort()[::-1][:20]
    top_features = [
        {
            "feature": FEATURE_COLUMNS[idx],
            "importance": float(feature_importances[idx]),
        }
        for idx in top_indices
    ]
    _save_json(ARTIFACT_DIR / "feature_importances.json", {"top_20": top_features})

    LOGGER.info("Generating SHAP summary plot")
    explainer = shap.TreeExplainer(interpretation_model)
    sample_size = min(5000, X_train.shape[0])
    shap_sample = X_train[:sample_size]
    shap_values = explainer.shap_values(shap_sample)
    import matplotlib.pyplot as plt

    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values
    shap.summary_plot(shap_to_plot, shap_sample, feature_names=FEATURE_COLUMNS, show=False)

    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    LOGGER.info("Persisting calibrated model to %s", MODEL_PATH)
    joblib.dump(
        {
            "model": calibrator,
            "feature_columns": FEATURE_COLUMNS,
            "interpretation_model": interpretation_model,
        },
        MODEL_PATH,
    )

    LOGGER.info("Training complete. Model ready for inference.")


if __name__ == "__main__":
    main()
