from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

_MODEL: Any | None = None


def _repo_root() -> pathlib.Path:
    # /app/src/inference.py -> parents[1] = /app (repo root dans Docker)
    return pathlib.Path(__file__).resolve().parents[1]


def _artifact_path() -> pathlib.Path:
    return _repo_root() / "src" / "models" / "model.joblib"


def load_model() -> Any:
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(_artifact_path())
    return _MODEL


def expected_features() -> list[str]:
    """Liste des colonnes attendues par le modèle (alignée avec l'entraînement)."""
    model = load_model()
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise RuntimeError(
            "Le modèle n'expose pas feature_names_in_. "
            "Impossible d'aligner strictement les features."
        )
    return list(names)


def _log_event(event: dict) -> None:
    """Log JSONL (1 event par ligne). Fail-safe : ne casse jamais le scoring."""
    try:
        log_dir = _repo_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "predictions.jsonl"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def predict_proba_default(df: pd.DataFrame, default_class: int | str = 1) -> pd.Series:
    """
    Retourne la probabilité de défaut (classe 1 par défaut) pour chaque ligne du DF.
    - df doit contenir au minimum toutes les colonnes attendues par l'entraînement.
    - colonnes en trop -> ignorées
    - colonnes manquantes -> erreur
    """
    model = load_model()
    feats = expected_features()

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes (strict align): {missing}")

    X = df[feats].copy()

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Le modèle ne supporte pas predict_proba().")

    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise RuntimeError("Impossible de récupérer model.classes_.")

    classes_list = list(classes)
    try:
        idx = classes_list.index(default_class)
    except ValueError:
        idx = classes_list.index(str(default_class))

    return pd.Series(proba[:, idx], index=df.index, name="p_default")


def score_one_client(
    features: dict,
    threshold: float,
    default_class: int | str = 1,
    log: bool = True,
) -> dict:
    """
    Score un seul client.
    - features : dict {col: value}
    - threshold : seuil métier
    Retourne un dict: p_default, decision (ACCORD/REFUS), latency_ms
    """
    t0 = time.perf_counter()

    df = pd.DataFrame([features])
    p_default = float(predict_proba_default(df, default_class=default_class).iloc[0])
    decision = "REFUS" if p_default >= float(threshold) else "ACCORD"
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out = {
        "p_default": p_default,
        "decision": decision,
        "latency_ms": round(latency_ms, 3),
    }

    if log:
        _log_event(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": "single",
                "threshold": float(threshold),
                "p_default": p_default,
                "decision": decision,
                "latency_ms": round(latency_ms, 3),
                "n_features": len(features),
            }
        )

    return out