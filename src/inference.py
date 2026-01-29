from __future__ import annotations
import pathlib
from typing import Tuple, Any
import joblib

_MODEL: Any | None = None


def _artifact_path() -> pathlib.Path:
    # src/inference.py -> parents[0]=src, parents[1]=repo root
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    return repo_root / "src" / "models" / "model.joblib"



def load_model() -> Any:
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(_artifact_path())
    return _MODEL


def _reset_model_cache_for_tests() -> None:
    """Utilitaire pour les tests : vide le cache du modèle."""
    global _MODEL
    _MODEL = None


def _find_text_column_via_ct(model: Any):
    """Tente de retrouver le nom de la colonne texte en inspectant un ColumnTransformer
    et en cherchant un TfidfVectorizer/CountVectorizer dans ses branches."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline as SkPipeline
    except Exception:
        return None

    # Extraire un ColumnTransformer depuis un Pipeline
    ct = None
    if hasattr(model, "steps"):
        for _, step in model.steps:
            if isinstance(step, ColumnTransformer):
                ct = step
                break
            # parfois le CT est dans une sous-étape
            if isinstance(step, SkPipeline):
                for _, sub in step.steps:
                    if isinstance(sub, ColumnTransformer):
                        ct = sub
                        break

    if ct is None:
        return None

    # Chercher un vectorizer texte dans chaque branche

    text_col = None
    transformers = getattr(ct, "transformers_", None) or getattr(ct, "transformers", [])
    for name, tr, cols in transformers:
        # dérouler les pipelines internes
        stack = [tr]
        while stack:
            cur = stack.pop()
            if hasattr(cur, "steps"):  # Pipeline
                stack.extend([s for _, s in cur.steps])
                continue
            mod = cur.__class__.__module__
            cls = cur.__class__.__name__
            if "feature_extraction.text" in mod and cls in (
                "TfidfVectorizer",
                "CountVectorizer",
            ):
                # récupérer la/les colonnes associées
                if isinstance(cols, (list, tuple)):
                    if len(cols) >= 1:
                        text_col = cols[0]
                        return text_col
                elif isinstance(cols, str):
                    return cols
        # continue pour d'autres branches
    return text_col


def _try_dataframe(names, text_col, text, defaults_style="numeric"):
    """Construit une ligne DataFrame avec toutes les colonnes.
    - text_col reçoit `text`
    - les autres reçoivent des valeurs par défaut selon defaults_style
      'numeric' -> 0
      'string'  -> ''
      'none'    -> None
    """
    import pandas as pd

    if text_col is None:
        # en dernier recours: première colonne
        text_col = names[0]

    if defaults_style == "numeric":
        row = {n: 0 for n in names}
    elif defaults_style == "string":
        row = {n: "" for n in names}
    else:
        row = {n: None for n in names}

    row[text_col] = text
    return pd.DataFrame([row])


def _coerce_input(model: Any, text: str):
    """
    Stratégie d'adaptation de l'entrée :
    1) Essayer [text] (1D)
    2) Si le modèle expose feature_names_in_, on construit un DataFrame
       et on ESSAIE plusieurs combinaisons :
       - détecter la colonne texte via ColumnTransformer
       - puis valeurs par défaut 'numeric' -> 'string' -> 'none'
       - sinon, on essaie chaque colonne comme candidate pour le texte.
    3) En dernier recours, [[text]] (2D anonyme)
    """
    # 1) Essai direct 1D
    X_1d = [text]
    try:
        _ = model.predict(X_1d)
        return X_1d
    except Exception:
        pass

    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) >= 1:
        # 2a) essayer avec la colonne texte détectée
        text_col = _find_text_column_via_ct(model)
        for style in ("numeric", "string", "none"):
            try:
                df = _try_dataframe(list(names), text_col, text, defaults_style=style)
                _ = model.predict(df)
                return df
            except Exception:
                continue

        # 2b) sinon, essayer toutes les colonnes comme candidate texte
        for cand in list(names):
            for style in ("numeric", "string", "none"):
                try:
                    df = _try_dataframe(list(names), cand, text, defaults_style=style)
                    _ = model.predict(df)
                    return df
                except Exception:
                    continue

    # 3) Fallback brut : 2D anonyme
    return [[text]]


def predict(text: str) -> Tuple[str, float]:
    model = load_model()

    X = _coerce_input(model, text)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # supporte list/tuple/ndarray
        if isinstance(probs, (list, tuple)):
            idx = max(range(len(probs)), key=lambda i: probs[i])
            best = float(probs[idx])
        elif hasattr(probs, "argmax"):
            idx = int(probs.argmax())
            best = float(probs[idx])
        else:
            seq = list(probs)
            idx = max(range(len(seq)), key=lambda i: seq[i])
            best = float(seq[idx])

        classes = getattr(model, "classes_", None)
        if classes is None:
            try:
                classes = model.named_steps["clf"].classes_
            except Exception:
                raise RuntimeError("Impossible de retrouver classes_ du modèle.")
        return str(classes[idx]), best

    pred = model.predict(X)[0]
    return str(pred), 1.0
