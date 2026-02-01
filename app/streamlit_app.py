import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import expected_features, load_model, predict_proba_default, score_one_client

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Scoring Crédit", layout="centered")
st.title("Scoring client – Demande de crédit")
st.caption("Démo MLOps • Streamlit + Docker • Déploiement Hugging Face (sync depuis GitHub)")


@st.cache_resource
def warmup():
    load_model()
    return True


def show_speedometer(p_default: float, threshold: float = 0.5) -> None:
    p_default = max(0.0, min(1.0, float(p_default)))
    threshold = max(0.0, min(1.0, float(threshold)))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_default * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Risque de défaut", "font": {"size": 18}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickvals": [0, 20, 40, 60, 80, 100]},
                "bar": {"color": "#0b1f2a", "thickness": 0.25},  # risque
                "steps": [
                    {"range": [0, 20], "color": "#1bb55c"},
                    {"range": [20, 40], "color": "#7ed321"},
                    {"range": [40, 60], "color": "#f8e71c"},
                    {"range": [60, 80], "color": "#f5a623"},
                    {"range": [80, 100], "color": "#d0021b"},
                ],
                # repère seuil (on peut changer la couleur si tu veux)
                "threshold": {
                    "line": {"color": "black", "width": 5},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def risk_level(p: float) -> tuple[str, str]:
    if p < 0.30:
        return "Risque faible", "success"
    if p < 0.60:
        return "Risque modéré", "warning"
    return "Risque élevé", "error"


# Warmup model
warmup()
st.success("Modèle chargé ✅")

tab_scoring, tab_monitoring, tab_about = st.tabs(["🧮 Scoring", "📊 Monitoring", "ℹ️ À propos"])


# -------------------- TAB: SCORING --------------------
with tab_scoring:
    st.subheader("Scoring")

    threshold = st.slider(
        "Seuil métier (probabilité de défaut)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Décision = REFUS si p(défaut) ≥ seuil, sinon ACCORD.",
    )

    sub_form, sub_csv = st.tabs(["🧍 1 client (formulaire)", "📄 Batch (CSV)"])

    # ---- Formulaire 1 client ----
    with sub_form:
        st.write(
            "Saisie métier pour **1 client**. "
            "On construit un DataFrame aligné sur les features d'entraînement."
        )

        feats = expected_features()

        # ⚠️ IMPORTANT :
        # Si ton modèle a beaucoup de features, on ne peut pas toutes les saisir à la main.
        # On remplit donc les features non saisies avec 0 par défaut (POC).
        # Le mode CSV (ci-dessous) est le mode 'réaliste' si tu as beaucoup de colonnes.

        st.info(
            f"Le modèle attend {len(feats)} features. "
            "Ce formulaire est un POC : les champs non saisis sont remplis à 0."
        )

        # Champs métier (à adapter aux variables réelles de ton dataset)
        # -> Mets ici les features les plus parlantes côté métier
        # -> Si une feature n'existe pas dans feats, on l'ignore proprement
        def set_if_exists(d, key, value):
            if key in feats:
                d[key] = value

        user = {}

        # Exemples courants (Home Credit-like). Adapte si tes colonnes diffèrent.
        set_if_exists(user, "AMT_INCOME_TOTAL", st.number_input("Revenu annuel", min_value=0.0, value=50000.0, step=1000.0))
        set_if_exists(user, "AMT_CREDIT", st.number_input("Montant du crédit", min_value=0.0, value=15000.0, step=500.0))
        set_if_exists(user, "AMT_ANNUITY", st.number_input("Mensualité", min_value=0.0, value=300.0, step=10.0))
        set_if_exists(user, "DAYS_BIRTH", st.number_input("Âge (jours négatifs si Home Credit)", value=-12000))
        set_if_exists(user, "DAYS_EMPLOYED", st.number_input("Ancienneté emploi (jours, souvent négatif)", value=-1000))
        set_if_exists(user, "CNT_CHILDREN", st.number_input("Nombre d'enfants", min_value=0, value=0, step=1))

        # Construire le dict complet strict (defaults)
        full = {c: 0 for c in feats}
        full.update(user)

        if st.button("Calculer le score", type="primary"):
            try:
                res = score_one_client(full, threshold=threshold, log=True)
                p_default = float(res["p_default"])
                decision = res["decision"]

                st.markdown("### Résultat")
                st.write(f"**Décision (seuil {threshold:.2f}) :** {decision}")

                show_speedometer(p_default, threshold=threshold)

                level, tone = risk_level(p_default)
                if tone == "success":
                    st.success(f"✅ {level}")
                elif tone == "warning":
                    st.warning(f"⚠️ {level}")
                else:
                    st.error(f"⛔ {level}")

                st.write(f"**Probabilité de défaut estimée :** {p_default:.2%}")
                st.caption("Barre sombre = risque estimé • Trait noir = seuil de décision")

                with st.expander("Aperçu des features envoyées au modèle (1 ligne)"):
                    st.dataframe(pd.DataFrame([full])[feats].head(1))

            except Exception as e:
                st.error("Erreur pendant le scoring (alignement features / types).")
                st.exception(e)

    # ---- Batch CSV ----
    with sub_csv:
        st.write("Upload un CSV contenant les **mêmes colonnes que l'entraînement** (sans la cible).")
        file = st.file_uploader("Fichier CSV", type=["csv"])

        if file is not None:
            try:
                df = pd.read_csv(file)
                feats = expected_features()

                missing = [c for c in feats if c not in df.columns]
                if missing:
                    st.error(f"Colonnes manquantes (strict align): {missing}")
                else:
                    p_defaults = predict_proba_default(df)
                    decisions = (p_defaults >= threshold).astype(int)

                    st.markdown("### Résultats batch")
                    st.metric("Nombre de clients", len(df))
                    st.write("**Distribution des décisions (1 = défaut/risque)**")
                    st.bar_chart(decisions.value_counts())

                    with st.expander("Aperçu (20 premières lignes)"):
                        out = pd.DataFrame(
                            {"p_default": p_defaults, "decision_1_defaut": decisions},
                            index=df.index,
                        )
                        st.dataframe(out.head(20))

                    st.caption("En batch, on ne loggue pas chaque ligne pour éviter des logs volumineux.")

            except Exception as e:
                st.error("Erreur lors du traitement du CSV.")
                st.exception(e)

    with st.expander("Interprétation (métier)"):
        st.write(
            "- Le modèle reçoit un **DataFrame pandas** aligné sur les features d'entraînement.\n"
            "- Il retourne une **probabilité de défaut** via `predict_proba`.\n"
            "- La décision est ensuite prise via un **seuil métier** ajustable.\n"
            "- Le formulaire est un POC : si le modèle a beaucoup de colonnes, le mode CSV est le plus fiable."
        )


# -------------------- TAB: MONITORING --------------------
with tab_monitoring:
    st.subheader("Monitoring")

    log_path = Path("logs") / "predictions.jsonl"
    if not log_path.exists():
        st.info("Aucun log pour l’instant. Lance des prédictions dans l’onglet Scoring (mode 1 client).")
    else:
        rows = []
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        if not rows:
            st.info("Logs vides ou illisibles.")
        else:
            df = pd.DataFrame(rows)

            st.metric("Nombre de prédictions loggées", len(df))

            if "p_default" in df.columns:
                st.write("**Probabilité de défaut (historique)**")
                st.line_chart(df["p_default"])

            if "decision" in df.columns:
                st.write("**Distribution des décisions**")
                st.bar_chart(df["decision"].value_counts())

            if "latency_ms" in df.columns:
                st.write("**Latence (ms)**")
                st.line_chart(df["latency_ms"])

            with st.expander("Voir les logs (50 derniers)"):
                st.dataframe(df.tail(50))


# -------------------- TAB: ABOUT --------------------
with tab_about:
    st.subheader("À propos")
    st.write(
        "Application de scoring crédit déployée sur Hugging Face via Docker, "
        "avec synchronisation depuis GitHub. "
        "Le modèle consomme un DataFrame strictement aligné sur les features d'entraînement."
    )
