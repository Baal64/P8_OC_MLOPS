import streamlit as st
import plotly.graph_objects as go

from src.inference import predict, load_model

import json
from pathlib import Path
import pandas as pd

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Scoring Crédit", layout="centered")
st.title("Scoring client – Demande de crédit")
st.caption("Démo MLOps • Streamlit + Docker • Déploiement Hugging Face (sync depuis GitHub)")

# ---------- BUSINESS MAPPING ----------
LABEL_MAPPING = {
    0: "Client solvable – prêt accordable",
    1: "Client à risque – prêt non accordable",
}

# Exemple de seuil métier (à ajuster si tu en as un)
DEFAULT_THRESHOLD = 0.50


@st.cache_resource
def warmup_model():
    load_model()
    return True


def show_speedometer(p_default: float, threshold: float = 0.5) -> None:
    """
    p_default: probabilité de défaut entre 0 et 1
    threshold: seuil métier entre 0 et 1
    """
    p_default = max(0.0, min(1.0, float(p_default)))
    threshold = max(0.0, min(1.0, float(threshold)))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_default * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Risque de défaut", "font": {"size": 18}},
            gauge={
                # forme "compteur"
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticktext": ["0", "20", "40", "60", "80", "100"],
                    "tickwidth": 1,
                    "tickcolor": "gray",
                },
                # “aiguille” / indicateur (barre)
                "bar": {"color": "#0b1f2a", "thickness": 0.25},
                # segments colorés (rouge -> vert)
                "steps": [
                    {"range": [0, 20], "color": "#1bb55c"},   # vert
                    {"range": [20, 40], "color": "#7ed321"},  # vert clair
                    {"range": [40, 60], "color": "#f8e71c"},  # jaune
                    {"range": [60, 80], "color": "#f5a623"},  # orange
                    {"range": [80, 100], "color": "#d0021b"}, # rouge
                ],
                # seuil métier (trait noir)
                "threshold": {
                    "line": {"color": "black", "width": 5},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )

    # Look & feel plus proche de ton exemple (demi-jauge clean)
    fig.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=55, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def risk_level(p: float) -> tuple[str, str]:
    # p = probabilité de défaut entre 0 et 1
    if p < 0.30:
        return "Risque faible", "success"
    if p < 0.60:
        return "Risque modéré", "warning"
    return "Risque élevé", "error"


# Warmup
warmup_model()
st.success("Modèle chargé ✅")

tab_scoring, tab_monitoring, tab_about = st.tabs(["🧮 Scoring", "📊 Monitoring", "ℹ️ À propos"])


# -------------------- TAB: SCORING --------------------
with tab_scoring:
    st.subheader("Évaluation de la demande")

    threshold = st.slider(
        "Seuil de décision (probabilité de défaut)",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Seuil métier : plus il est bas, plus on refuse facilement (politique plus conservatrice).",
    )

    text = st.text_area(
        "Texte à analyser",
        placeholder="Ex: Demande de crédit auto 15k€, CDI depuis 3 ans, charges mensuelles..., historique...",
        height=200,
    )

    run = st.button("Calculer le score", type="primary")

    if run:
        if not text.strip():
            st.warning("Merci de saisir un texte avant de lancer le scoring.")
        else:
            try:
                # Ton wrapper renvoie (label, score) où label est 0/1.
                label, score = predict(text)
                label_int = int(label)

                # On construit une probabilité UNIFIÉE de défaut pour alimenter la jauge.
                # - si label=1 : score ~ P(défaut)
                # - si label=0 : score ~ P(solvable) donc P(défaut)=1-score
                p_default = float(score) if label_int == 1 else (1.0 - float(score))

                st.markdown("### Résultat")
                st.write(f"**Classe prédite :** {label_int} — {LABEL_MAPPING.get(label_int, str(label_int))}")

                # Décision métier selon le seuil choisi
                decision = "REFUS (risque élevé)" if p_default >= threshold else "ACCORD (risque acceptable)"
                st.write(f"**Décision (seuil {threshold:.2f}) :** {decision}")

                # Jauge vert -> rouge
                show_speedometer(p_default, threshold=threshold)

                level, tone = risk_level(p_default)

                if tone == "success":
                    st.success(f"✅ {level}")
                elif tone == "warning":
                    st.warning(f"⚠️ {level}")
                else:
                    st.error(f"⛔ {level}")

                st.write(f"**Probabilité de défaut estimée :** {p_default:.2%}")

                decision = "REFUS (risque élevé)" if p_default >= threshold else "ACCORD (risque acceptable)"
                st.info(f"Décision selon le seuil {threshold:.2f} : **{decision}**")

            except Exception as e:
                st.error("Erreur pendant le calcul du score.")
                st.exception(e)

    with st.expander("Interprétation"):
        st.write(
            "- `predict()` renvoie une classe binaire (0/1).\n"
            "- La jauge affiche une **probabilité de défaut** unifiée (entre 0 et 1).\n"
            "- La décision (accord/refus) est une **règle métier** basée sur un **seuil ajustable**."
        )


# -------------------- TAB: MONITORING --------------------
with tab_monitoring:
    st.subheader("Monitoring")

    log_path = Path("logs") / "predictions.jsonl"

    if not log_path.exists():
        st.info("Aucun log pour l’instant. Lance quelques prédictions dans l’onglet Scoring.")
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

            st.metric("Nombre de prédictions", len(df))

            # Distribution classes
            if "label" in df.columns:
                counts = df["label"].value_counts()
                st.write("**Répartition des classes**")
                st.bar_chart(counts)

            # Stats score/latence
            if "score" in df.columns:
                st.write("**Score (dernières prédictions)**")
                st.line_chart(df["score"])

            if "latency_ms" in df.columns:
                st.write("**Latence (ms)**")
                st.line_chart(df["latency_ms"])

            with st.expander("Voir les logs (aperçu)"):
                st.dataframe(df.tail(50))


# -------------------- TAB: ABOUT --------------------
with tab_about:
    st.subheader("À propos")
    st.write(
        "Cette application illustre un workflow MLOps : packaging du modèle, "
        "déploiement Docker sur Hugging Face Spaces, et synchronisation automatisée depuis GitHub."
    )
