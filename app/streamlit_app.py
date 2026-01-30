import streamlit as st
import plotly.graph_objects as go

from src.inference import predict, load_model

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


def show_risk_gauge(p_default: float, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Affiche une jauge de risque (0-100%), vert -> jaune -> rouge."""
    p_default = max(0.0, min(1.0, float(p_default)))
    threshold = max(0.0, min(1.0, float(threshold)))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_default * 100,
            number={"suffix": "%"},
            title={"text": "Risque de défaut (probabilité)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkgray"},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},   # vert
                    {"range": [30, 60], "color": "#f1c40f"},  # jaune
                    {"range": [60, 100], "color": "#e74c3c"}, # rouge
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    st.plotly_chart(fig, use_container_width=True)


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
                show_risk_gauge(p_default, threshold=threshold)

                st.write(f"**Probabilité de défaut estimée :** {p_default:.2%}")

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
    st.subheader("Monitoring (à compléter)")
    st.write(
        "Prochaine évolution : journaliser les prédictions (classe, probabilité, latence) "
        "et afficher ici des statistiques (volume, distribution des scores, etc.)."
    )


# -------------------- TAB: ABOUT --------------------
with tab_about:
    st.subheader("À propos")
    st.write(
        "Cette application illustre un workflow MLOps : packaging du modèle, "
        "déploiement Docker sur Hugging Face Spaces, et synchronisation automatisée depuis GitHub."
    )
