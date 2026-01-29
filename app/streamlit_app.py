import streamlit as st
from src.inference import predict, load_model

# ---------- UI CONFIG ----------
st.set_page_config(page_title="Scoring Crédit", layout="centered")
st.title("Scoring client – Demande de crédit")
st.caption("Démo MLOps • Streamlit + Docker • Déploiement Hugging Face (sync depuis GitHub)")

# ---------- BUSINESS MAPPING ----------
# Convention standard : 0 = solvable (accordable), 1 = défaut/risque (non accordable)
LABEL_MAPPING = {
    0: "Client solvable – prêt accordable",
    1: "Client à risque – prêt non accordable",
}


@st.cache_resource
def warmup_model():
    """Charge le modèle une seule fois au démarrage du conteneur."""
    load_model()
    return True


# Warmup
warmup_model()
st.success("Modèle chargé ✅")

# ---------- INPUT ----------
st.subheader("Entrée")
st.write(
    "Colle ici les informations nécessaires à l’évaluation (format texte). "
    "Exemple : résumé du dossier, contexte, éléments saillants."
)

text = st.text_area(
    "Texte à analyser",
    placeholder="Ex: Demande de crédit auto 15k€, CDI depuis 3 ans, charges mensuelles..., historique...",
    height=200,
)

col1, col2 = st.columns([1, 2])
with col1:
    run = st.button("Calculer le score", type="primary")
with col2:
    st.write("")

# ---------- PREDICT ----------
if run:
    if not text.strip():
        st.warning("Merci de saisir un texte avant de lancer le scoring.")
    else:
        try:
            label, score = predict(text)

            # Ton predict renvoie (label, score). Ici label est 0/1.
            label_int = int(label)

            st.markdown("### Résultat du scoring")
            st.write(f"**Décision :** {LABEL_MAPPING.get(label_int, str(label_int))}")

            # Interprétation du score :
            # - si label=1 : score = proba de défaut (risque)
            # - si label=0 : score = proba de solvabilité
            if label_int == 1:
                st.write(f"**Probabilité de défaut estimée :** {score:.2%}")
                st.warning("Risque élevé détecté (classe 1).")
            else:
                st.write(f"**Probabilité de solvabilité estimée :** {score:.2%}")
                st.success("Risque faible détecté (classe 0).")

        except Exception as e:
            st.error("Erreur pendant le calcul du score.")
            st.exception(e)

# ---------- EXPLAIN ----------
with st.expander("Interprétation (métier)"):
    st.write(
        "- `predict()` renvoie une **classe binaire** : 0 (solvable) ou 1 (risque/défaut).\n"
        "- Si le modèle expose `predict_proba()`, on récupère une **probabilité** (score) associée à la classe prédite.\n"
        "- En scoring crédit, la **décision** peut dépendre d’un **seuil métier** ajustable (coût FP/FN, politique risque)."
    )
