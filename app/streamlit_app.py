import streamlit as st
from src.inference import predict, load_model

st.set_page_config(page_title="P8 - Scoring", layout="centered")
st.title("P8 – Application de scoring")
st.caption("Modèle chargé une seule fois • Déploiement Docker sur Hugging Face Spaces")

@st.cache_resource
def warmup():
    # force le chargement du modèle au démarrage du container
    load_model()
    return True

warmup()
st.success("Modèle chargé ✅")

st.subheader("Entrée texte")
text = st.text_area(
    "Saisis le texte à scorer",
    placeholder="Colle ici le texte (ex: description, commentaire, etc.)",
    height=180,
)

col1, col2 = st.columns([1, 2])
with col1:
    do_predict = st.button("Prédire", type="primary")
with col2:
    st.write("")

if do_predict:
    if not text.strip():
        st.warning("Merci de saisir un texte avant de lancer la prédiction.")
    else:
        try:
            label, score = predict(text)
            st.markdown("### Résultat")
            st.write(f"**Classe prédite :** {label}")
            st.write(f"**Score / probabilité :** {score:.4f}")
        except Exception as e:
            st.error("Erreur pendant la prédiction.")
            st.exception(e)

with st.expander("Détails techniques"):
    st.write(
        "- La fonction `predict(text)` vient de `src/inference.py`.\n"
        "- Le modèle est chargé une seule fois au démarrage grâce à `st.cache_resource`."
    )
