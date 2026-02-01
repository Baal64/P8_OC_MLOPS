import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import expected_features, load_model, predict_proba_default

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

    # ✅ Un seul slider (partagé par Formulaire + CSV)
    threshold = st.slider(
        "Seuil métier (probabilité de défaut)",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Décision = REFUS si p(défaut) ≥ seuil, sinon ACCORD.",
        key="threshold_global",
    )

    sub_form, sub_csv = st.tabs(["🧍 1 client (formulaire)", "📄 Batch (CSV)"])

    # ---- Formulaire 1 client ----
    with sub_form:
        st.write(
            "Formulaire complet (**27 features**) : les champs sont générés automatiquement à partir "
            "des features d'entraînement du modèle. "
            "Le DataFrame est ensuite envoyé avec l'ordre exact attendu."
        )

        feats = expected_features()

        st.divider()
        st.caption("Astuce : tu peux laisser des valeurs par défaut et ne renseigner que les champs clés pour la démo.")

        col_left, col_right = st.columns(2)

        values = {}
        for i, c in enumerate(feats):
            target_col = col_left if i % 2 == 0 else col_right
            with target_col:
                # Ajoute une key unique par champ (super important si labels identiques un jour)
                widget_key = f"feat_{c}"

                if any(k in c.upper() for k in ["FLAG", "IND", "IS_", "HAS_"]) or c.upper().endswith("_YN"):
                    values[c] = st.number_input(c, value=0, step=1, key=widget_key)
                else:
                    values[c] = st.number_input(c, value=0.0, key=widget_key)

        if st.button("Calculer le score", type="primary", key="btn_score_single"):
            try:
                df = pd.DataFrame([values])[feats]  # ordre strict
                p_defaults = predict_proba_default(df)
                p_default = float(p_defaults.iloc[0])

                decision = "REFUS" if p_default >= threshold else "ACCORD"

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

                with st.expander("DataFrame envoyé au modèle (1 ligne)"):
                    st.dataframe(df)

            except Exception as e:
                st.error("Erreur pendant le scoring (alignement features / types).")
                st.exception(e)

    # ---- Batch CSV ----
    with sub_csv:
        st.write("Upload un CSV contenant les **mêmes colonnes que l'entraînement** (sans la cible).")
        file = st.file_uploader("Fichier CSV", type=["csv"], key="csv_uploader")

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
            "- Ici, tu peux scorer un client via formulaire, ou scorer un batch via CSV."
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
