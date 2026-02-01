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

# Mapping UI -> modèle (num pipeline)
GENDER_MAP = {"F": 0, "M": 1}

EDU_MAP = {
    "Bac": 0,
    "Bac+2": 1,
    "Licence": 2,
    "Master": 3,
    "Doctorat": 4,
    "Autre": 5,
}



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
                # repère seuil
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


def key_for(name: str) -> str:
    return f"feat_{name}"


def as_int_bool(x: bool) -> int:
    return 1 if x else 0

@st.cache_resource
def get_feature_groups():
    model = load_model()
    ct = model.named_steps["prep"]
    num_cols = []
    cat_cols = []
    for name, _, cols in ct.transformers:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    return num_cols, cat_cols


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
        key="threshold_global",
    )

    sub_form, sub_csv = st.tabs(["🧍 1 client (formulaire)", "📄 Batch (CSV)"])

    # ---- Formulaire 1 client ----
    with sub_form:
        st.write("Formulaire métier (27 variables) — saisie d’un collaborateur / client interne.")

        feats = expected_features()
        values = {c: 0 for c in feats}

        st.divider()
        col_left, col_right = st.columns(2)

        # Colonne gauche
        with col_left:
            st.markdown("### Identité")
            values["age"] = st.number_input("Âge", 16, 80, 35, 1, key=key_for("age"))

            # UI: F/M -> modèle: 0/1 (num pipeline)
            genre_ui = st.selectbox("Genre", ["F", "M"], index=0, key=key_for("genre_ui"))
            values["genre"] = GENDER_MAP[genre_ui]

            st.markdown("### Situation")
            values["statut_marital"] = st.selectbox(
                "Statut marital",
                ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve", "Autre"],
                index=0,
                key=key_for("statut_marital"),
            )

            st.markdown("### Localisation / Poste")
            values["departement"] = str(
                st.text_input("Département (ex: 75, 92...)", value="75", key=key_for("departement"))
            )
            values["poste"] = str(st.text_input("Poste / Intitulé", value="Employé", key=key_for("poste")))

            st.markdown("### Revenus & charge")
            values["revenu_mensuel"] = st.number_input(
                "Revenu mensuel (€)", min_value=0.0, value=2500.0, step=100.0, key=key_for("revenu_mensuel")
            )
            values["distance_domicile_travail"] = st.number_input(
                "Distance domicile–travail (km)", min_value=0.0, value=10.0, step=1.0, key=key_for("distance_domicile_travail")
            )

            st.markdown("### Éducation")
            edu_ui = st.selectbox(
                "Niveau d’éducation",
                ["Bac", "Bac+2", "Licence", "Master", "Doctorat", "Autre"],
                index=3,  # Master par défaut si tu veux
                key=key_for("niveau_education_ui"),
            )
            values["niveau_education"] = EDU_MAP[edu_ui]
            values["domaine_etude"] = str(st.text_input("Domaine d’étude", value="Général", key=key_for("domaine_etude")))

        # Colonne droite
        with col_right:
            st.markdown("### Expérience / carrière")
            values["nombre_experiences_precedentes"] = st.number_input(
                "Nombre d’expériences précédentes", 0, 50, 2, 1, key=key_for("nombre_experiences_precedentes")
            )
            values["annee_experience_totale"] = st.number_input(
                "Années d’expérience totale", 0, 60, 8, 1, key=key_for("annee_experience_totale")
            )
            values["annees_dans_l_entreprise"] = st.number_input(
                "Années dans l’entreprise", 0, 60, 3, 1, key=key_for("annees_dans_l_entreprise")
            )
            values["annees_dans_le_poste_actuel"] = st.number_input(
                "Années dans le poste actuel", 0, 60, 2, 1, key=key_for("annees_dans_le_poste_actuel")
            )
            values["annes_sous_responsable_actuel"] = st.number_input(
                "Années sous le responsable actuel", 0, 60, 2, 1, key=key_for("annes_sous_responsable_actuel")
            )
            values["annees_depuis_la_derniere_promotion"] = st.number_input(
                "Années depuis la dernière promotion", 0, 60, 1, 1, key=key_for("annees_depuis_la_derniere_promotion")
            )

            st.markdown("### Organisation / formation")
            values["niveau_hierarchique_poste"] = st.number_input(
                "Niveau hiérarchique du poste", 1, 10, 2, 1, key=key_for("niveau_hierarchique_poste")
            )
            values["nb_formations_suivies"] = st.number_input(
                "Nombre de formations suivies", 0, 100, 1, 1, key=key_for("nb_formations_suivies")
            )
            values["nombre_participation_pee"] = st.number_input(
                "Nombre de participations PEE", 0, 100, 0, 1, key=key_for("nombre_participation_pee")
            )

            st.markdown("### Déplacements")
            values["frequence_deplacement"] = st.selectbox(
                "Fréquence de déplacement", ["Jamais", "Rare", "Fréquent"], index=0, key=key_for("frequence_deplacement")
            )

        st.divider()

        # Satisfaction
        st.markdown("### Satisfaction (échelle 1–4)")
        sat_cols = st.columns(2)
        with sat_cols[0]:
            values["satisfaction_employee_environnement"] = st.slider(
                "Satisfaction environnement", 1, 4, 3, key=key_for("satisfaction_employee_environnement")
            )
            values["satisfaction_employee_nature_travail"] = st.slider(
                "Satisfaction nature du travail", 1, 4, 3, key=key_for("satisfaction_employee_nature_travail")
            )
            values["satisfaction_employee_equilibre_pro_perso"] = st.slider(
                "Satisfaction équilibre pro/perso", 1, 4, 3, key=key_for("satisfaction_employee_equilibre_pro_perso")
            )
        with sat_cols[1]:
            values["satisfaction_employee_equipe"] = st.slider(
                "Satisfaction équipe", 1, 4, 3, key=key_for("satisfaction_employee_equipe")
            )

        # Performance / salaire
        st.markdown("### Performance / salaire")
        perf_cols = st.columns(2)
        with perf_cols[0]:
            values["note_evaluation_precedente"] = st.slider(
                "Note évaluation précédente (1–5)", 1, 5, 3, key=key_for("note_evaluation_precedente")
            )
            values["note_evaluation_actuelle"] = st.slider(
                "Note évaluation actuelle (1–5)", 1, 5, 3, key=key_for("note_evaluation_actuelle")
            )
        with perf_cols[1]:
            values["augementation_salaire_precedente"] = st.number_input(
                "Augmentation salaire précédente (%)", 0.0, 100.0, 5.0, 0.5, key=key_for("augementation_salaire_precedente")
            )
            values["heure_supplementaires"] = as_int_bool(
                st.checkbox("Heures supplémentaires", value=False, key=key_for("heure_supplementaires"))
            )

        # Construire DF strict
        df = pd.DataFrame([values])[feats]

        if st.button("Calculer le score", type="primary", key="btn_score_single"):
            try:
                num_cols, cat_cols = get_feature_groups()

                for c in num_cols:
                    df[c] = pd.to_numeric(df[c], errors="raise")

                for c in cat_cols:
                    df[c] = df[c].astype(str)

                p_default = float(predict_proba_default(df).iloc[0])
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
                st.error("Erreur pendant le scoring (types / encodage).")
                st.exception(e)

    # ---- Batch CSV ----
    with sub_csv:
        st.write("Upload un CSV contenant les **mêmes colonnes que l'entraînement** (sans la cible).")
        file = st.file_uploader("Fichier CSV", type=["csv"], key="csv_uploader")

        if file is not None:
            try:
                df_in = pd.read_csv(file)
                feats = expected_features()

                missing = [c for c in feats if c not in df_in.columns]
                if missing:
                    st.error(f"Colonnes manquantes (strict align): {missing}")
                else:
                    p_defaults = predict_proba_default(df_in)
                    decisions = (p_defaults >= threshold).astype(int)

                    st.markdown("### Résultats batch")
                    st.metric("Nombre de clients", len(df_in))
                    st.write("**Distribution des décisions (1 = défaut/risque)**")
                    st.bar_chart(decisions.value_counts())

                    with st.expander("Aperçu (20 premières lignes)"):
                        out = pd.DataFrame({"p_default": p_defaults, "decision_1_defaut": decisions})
                        st.dataframe(out.head(20))

            except Exception as e:
                st.error("Erreur lors du traitement du CSV.")
                st.exception(e)

    with st.expander("Interprétation (métier)"):
        st.write(
            "- Le modèle reçoit un **DataFrame pandas** aligné sur les features d'entraînement.\n"
            "- Prétraitement : numériques (imputer median + scaler), catégorielles (imputer most_frequent + one-hot).\n"
            "- `genre` est numérique côté modèle : l’UI F/M est convertie en 0/1.\n"
            "- Le modèle retourne une **probabilité de défaut** via `predict_proba`.\n"
            "- La décision est ensuite prise via un **seuil métier** ajustable."
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
            df_logs = pd.DataFrame(rows)
            st.metric("Nombre de prédictions loggées", len(df_logs))

            if "p_default" in df_logs.columns:
                st.write("**Probabilité de défaut (historique)**")
                st.line_chart(df_logs["p_default"])

            if "decision" in df_logs.columns:
                st.write("**Distribution des décisions**")
                st.bar_chart(df_logs["decision"].value_counts())

            if "latency_ms" in df_logs.columns:
                st.write("**Latence (ms)**")
                st.line_chart(df_logs["latency_ms"])

            with st.expander("Voir les logs (50 derniers)"):
                st.dataframe(df_logs.tail(50))


# -------------------- TAB: ABOUT --------------------
with tab_about:
    st.subheader("À propos")
    st.write(
        "Application de scoring déployée sur Hugging Face via Docker, "
        "avec synchronisation depuis GitHub.\n\n"
        "Le modèle est un Pipeline sklearn : ColumnTransformer (num + cat) + LogisticRegression.\n"
        "Les features d’entrée doivent être strictement alignées avec l’entraînement."
    )