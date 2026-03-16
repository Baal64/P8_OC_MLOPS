import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import numpy as np
import uuid

from src.inference import (
    expected_features,
    load_model,
    predict_proba_default,
    score_one_client,
)

# ---------- CONFIG ----------
st.set_page_config(page_title="Scoring Crédit", layout="wide")
st.title("Scoring client – Demande de crédit")
st.caption("Démo MLOps • Streamlit + Docker • Déploiement Hugging Face")

if "scored_client_id" not in st.session_state:
    st.session_state.scored_client_id = None

CLIENTS_PATH = Path("data/reference/clients_demo.csv")
THRESHOLD = 0.50

GENDER_MAP = {"F": 0, "M": 1}
EDU_MAP = {
    "Bac": 0,
    "Bac+2": 1,
    "Licence": 2,
    "Master": 3,
    "Doctorat": 4,
    "Autre": 5,
}
INV_GENDER_MAP = {v: k for k, v in GENDER_MAP.items()}
INV_EDU_MAP = {v: k for k, v in EDU_MAP.items()}


@st.cache_resource
def warmup():
    load_model()
    return True


@st.cache_data
def load_clients() -> pd.DataFrame:
    if not CLIENTS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CLIENTS_PATH}. "
            "Ajoute clients_demo.csv dans data/reference/."
        )
    return pd.read_csv(CLIENTS_PATH)


@st.cache_resource
def get_feature_groups():
    model = load_model()
    ct = model.named_steps["prep"]

    num_cols, cat_cols = [], []
    for name, _, cols in ct.transformers:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)

    return num_cols, cat_cols


def coerce_df_types(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = get_feature_groups()
    df = df.copy()

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise")

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


def risk_level(p: float) -> tuple[str, str]:
    if p < 0.30:
        return "Risque faible", "success"
    if p < 0.60:
        return "Risque modéré", "warning"
    return "Risque élevé", "error"


def show_speedometer(p_default: float, threshold: float = 0.5) -> None:
    p_default = max(0.0, min(1.0, float(p_default)))
    threshold = max(0.0, min(1.0, float(threshold)))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_default * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={
                "text": f"Seuil de décision : {threshold:.2f}",
                "font": {"size": 18},
            },
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickvals": [0, 20, 40, 60, 80, 100]},
                "bar": {"color": "#0b1f2a", "thickness": 0.25},
                "steps": [
                    {"range": [0, 20], "color": "#1bb55c"},
                    {"range": [20, 40], "color": "#7ed321"},
                    {"range": [40, 60], "color": "#f8e71c"},
                    {"range": [60, 80], "color": "#f5a623"},
                    {"range": [80, 100], "color": "#d0021b"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 5},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def client_to_feature_dict(row: pd.Series, feats: list[str]) -> dict:
    values = {c: 0 for c in feats}
    for c in feats:
        if c in row.index:
            values[c] = row[c]
    return values


def save_prediction_log(p_default: float, decision: str, n_features: int) -> None:
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "predictions.jsonl"
        payload = {
            "p_default": float(p_default),
            "decision": decision,
            "n_features": n_features,
        }
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def compute_psi(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    breakpoints = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, breakpoints)

    # éviter des bins identiques si variable peu variée
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum(
        (actual_counts - expected_counts)
        * np.log((actual_counts + 1e-6) / (expected_counts + 1e-6))
    )

    return float(psi)


def drift_status(psi_value: float) -> tuple[str, str]:
    if np.isnan(psi_value):
        return "Indisponible", "info"
    if psi_value < 0.10:
        return "Stable", "success"
    if psi_value < 0.25:
        return "Drift modéré", "warning"
    return "Drift important", "error"


warmup()

if "current_client_features" not in st.session_state:
    st.session_state.current_client_features = None

if "current_client_id" not in st.session_state:
    st.session_state.current_client_id = None

if "current_score" not in st.session_state:
    st.session_state.current_score = None

if "current_decision" not in st.session_state:
    st.session_state.current_decision = None

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


tab_client, tab_decision, tab_monitoring, tab_about = st.tabs(
    ["👤 Client", "🧮 Décision", "📊 Monitoring", "ℹ️ Explications"]
)

# -------------------- TAB: CLIENT --------------------
with tab_client:
    st.subheader("Sélection et édition d’un client")

    try:
        clients_df = load_clients()
    except Exception as e:
        st.error("Impossible de charger le fichier clients_demo.csv")
        st.exception(e)
        st.stop()

    feats = expected_features()

    if "client_id" not in clients_df.columns:
        st.error("Le fichier clients_demo.csv doit contenir une colonne 'client_id'.")
        st.stop()

    st.info(f"Seuil métier utilisé : {THRESHOLD:.2f}")

    # ----------------------------
    # Ligne d'action compacte
    # ----------------------------
    action_col1, action_col2, action_col3 = st.columns([2, 1, 1])

    default_client_id = st.session_state.current_client_id or "C001"

    with action_col1:
        client_id_input = st.text_input(
            "Identifiant client",
            value=str(default_client_id),
            key="client_id_search",
        )

    load_clicked = False
    save_clicked = False

    with action_col2:
        load_clicked = st.button(
            "Charger le client", key="load_client_button", use_container_width=True
        )

    with action_col3:
        calculate_clicked = st.button(
            "Calculer la décision",
            key="calculate_decision_button_top",
            use_container_width=True,
            type="primary",
        )

    # ----------------------------
    # Chargement du client
    # ----------------------------
    if load_clicked:
        match = clients_df.loc[
            clients_df["client_id"].astype(str) == str(client_id_input).strip()
        ]
        if match.empty:
            st.error(f"Client introuvable : {client_id_input}")
        else:
            selected_row = match.iloc[0]
            st.session_state.current_client_id = str(client_id_input).strip()
            st.session_state.current_client_features = client_to_feature_dict(
                selected_row, feats
            )

            st.session_state.current_score = None
            st.session_state.current_decision = None

            st.success(f"Client {client_id_input} chargé.")

    if st.session_state.current_client_features is None:
        st.warning(
            "Aucun client chargé. Saisis un identifiant puis clique sur 'Charger le client'."
        )
        st.stop()

    base_values = st.session_state.current_client_features.copy()
    edited = dict(base_values)

    # ----------------------------
    # Formulaire compact 3 colonnes
    # ----------------------------
    col1, col2, col3 = st.columns(3)

    # ----- Colonne 1 -----
    with col1:
        st.markdown("### Identité")
        edited["age"] = st.number_input(
            "Âge",
            min_value=16,
            max_value=80,
            value=int(base_values.get("age", 35)),
            step=1,
            key=f"age_input_{st.session_state.current_client_id}",
        )

        genre_value = int(base_values.get("genre", 0))
        genre_default = INV_GENDER_MAP.get(genre_value, "F")
        genre_ui = st.selectbox(
            "Genre",
            ["F", "M"],
            index=["F", "M"].index(genre_default),
            key=f"genre_input_{st.session_state.current_client_id}",
        )
        edited["genre"] = GENDER_MAP[genre_ui]

        marital_options = [
            "Célibataire",
            "Marié(e)",
            "Divorcé(e)",
            "Veuf/Veuve",
            "Autre",
        ]
        marital_default = str(base_values.get("statut_marital", "Célibataire"))
        marital_index = (
            marital_options.index(marital_default)
            if marital_default in marital_options
            else 0
        )
        edited["statut_marital"] = st.selectbox(
            "Statut marital",
            marital_options,
            index=marital_index,
            key=f"statut_marital_input_{st.session_state.current_client_id}",
        )

        edited["departement"] = str(
            st.text_input(
                "Département",
                value=str(base_values.get("departement", "75")),
                key=f"departement_input_{st.session_state.current_client_id}",
            )
        )

        edited["poste"] = str(
            st.text_input(
                "Poste",
                value=str(base_values.get("poste", "Employé")),
                key=f"poste_input_{st.session_state.current_client_id}",
            )
        )

        edited["revenu_mensuel"] = st.number_input(
            "Revenu mensuel (€)",
            min_value=0.0,
            value=float(base_values.get("revenu_mensuel", 2500.0)),
            step=100.0,
            key=f"revenu_input_{st.session_state.current_client_id}",
        )

        edited["distance_domicile_travail"] = st.number_input(
            "Distance domicile-travail",
            min_value=0.0,
            value=float(base_values.get("distance_domicile_travail", 10.0)),
            step=1.0,
            key=f"distance_input_{st.session_state.current_client_id}",
        )

    # ----- Colonne 2 -----
    with col2:
        st.markdown("### Éducation / carrière")
        edu_value = int(base_values.get("niveau_education", 2))
        edu_default = INV_EDU_MAP.get(edu_value, "Licence")
        edu_options = ["Bac", "Bac+2", "Licence", "Master", "Doctorat", "Autre"]
        edited["niveau_education"] = EDU_MAP[
            st.selectbox(
                "Niveau d’éducation",
                edu_options,
                index=edu_options.index(edu_default),
                key=f"education_input_{st.session_state.current_client_id}",
            )
        ]

        edited["domaine_etude"] = str(
            st.text_input(
                "Domaine d’étude",
                value=str(base_values.get("domaine_etude", "Général")),
                key=f"domaine_input_{st.session_state.current_client_id}",
            )
        )

        travel_options = ["Jamais", "Rare", "Fréquent"]
        travel_default = str(base_values.get("frequence_deplacement", "Jamais"))
        travel_index = (
            travel_options.index(travel_default)
            if travel_default in travel_options
            else 0
        )
        edited["frequence_deplacement"] = st.selectbox(
            "Fréquence de déplacement",
            travel_options,
            index=travel_index,
            key=f"deplacement_input_{st.session_state.current_client_id}",
        )

        edited["nombre_experiences_precedentes"] = st.number_input(
            "Expériences précédentes",
            min_value=0,
            max_value=50,
            value=int(base_values.get("nombre_experiences_precedentes", 2)),
            step=1,
            key=f"exp_prev_input_{st.session_state.current_client_id}",
        )

        edited["annee_experience_totale"] = st.number_input(
            "Expérience totale",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annee_experience_totale", 8)),
            step=1,
            key=f"exp_tot_input_{st.session_state.current_client_id}",
        )

        edited["annees_dans_l_entreprise"] = st.number_input(
            "Années dans l’entreprise",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annees_dans_l_entreprise", 3)),
            step=1,
            key=f"exp_entreprise_input_{st.session_state.current_client_id}",
        )

        edited["annees_dans_le_poste_actuel"] = st.number_input(
            "Années dans le poste",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annees_dans_le_poste_actuel", 2)),
            step=1,
            key=f"exp_poste_input_{st.session_state.current_client_id}",
        )

        edited["annes_sous_responsable_actuel"] = st.number_input(
            "Années sous responsable",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annes_sous_responsable_actuel", 2)),
            step=1,
            key=f"responsable_input_{st.session_state.current_client_id}",
        )

        edited["annees_depuis_la_derniere_promotion"] = st.number_input(
            "Années depuis promotion",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annees_depuis_la_derniere_promotion", 1)),
            step=1,
            key=f"promotion_input_{st.session_state.current_client_id}",
        )

    # ----- Colonne 3 -----
    with col3:
        st.markdown("### Organisation / performance")
        edited["niveau_hierarchique_poste"] = st.number_input(
            "Niveau hiérarchique",
            min_value=1,
            max_value=10,
            value=int(base_values.get("niveau_hierarchique_poste", 2)),
            step=1,
            key=f"hierarchie_input_{st.session_state.current_client_id}",
        )

        edited["nb_formations_suivies"] = st.number_input(
            "Formations suivies",
            min_value=0,
            max_value=100,
            value=int(base_values.get("nb_formations_suivies", 1)),
            step=1,
            key=f"formations_input_{st.session_state.current_client_id}",
        )

        edited["nombre_participation_pee"] = st.number_input(
            "Participations PEE",
            min_value=0,
            max_value=100,
            value=int(base_values.get("nombre_participation_pee", 0)),
            step=1,
            key=f"pee_input_{st.session_state.current_client_id}",
        )

        edited["heure_supplementaires"] = (
            1
            if st.checkbox(
                "Heures supplémentaires",
                value=bool(base_values.get("heure_supplementaires", 0)),
                key=f"heures_sup_input_{st.session_state.current_client_id}",
            )
            else 0
        )

        edited["augementation_salaire_precedente"] = st.number_input(
            "Augmentation salaire (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(base_values.get("augementation_salaire_precedente", 5.0)),
            step=0.5,
            key=f"augmentation_input_{st.session_state.current_client_id}",
        )

        edited["note_evaluation_precedente"] = st.slider(
            "Éval précédente",
            1,
            5,
            int(base_values.get("note_evaluation_precedente", 3)),
            key=f"eval_prev_input_{st.session_state.current_client_id}",
        )

        edited["note_evaluation_actuelle"] = st.slider(
            "Éval actuelle",
            1,
            5,
            int(base_values.get("note_evaluation_actuelle", 3)),
            key=f"eval_current_input_{st.session_state.current_client_id}",
        )

        edited["satisfaction_employee_environnement"] = st.slider(
            "Satisfaction environnement",
            1,
            4,
            int(base_values.get("satisfaction_employee_environnement", 3)),
            key=f"sat_env_input_{st.session_state.current_client_id}",
        )

        edited["satisfaction_employee_nature_travail"] = st.slider(
            "Satisfaction travail",
            1,
            4,
            int(base_values.get("satisfaction_employee_nature_travail", 3)),
            key=f"sat_travail_input_{st.session_state.current_client_id}",
        )

        edited["satisfaction_employee_equilibre_pro_perso"] = st.slider(
            "Équilibre pro/perso",
            1,
            4,
            int(base_values.get("satisfaction_employee_equilibre_pro_perso", 3)),
            key=f"sat_eq_input_{st.session_state.current_client_id}",
        )

        edited["satisfaction_employee_equipe"] = st.slider(
            "Satisfaction équipe",
            1,
            4,
            int(base_values.get("satisfaction_employee_equipe", 3)),
            key=f"sat_equipe_input_{st.session_state.current_client_id}",
        )

    # ----------------------------
    # Traitement du bouton save
    # ----------------------------
    if calculate_clicked:
        df_current = pd.DataFrame([edited])[feats]
        try:
            df_current = coerce_df_types(df_current)
            st.session_state.current_client_features = df_current.iloc[0].to_dict()

            result = score_one_client(
                st.session_state.current_client_features,
                threshold=THRESHOLD,
                log=True,
                session_id=st.session_state.session_id
            )

            st.session_state.current_score = float(result["p_default"])
            st.session_state.current_decision = result["decision"]
            st.session_state.scored_client_id = st.session_state.current_client_id

            st.success(
                "Décision calculée. Va dans l’onglet Décision pour voir le résultat."
            )
        except Exception as e:
            st.error("Erreur pendant le calcul de la décision.")
            st.exception(e)

# -------------------- TAB: DECISION --------------------
with tab_decision:
    st.subheader("Décision")

    if st.session_state.current_client_features is None:
        st.info("Aucun client sélectionné. Va d’abord dans l’onglet Client.")
        st.stop()

    if (
        st.session_state.current_score is None
        or st.session_state.current_decision is None
    ):
        st.info(
            "Aucune décision calculée. Va dans l’onglet Client puis clique sur 'Calculer la décision'."
        )
        st.stop()

    if st.session_state.scored_client_id != st.session_state.current_client_id:
        st.info("Le client courant a changé. Clique sur 'Calculer la décision' dans l’onglet Client.")
        st.stop() 
    
    feats = expected_features()
    df = pd.DataFrame([st.session_state.current_client_features])[feats]

    try:
        df = coerce_df_types(df)
    except Exception as e:
        st.error("Erreur de typage sur les données du client.")
        st.exception(e)
        st.stop()

    p_default = st.session_state.current_score
    decision = st.session_state.current_decision

    # ----------------------------
    # Résultat principal
    # ----------------------------
    show_speedometer(p_default, threshold=THRESHOLD)

    if decision == "ACCORD":
        st.success("✅ Décision finale : ACCORD")
    else:
        st.error("⛔ Décision finale : REFUS")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### Résultat")
        st.write(f"**Client :** {st.session_state.current_client_id}")
        st.write(f"**Décision (seuil {THRESHOLD:.2f}) :** {decision}")

        level, tone = risk_level(p_default)
        if tone == "success":
            st.success(f"✅ {level}")
        elif tone == "warning":
            st.warning(f"⚠️ {level}")
        else:
            st.error(f"⛔ {level}")

        st.write(f"**Probabilité de défaut estimée :** {p_default:.2%}")
        st.caption("Barre sombre = risque estimé • Trait blanc = seuil de décision")

    with col_b:
        st.markdown("### Récapitulatif client")
        recap_cols = [
            "age",
            "revenu_mensuel",
            "poste",
            "statut_marital",
            "annee_experience_totale",
        ]
        recap_cols = [c for c in recap_cols if c in df.columns]
        st.dataframe(df[recap_cols], use_container_width=True)

        st.divider()
        st.markdown("## Visualisations")

        # ----------------------------
        # Population de référence
        # ----------------------------
        clients_df = load_clients()
        df_ref = clients_df.copy()

        if "client_id" in df_ref.columns:
            df_ref_no_id = df_ref.drop(columns=["client_id"])
        else:
            df_ref_no_id = df_ref.copy()

        try:
            df_ref_no_id = df_ref_no_id[feats]
            df_ref_no_id = coerce_df_types(df_ref_no_id)
            ref_scores = predict_proba_default(df_ref_no_id)
        except Exception:
            ref_scores = None

        # ============================
        # Graphique 1 : position du client
        # ============================
        st.markdown("### 1. Position du client dans la population de référence")

        if ref_scores is not None and len(ref_scores) > 0:
            fig_dist = go.Figure()

            fig_dist.add_trace(
                go.Histogram(
                    x=ref_scores,
                    nbinsx=15,
                    marker_color="#7ed321",
                    opacity=0.75,
                    name="Population de référence",
                )
            )

            fig_dist.add_vline(
                x=THRESHOLD,
                line_width=3,
                line_dash="dash",
                line_color="white",
                annotation_text=f"Seuil {THRESHOLD:.3f}",
                annotation_position="top left",
            )

            fig_dist.add_vline(
                x=p_default,
                line_width=4,
                line_dash="solid",
                line_color="red",
                annotation_text="Client courant",
                annotation_position="top right",
            )

            fig_dist.update_layout(
                xaxis_title="Probabilité de défaut",
                yaxis_title="Nombre de clients",
                bargap=0.1,
                height=350,
                showlegend=False,
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            if abs(p_default - THRESHOLD) < 0.05:
                st.warning(
                    "Le dossier est proche du seuil de décision : une légère amélioration du profil pourrait faire évoluer la décision."
                )
            elif p_default >= THRESHOLD:
                st.error("Le dossier se situe nettement dans la zone de refus.")
            else:
                st.success("Le dossier se situe sous le seuil de risque.")
        else:
            st.info("Impossible de calculer la distribution de référence.")

        # ============================
        # Graphique 2 : leviers d'amélioration
        # ============================
        st.markdown("### 2. Leviers d'amélioration du profil")

        actionable_vars = [
            "revenu_mensuel",
            "distance_domicile_travail",
            "annee_experience_totale",
            "nb_formations_suivies",
        ]
        actionable_vars = [
            c for c in actionable_vars if c in df.columns and c in df_ref_no_id.columns
        ]

        if actionable_vars:
            client_vals = df[actionable_vars].iloc[0]
            mean_vals = df_ref_no_id[actionable_vars].mean()

            fig_action = go.Figure()
            fig_action.add_trace(
                go.Bar(
                    x=actionable_vars,
                    y=client_vals.values,
                    name="Client",
                    marker_color="#0b1f2a",
                )
            )
            fig_action.add_trace(
                go.Bar(
                    x=actionable_vars,
                    y=mean_vals.values,
                    name="Référence moyenne",
                    marker_color="#f5a623",
                )
            )

            fig_action.update_layout(
                barmode="group",
                yaxis_title="Valeur",
                height=350,
            )

            st.plotly_chart(fig_action, use_container_width=True)

            st.caption(
                "Ce graphique met en évidence des variables potentiellement améliorables "
                "dans un cas proche du seuil. Il s'agit d'une aide à l'interprétation, "
                "pas d'une causalité directe du modèle."
            )
        else:
            st.info("Variables d'amélioration indisponibles.")

        # ============================
        # Graphique 3 : variables pénalisantes
        # ============================
        st.markdown("### 3. Principaux écarts défavorables du profil")

        explain_vars = [
            "revenu_mensuel",
            "distance_domicile_travail",
            "annee_experience_totale",
            "annees_dans_l_entreprise",
            "nb_formations_suivies",
        ]
        explain_vars = [
            c for c in explain_vars if c in df.columns and c in df_ref_no_id.columns
        ]

        if explain_vars:
            client_vals = df[explain_vars].iloc[0]
            mean_vals = df_ref_no_id[explain_vars].mean()

            # Écart relatif à la moyenne
            relative_gap = (client_vals - mean_vals) / mean_vals.replace(0, 1)

            # On réoriente certaines variables pour que "plus haut = plus défavorable"
            # revenu, expérience, formations : moins que la moyenne = défavorable
            # distance : plus que la moyenne = défavorable
            oriented_gap = relative_gap.copy()
            for col in [
                "revenu_mensuel",
                "annee_experience_totale",
                "nb_formations_suivies",
            ]:
                if col in oriented_gap.index:
                    oriented_gap[col] = -oriented_gap[col]

            # distance_domicile_travail reste telle quelle
            # age peut être laissé neutre/interprétatif

            oriented_gap = oriented_gap.sort_values(ascending=False)

            fig_explain = go.Figure()
            fig_explain.add_trace(
                go.Bar(
                    x=oriented_gap.values,
                    y=oriented_gap.index,
                    orientation="h",
                    marker_color="#d0021b",
                )
            )

            fig_explain.update_layout(
                xaxis_title="Écart défavorable relatif à la moyenne",
                yaxis_title="Variable",
                height=400,
            )

            st.plotly_chart(fig_explain, use_container_width=True)

            st.caption(
                "Plus la barre est élevée, plus la variable s'écarte défavorablement de la population de référence. "
                "Cela permet d'expliquer de manière synthétique pourquoi le dossier est pénalisé."
            )
        else:
            st.info("Variables explicatives indisponibles.")

# -------------------- TAB: MONITORING --------------------
with tab_monitoring:
    st.subheader("Monitoring de l'application")

    log_path = Path("logs") / "predictions.jsonl"

    if not log_path.exists():
        st.info("Aucune prédiction enregistrée pour le moment.")
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
            st.info("Logs vides.")
        else:
            df_logs = pd.DataFrame(rows)

            # Typage minimal utile
            if "ts" in df_logs.columns:
                df_logs["ts"] = pd.to_datetime(df_logs["ts"], errors="coerce")

            # ============================
            # KPI principaux
            # ============================
            st.markdown("### Indicateurs clés")

            k1, k2, k3, k4 = st.columns(4)

            with k1:
                st.metric("Décisions calculées", len(df_logs))

            with k2:
                if "session_id" in df_logs.columns:
                    st.metric("Nombre de sessions", df_logs["session_id"].nunique())
                else:
                    st.metric("Nombre de sessions", "N/A")

            with k3:
                if "p_default" in df_logs.columns:
                    st.metric("Risque moyen", f"{df_logs['p_default'].mean():.2%}")
                else:
                    st.metric("Risque moyen", "N/A")

            with k4:
                if "latency_ms" in df_logs.columns:
                    st.metric("Latence moyenne", f"{df_logs['latency_ms'].mean():.1f} ms")
                else:
                    st.metric("Latence moyenne", "N/A")

            st.divider()

            # ============================
            # Activité par session
            # ============================
            left, right = st.columns(2)

            with left:
                st.markdown("### Activité par session")

                if "session_id" in df_logs.columns:
                    session_counts = df_logs["session_id"].value_counts()

                    st.bar_chart(session_counts)

                    with st.expander("Détail activité par session"):
                        st.dataframe(
                            session_counts.rename_axis("session_id")
                            .reset_index(name="nb_decisions"),
                            use_container_width=True,
                        )
                else:
                    st.info("Aucun identifiant de session disponible dans les logs.")

            # ============================
            # Répartition des décisions
            # ============================
            with right:
                st.markdown("### Répartition des décisions")

                if "decision" in df_logs.columns:
                    decision_counts = df_logs["decision"].value_counts()

                    fig_decision = go.Figure(
                        data=[
                            go.Pie(
                                labels=decision_counts.index,
                                values=decision_counts.values,
                                hole=0.4,
                            )
                        ]
                    )
                    fig_decision.update_layout(height=350)
                    st.plotly_chart(fig_decision, use_container_width=True)
                else:
                    st.info("Aucune colonne `decision` dans les logs.")

            st.divider()

            # ============================
            # Activité dans le temps
            # ============================
            st.markdown("### Activité dans le temps")

            if "ts" in df_logs.columns and df_logs["ts"].notna().any():
                activity_df = (
                    df_logs.dropna(subset=["ts"])
                    .set_index("ts")
                    .resample("1min")
                    .size()
                    .rename("nb_decisions")
                    .reset_index()
                )

                if not activity_df.empty:
                    fig_activity = go.Figure()
                    fig_activity.add_trace(
                        go.Scatter(
                            x=activity_df["ts"],
                            y=activity_df["nb_decisions"],
                            mode="lines+markers",
                            name="Décisions",
                        )
                    )
                    fig_activity.update_layout(
                        xaxis_title="Temps",
                        yaxis_title="Nombre de décisions",
                        height=350,
                    )
                    st.plotly_chart(fig_activity, use_container_width=True)
                else:
                    st.info("Pas assez de données temporelles pour afficher l'activité.")
            else:
                st.info("Aucun timestamp exploitable dans les logs.")

            st.divider()

            # ============================
            # Evolution du risque
            # ============================
            st.markdown("### Évolution du risque estimé")

            if "p_default" in df_logs.columns:
                risk_df = df_logs.copy()
                risk_df["prediction_index"] = range(1, len(risk_df) + 1)

                fig_risk = go.Figure()
                fig_risk.add_trace(
                    go.Scatter(
                        x=risk_df["prediction_index"],
                        y=risk_df["p_default"],
                        mode="lines+markers",
                        name="Risque",
                    )
                )
                fig_risk.update_layout(
                    xaxis_title="Prédiction",
                    yaxis_title="Probabilité de défaut",
                    height=350,
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.info("Aucune colonne `p_default` dans les logs.")

            st.divider()

            # ============================
            # Latence
            # ============================
            st.markdown("### Latence des calculs")

            if "latency_ms" in df_logs.columns:
                latency_df = df_logs.copy()
                latency_df["prediction_index"] = range(1, len(latency_df) + 1)

                fig_latency = go.Figure()
                fig_latency.add_trace(
                    go.Scatter(
                        x=latency_df["prediction_index"],
                        y=latency_df["latency_ms"],
                        mode="lines+markers",
                        name="Latence (ms)",
                    )
                )
                fig_latency.update_layout(
                    xaxis_title="Prédiction",
                    yaxis_title="Latence (ms)",
                    height=350,
                )
                st.plotly_chart(fig_latency, use_container_width=True)
            else:
                st.info("Aucune colonne `latency_ms` dans les logs.")

            with st.expander("Voir les logs (50 derniers)"):
                st.dataframe(df_logs.tail(50), use_container_width=True)
                
# -------------------- TAB: ABOUT --------------------
with tab_about:
    st.subheader("Explications")

    st.write(
        """
Cette application illustre une chaîne MLOps simple autour d’un modèle de scoring :

- **Onglet Client** : sélection d’un client de démo et modification des informations.
- **Onglet Décision** : calcul du score de risque et prise de décision selon un seuil métier fixé à **0.50**.
- **Onglet Monitoring** : suivi des prédictions effectuées dans l’application.
- **Pipeline modèle** : `ColumnTransformer` (numériques + catégorielles) puis `LogisticRegression`.

### Points clés
- Le modèle retourne une **probabilité de défaut**.
- La décision est une **règle métier** appliquée à cette probabilité.
- Les types des colonnes sont validés automatiquement à partir du pipeline d’entraînement.
- Les clients de démonstration sont stockés dans un **CSV local**, sans base de données.
        """
    )
