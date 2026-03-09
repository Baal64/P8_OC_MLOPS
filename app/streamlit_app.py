import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.inference import expected_features, load_model, predict_proba_default

# ---------- CONFIG ----------
st.set_page_config(page_title="Scoring Crédit", layout="wide")
st.title("Scoring client – Demande de crédit")
st.caption("Démo MLOps • Streamlit + Docker • Déploiement Hugging Face")

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
            title={"text": "Risque de défaut", "font": {"size": 18}},
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


warmup()

if "current_client_features" not in st.session_state:
    st.session_state.current_client_features = None

if "current_client_id" not in st.session_state:
    st.session_state.current_client_id = None

if "current_score" not in st.session_state:
    st.session_state.current_score = None


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
    # Chargement du client par ID
    # ----------------------------
    default_client_id = st.session_state.current_client_id or "C001"
    client_id_input = st.text_input(
        "Identifiant client",
        value=str(default_client_id),
        key="client_id_search",
    )

    if st.button("Charger le client", key="load_client_button"):
        match = clients_df.loc[clients_df["client_id"].astype(str) == str(client_id_input).strip()]
        if match.empty:
            st.error(f"Client introuvable : {client_id_input}")
        else:
            selected_row = match.iloc[0]
            st.session_state.current_client_id = str(client_id_input).strip()
            st.session_state.current_client_features = client_to_feature_dict(selected_row, feats)
            st.success(f"Client {client_id_input} chargé.")

    if st.session_state.current_client_features is None:
        st.warning("Aucun client chargé. Saisis un identifiant puis clique sur 'Charger le client'.")
        st.stop()

    base_values = st.session_state.current_client_features.copy()

    # ----------------------------
    # Formulaire prérempli
    # ----------------------------
    col_left, col_right = st.columns(2)
    edited = dict(base_values)

    with col_left:
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

        st.markdown("### Situation")
        marital_options = ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve", "Autre"]
        marital_default = str(base_values.get("statut_marital", "Célibataire"))
        marital_index = marital_options.index(marital_default) if marital_default in marital_options else 0
        edited["statut_marital"] = st.selectbox(
            "Statut marital",
            marital_options,
            index=marital_index,
            key=f"statut_marital_input_{st.session_state.current_client_id}",
        )

        st.markdown("### Localisation / Poste")
        edited["departement"] = str(
            st.text_input(
                "Département",
                value=str(base_values.get("departement", "75")),
                key=f"departement_input_{st.session_state.current_client_id}",
            )
        )
        edited["poste"] = str(
            st.text_input(
                "Poste / Intitulé",
                value=str(base_values.get("poste", "Employé")),
                key=f"poste_input_{st.session_state.current_client_id}",
            )
        )

        st.markdown("### Revenus & charge")
        edited["revenu_mensuel"] = st.number_input(
            "Revenu mensuel (€)",
            min_value=0.0,
            value=float(base_values.get("revenu_mensuel", 2500.0)),
            step=100.0,
            key=f"revenu_input_{st.session_state.current_client_id}",
        )
        edited["distance_domicile_travail"] = st.number_input(
            "Distance domicile–travail (km)",
            min_value=0.0,
            value=float(base_values.get("distance_domicile_travail", 10.0)),
            step=1.0,
            key=f"distance_input_{st.session_state.current_client_id}",
        )

    with col_right:
        st.markdown("### Éducation / déplacements")
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
        travel_index = travel_options.index(travel_default) if travel_default in travel_options else 0
        edited["frequence_deplacement"] = st.selectbox(
            "Fréquence de déplacement",
            travel_options,
            index=travel_index,
            key=f"deplacement_input_{st.session_state.current_client_id}",
        )

        st.markdown("### Expérience / carrière")
        edited["nombre_experiences_precedentes"] = st.number_input(
            "Nombre d’expériences précédentes",
            min_value=0,
            max_value=50,
            value=int(base_values.get("nombre_experiences_precedentes", 2)),
            step=1,
            key=f"exp_prev_input_{st.session_state.current_client_id}",
        )
        edited["annee_experience_totale"] = st.number_input(
            "Années d’expérience totale",
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
            "Années dans le poste actuel",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annees_dans_le_poste_actuel", 2)),
            step=1,
            key=f"exp_poste_input_{st.session_state.current_client_id}",
        )
        edited["annes_sous_responsable_actuel"] = st.number_input(
            "Années sous le responsable actuel",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annes_sous_responsable_actuel", 2)),
            step=1,
            key=f"responsable_input_{st.session_state.current_client_id}",
        )
        edited["annees_depuis_la_derniere_promotion"] = st.number_input(
            "Années depuis la dernière promotion",
            min_value=0,
            max_value=60,
            value=int(base_values.get("annees_depuis_la_derniere_promotion", 1)),
            step=1,
            key=f"promotion_input_{st.session_state.current_client_id}",
        )

        st.markdown("### Organisation / formation")
        edited["niveau_hierarchique_poste"] = st.number_input(
            "Niveau hiérarchique du poste",
            min_value=1,
            max_value=10,
            value=int(base_values.get("niveau_hierarchique_poste", 2)),
            step=1,
            key=f"hierarchie_input_{st.session_state.current_client_id}",
        )
        edited["nb_formations_suivies"] = st.number_input(
            "Nombre de formations suivies",
            min_value=0,
            max_value=100,
            value=int(base_values.get("nb_formations_suivies", 1)),
            step=1,
            key=f"formations_input_{st.session_state.current_client_id}",
        )
        edited["nombre_participation_pee"] = st.number_input(
            "Nombre de participations PEE",
            min_value=0,
            max_value=100,
            value=int(base_values.get("nombre_participation_pee", 0)),
            step=1,
            key=f"pee_input_{st.session_state.current_client_id}",
        )

    st.divider()

    sat_cols = st.columns(2)
    with sat_cols[0]:
        st.markdown("### Satisfaction")
        edited["satisfaction_employee_environnement"] = st.slider(
            "Satisfaction environnement",
            1, 4, int(base_values.get("satisfaction_employee_environnement", 3)),
            key=f"sat_env_input_{st.session_state.current_client_id}",
        )
        edited["satisfaction_employee_nature_travail"] = st.slider(
            "Satisfaction nature du travail",
            1, 4, int(base_values.get("satisfaction_employee_nature_travail", 3)),
            key=f"sat_travail_input_{st.session_state.current_client_id}",
        )
        edited["satisfaction_employee_equilibre_pro_perso"] = st.slider(
            "Satisfaction équilibre pro/perso",
            1, 4, int(base_values.get("satisfaction_employee_equilibre_pro_perso", 3)),
            key=f"sat_eq_input_{st.session_state.current_client_id}",
        )

    with sat_cols[1]:
        st.markdown("### Performance / salaire")
        edited["satisfaction_employee_equipe"] = st.slider(
            "Satisfaction équipe",
            1, 4, int(base_values.get("satisfaction_employee_equipe", 3)),
            key=f"sat_equipe_input_{st.session_state.current_client_id}",
        )
        edited["note_evaluation_precedente"] = st.slider(
            "Note évaluation précédente",
            1, 5, int(base_values.get("note_evaluation_precedente", 3)),
            key=f"eval_prev_input_{st.session_state.current_client_id}",
        )
        edited["note_evaluation_actuelle"] = st.slider(
            "Note évaluation actuelle",
            1, 5, int(base_values.get("note_evaluation_actuelle", 3)),
            key=f"eval_current_input_{st.session_state.current_client_id}",
        )
        edited["augementation_salaire_precedente"] = st.number_input(
            "Augmentation salaire précédente (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(base_values.get("augementation_salaire_precedente", 5.0)),
            step=0.5,
            key=f"augmentation_input_{st.session_state.current_client_id}",
        )
        edited["heure_supplementaires"] = 1 if st.checkbox(
            "Heures supplémentaires",
            value=bool(base_values.get("heure_supplementaires", 0)),
            key=f"heures_sup_input_{st.session_state.current_client_id}",
        ) else 0

    if st.button("Enregistrer ce client pour la décision", type="primary", key="save_client_button"):
        df_current = pd.DataFrame([edited])[feats]
        try:
            df_current = coerce_df_types(df_current)
            st.session_state.current_client_features = df_current.iloc[0].to_dict()
            st.success("Client enregistré. Passe à l’onglet Décision.")
        except Exception as e:
            st.error("Erreur de typage sur les données du client.")
            st.exception(e)

# -------------------- TAB: DECISION --------------------
with tab_decision:
    st.subheader("Décision")

    if st.session_state.current_client_features is None:
        st.info("Aucun client sélectionné. Sélectionnez d'abord un client dans l’onglet Client.")
    else:
        feats = expected_features()
        df = pd.DataFrame([st.session_state.current_client_features])[feats]

        try:
            df = coerce_df_types(df)
            p_default = float(predict_proba_default(df).iloc[0])
            decision = "REFUS" if p_default >= THRESHOLD else "ACCORD"
            st.session_state.current_score = p_default
            save_prediction_log(p_default, decision, len(feats))
        except Exception as e:
            st.error("Erreur pendant le scoring.")
            st.exception(e)
            st.stop()

        show_speedometer(p_default, threshold=THRESHOLD)

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
            st.markdown("### Client courant")
            st.dataframe(df)

        st.divider()
        st.markdown("### Visualisations")

        g1, g2, g3 = st.columns(3)

        with g1:
            st.markdown("**Score vs seuil**")
            score_df = pd.DataFrame(
                {
                    "Mesure": ["Score client", "Seuil"],
                    "Valeur": [p_default, THRESHOLD],
                }
            )
            st.bar_chart(score_df.set_index("Mesure"))

        with g2:
            st.markdown("**Variables clés du client**")
            key_vars = ["age", "revenu_mensuel", "annee_experience_totale", "distance_domicile_travail"]
            key_vars = [c for c in key_vars if c in df.columns]
            if key_vars:
                st.bar_chart(df[key_vars].T.rename(columns={df.index[0]: "Valeur"}))
            else:
                st.info("Aucune variable clé disponible.")

        with g3:
            st.markdown("**Comparaison à la population de démo**")
            clients_df = load_clients()
            compare_var = "revenu_mensuel"
            if compare_var in clients_df.columns and compare_var in df.columns:
                compare_df = pd.DataFrame(
                    {
                        "Référence": [clients_df[compare_var].mean()],
                        "Client": [float(df[compare_var].iloc[0])],
                    }
                )
                st.bar_chart(compare_df.T.rename(columns={0: compare_var}))
            else:
                st.info("Comparaison indisponible.")

# -------------------- TAB: MONITORING --------------------
with tab_monitoring:
    st.subheader("Monitoring")

    log_path = Path("logs") / "predictions.jsonl"
    if not log_path.exists():
        st.info("Aucun log pour l’instant. Lance une décision dans l’onglet Décision.")
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

            c1, c2, c3 = st.columns(3)

            with c1:
                if "p_default" in df_logs.columns:
                    st.write("**Historique du risque**")
                    st.line_chart(df_logs["p_default"])

            with c2:
                if "decision" in df_logs.columns:
                    st.write("**Distribution des décisions**")
                    st.bar_chart(df_logs["decision"].value_counts())

            with c3:
                if "n_features" in df_logs.columns:
                    st.write("**Nombre de features utilisées**")
                    st.line_chart(df_logs["n_features"])

            with st.expander("Voir les logs"):
                st.dataframe(df_logs.tail(50))

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