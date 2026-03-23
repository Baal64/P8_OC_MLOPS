---
title: P8 OC MLOps - Scoring
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Projet 8 – Mise en production d’un modèle de scoring (MLOps)

Ce projet s’inscrit dans le cadre de la formation **Data Scientist – Machine Learning** d’OpenClassrooms.  
Il a pour objectif de mettre en production un modèle de Machine Learning existant en appliquant les bonnes pratiques **MLOps**.

Le projet repose sur un modèle de scoring développé lors d’un projet précédent et vise à construire un environnement complet incluant :
- une API de prédiction,
- des tests automatisés,
- une conteneurisation avec Docker,
- une chaîne CI/CD,
- des mécanismes de monitoring et d’analyse de dérive des données,
- une analyse des performances et des optimisations.

---

## 🎯 Objectifs du projet

- Mettre à disposition un modèle de Machine Learning via une API
- Garantir la qualité du code grâce aux tests automatisés
- Automatiser l’intégration et le déploiement (CI/CD)
- Assurer le suivi des données et des performances en production
- Appliquer une démarche MLOps reproductible et documentée

---

## 🧱 Structure du projet

P8_OC_MLOPS/
├── api/ # API FastAPI (endpoints, schémas)
├── src/ # Logique métier (inférence, utilitaires, monitoring)
├── tests/ # Tests unitaires et d’intégration
├── docker/ # Fichiers Docker
├── notebooks/ # Analyses (drift, exploration, reporting)
├── data/
│ ├── reference/ # Données de référence
│ └── production/ # Données/logs de production
├── .github/workflows/ # Pipeline CI/CD
├── requirements.txt # Dépendances Python
├── .gitignore
└── README.md

---

## 🛠️ Technologies utilisées

- Python
- FastAPI
- Pytest
- Docker
- GitHub Actions
- Outils de monitoring et d’analyse de dérive (à définir)

---

## 🚀 Déploiement (Hugging Face Spaces)

L’application est déployée sur **Hugging Face Spaces** via un **Space Docker**.

La synchronisation est automatisée :
- **GitHub** est la source de vérité.
- Une **GitHub Action** pousse automatiquement le code vers le Space Hugging Face à chaque `push` sur la branche `main`.
- Le Space Hugging Face reconstruit et redéploie l’application automatiquement après chaque synchronisation.

> Les informations sensibles (token Hugging Face) sont gérées via **GitHub Secrets**.

---

## Accès au modèle

Le modèle est accessible via l'application Streamlit.

Dans un contexte industriel, une API REST pourrait être utilisée, mais dans ce projet l'inférence est directement intégrée à l'interface.

## Monitoring

Les prédictions sont enregistrées dans un fichier `predictions.jsonl`, permettant le suivi des performances et l'analyse du data drift.

## Data Drift

Une analyse du data drift est réalisée dans un notebook dédié, avec utilisation de la bibliothèque Evidently.

---

## 👤 Auteur

Projet réalisé par **Alexandre Ba** dans le cadre de la formation OpenClassrooms.