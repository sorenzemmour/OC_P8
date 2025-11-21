# 📘 API de Scoring Client — Projet 7 OpenClassrooms

Cette API FastAPI permet de servir un modèle de scoring de crédit entraîné dans le cadre du projet 7 du parcours Data Scientist OpenClassrooms.  
Elle retourne une probabilité de défaut ainsi qu’une décision binaire basée sur un seuil optimisé par validation croisée.

---

## 🏗️ Structure du dossier

```
api/
│── main.py                     # Point d'entrée FastAPI (définition des endpoints)
│── __init__.py
│
├── model/
│   │── loader.py               # Chargement du modèle depuis MLflow ou fallback local
│   └── __init__.py
│
├── schemas/
│   │── input_schema.py         # Schéma Pydantic des features d'entrée
│   └── __init__.py
│
├── utils/
│   │── business_cost.py        # Définition du coût métier FN / FP
│   └── __init__.py
│
└── tests/
    │── test_api.py             # Tests unitaires de l’API
    └── conftest.py             # Ajout du chemin racine au PYTHONPATH pour pytest
```

---

## 🎯 Objectifs de l’API

- Charger automatiquement le meilleur modèle produit par MLflow (ou un fallback local si nécessaire)
- Exposer un endpoint de prédiction `/predict`
- Exposer un endpoint de santé `/health`
- Valider les données entrantes grâce à Pydantic
- Intégrer le seuil de décision optimisé
- Fournir les valeurs de coûts métier FN / FP
- Servir de moteur d’inférence prêt pour le déploiement Cloud

---

## 🚀 Lancer l’API en local

```
uvicorn api.main:app --reload
```

URL de base :

```
http://127.0.0.1:8000
```

Documentation Swagger :

```
http://127.0.0.1:8000/docs
```

---

## 🔍 Endpoints

### ✔️ GET /health

```json
{
  "status": "ok"
}
```

### ✔️ POST /predict

Entrée :

```json
{
  "EXT_SOURCE_3": 0.45,
  "EXT_SOURCE_2": 0.62,
  "EXT_SOURCE_1": 0.75,
  "REG_CITY_NOT_WORK_CITY": 1,
  "DAYS_ID_PUBLISH": -500,
  "DAYS_LAST_PHONE_CHANGE": -300.5,
  "REGION_RATING_CLIENT": 2,
  "REGION_RATING_CLIENT_W_CITY": 2,
  "DAYS_EMPLOYED": -2000,
  "DAYS_BIRTH": -12000
}
```

Sortie :

```json
{
  "probability_default": 0.1234,
  "prediction": 0,
  "threshold_used": 0.42,
  "business_cost_FN": 10000,
  "business_cost_FP": 500
}
```

---

## 🧪 Tests unitaires

```
pytest -q
```

---

## 📦 Dépendances principales

fastapi, uvicorn, pydantic, mlflow, scikit-learn, numpy, httpx, pytest

---

## 📄 Rôle du dossier API

Découplage complet de la modélisation, moteur d’inférence uniquement.
