# README – Projet 7 : Crédit Scoring & MLOps (OpenClassrooms)

## 🎯 Objectif du projet
Ce projet consiste à développer un système complet de scoring crédit, permettant d’estimer la probabilité de défaut d’un client pour l’entreprise *Prêt à Dépenser*.  
L’objectif englobe :

- Modélisation et optimisation d’un algorithme de classification  
- Prise en compte du coût métier (FN ≫ FP)  
- Mise en place d’un pipeline MLOps complet :  
  - tracking d’expériences  
  - tracking des modèles via MLflow
  - API de prédiction  
  - interface utilisateur Streamlit  
  - tests unitaires  
  - CI/CD via GitHub Actions  
  - monitoring du data drift via Evidently  

---

# 📁 Structure du repository

```
OC_p7/
│
├── api/
├── streamlit_app/
├── monitoring/
├── tests/
├── notebooks/
├── model/
├── .github/workflows/
├── requirements.txt
├── render.yaml
└── README.md
```

---

# 🔧 Modélisation & optimisation

- Feature engineering varié (baseline, domain, polynomial, feature tools)  
- Modèles testés : Logistic Regression, LightGBM, XGBoost  
- Optimisation basée sur le **business cost**  
- Optuna pour recherche d’hyperparamètres  
- Choix automatique du seuil optimal  
- CV stratifiée (K=5)  
- Gestion du déséquilibre (SMOTE, class_weight…)  
- Tracking + Model Registry : **MLflow**

---

# 🌐 API – Documentation des endpoints

## GET /health
Vérifie que l’API est opérationnelle.

### Réponse :
```json
{"status": "ok"}

```

---

## POST /predict
Calcule probabilité de défaut + décision crédit.

### Exemple d'entrée :
```json
{
  "EXT_SOURCE_1": 0.56,
  "EXT_SOURCE_2": 0.72,
  "EXT_SOURCE_3": 0.69,
  "REG_CITY_NOT_WORK_CITY": 1,
  "DAYS_ID_PUBLISH": -500,
  "DAYS_LAST_PHONE_CHANGE": -300,
  "REGION_RATING_CLIENT": 2,
  "REGION_RATING_CLIENT_W_CITY": 2,
  "DAYS_EMPLOYED": -2000,
  "DAYS_BIRTH": -12000
}

```

### Exemple de sortie :
```json
{
  "probability_default": 0.217,
  "prediction": 0,
  "threshold_used": 0.42,
  "business_cost_FN": 10000,
  "business_cost_FP": 500
}

```

---

## ⚙️ CI / CD

Une pipeline GitHub Actions est mise en place :

- Déclenchée à chaque `push` ou `pull_request`
- Exécute automatiquement :
  - installation des dépendances
  - lancement des tests unitaires (`pytest`)
- Mode `TESTING=1` :
  - chargement d’un DummyModel
  - pas de dépendance au modèle réel
  - CI rapide et fiable

Le déploiement est ensuite assuré automatiquement par Render.

---

## 🚀 Déploiement

L’API est déployée sur Render via :

- `render.yaml`
- `start.sh`
- `requirements.txt`

Commande de lancement :
```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```


---

# 🧪 Tests unitaires

```bash
pytest
```

---

---

## 📉 Monitoring – Data Drift

Une analyse de dérive des données est réalisée avec **Evidently** :

- comparaison :
  - données d'entraînement
  - données de production simulées
- nettoyage + imputation cohérente
- analyse uniquement sur les features du modèle
- génération automatique :
  - rapport HTML
  - résumé JSON

Objectif :
- détecter un changement de distribution
- anticiper une dégradation des performances
- décider d’un retraining

---

# 🏗️ Architecture du projet

```
Données → Feature engineering → Modélisation (MLflow)
                         ↓
                  API FastAPI (Render)
                         ↓
                   Interface Streamlit
                         ↓
                Monitoring (Evidently)

```

---

# 📦 Installation & exécution

## 1. Cloner
```bash
git clone https://github.com/sorenzemmour/OC_p7
cd OC_p7
```

## 2. Installer
```bash
pip install -r requirements.txt
```

## 3. Lancer l’API
```bash
uvicorn api.main:app --reload
```

## 4. Lancer Streamlit
```bash
streamlit run streamlit_app/app.py
```

---

# 🚀 Améliorations possibles
- Monitoring automatisé  
- Explicabilité avancée  
- Retraining automatique  

