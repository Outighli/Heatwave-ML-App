# 🌡️ HeatWave AI - Système Intelligent de Prédiction et d'Alerte Précoce des Canicules

## 📋 Description
Système de Machine Learning pour la prédiction et l'alerte précoce des canicules 
à partir de données climatiques historiques (2010-2020), développé dans le cadre 
du Master Intelligence Artificielle — Faculté des Sciences Ben M'Sick, 
Université Hassan II de Casablanca.

## 🎯 Objectifs
- Prédire l'apparition des canicules (Classification binaire is_heatwave)
- Prédire la température maximale à J+3 (Régression temp_max_J3)
- Alerter les populations vulnérables via un Centre d'Alertes
- Visualiser les données climatiques historiques (2010–2020)
- Expliquer les prédictions du modèle IA (Feature Importance Random Forest)

---

## 🖥️ Interface de l'Application

### Dashboard Principal
![Dashboard](static/images/screens/dashboard.png)
> Vue d'ensemble avec KPI, conditions météo par ville et carte mondiale

### Prédiction IA
![Prediction](static/images/screens/prediction.png)
> Formulaire de prédiction avec jauge circulaire et Feature Importance

### Analyse des Données
![Analytics](static/images/screens/analytics.png)
> EDA interactif — distributions, tendances climatiques, saisonnalité

### Centre d'Alertes
![Alerts](static/images/screens/alerts.png)
> Alertes actives et notifications par ville et région

### Rapports
![Reports](static/images/screens/reports.png)
> Génération et export PDF/CSV des statistiques détaillées

---

## 📊 Données du Projet
- **Observations** : 342 978
- **Villes** : 100
- **Régions** : 7
- **Période** : 2010–2020
- **Features** : 13 variables construites par feature engineering
- **Variable cible classification** : `is_heatwave` (seuil 35°C OMM)
- **Variable cible régression** : `temp_max_J3`
- **Taux de canicule** : 5.16% (ratio 1:18)

---

## 🤖 Modèles ML
| Tâche | Modèle | Métrique | Valeur |
|---|---|---|---|
| Classification | Random Forest | ROC-AUC | 0.9959 |
| Classification | Random Forest | F1-Score | 0.8476 |
| Classification | Random Forest | Precision | 0.9095 |
| Classification | Random Forest | Recall | 0.7937 |
| Régression | XGBoost | MAE | 2.332°C |
| Régression | XGBoost | RMSE | 3.199°C |
| Régression | XGBoost | R² | 0.8899 |

PROJET_ML/

├── app.py                          # Application Flask principale

├── requirements.txt                # Dépendances Python

├── README.md                       # Documentation

├── .gitignore

├── static/

│   ├── css/

│   │   └── style.css              # Styles Glassmorphism + Light Mode

│   └── images/

│       └── screens/               # Screenshots de l'interface

│           ├── dashboard.png

│           ├── prediction.png

│           ├── analytics.png

│           ├── alerts.png

│           └── reports.png

├── templates/

│   ├── base.html                  # Template de base (layout)

│   ├── dashboard.html             # Page 1 : Dashboard

│   ├── prediction.html            # Page 2 : Prédiction IA

│   ├── analytics.html             # Page 3 : Analyse des Données

│   ├── alerts.html                # Page 4 : Centre d'Alertes

│   └── reports.html               # Page 5 : Rapports

├── model/

│   ├── best_classifier.pkl        # Random Forest (classification)

│   ├── best_regressor.pkl         # XGBoost (régression J+3)

│   ├── scaler.pkl                 # StandardScaler

│   ├── le_region.pkl              # LabelEncoder régions

│   ├── le_city.pkl                # LabelEncoder villes

│   ├── model_metadata.json        # Métriques et features

│   └── feature_engineering_metadata.json

└── database/

├── heatwave_features.parquet  # Dataset enrichi (features)

├── heatwave_final.parquet     # Dataset avec is_heatwave

├── heatwave_merged.parquet    # Dataset fusionné nettoyé

└── city_temperature.csv       # Source brute Kaggle
---

## 🚀 Installation Locale

```bash
# 1. Cloner le projet
git clone <url_du_projet>
cd PROJET_ML

# 2. Créer l'environnement virtuel
python -m venv venv

# 3. Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Lancer l'application
python app.py
```

---

## ☁️ Déploiement AWS EC2

### 🔗 Application en ligne
**URL de production** : `http://<adresse-ip-publique-ec2>:5000`

> Remplacer `<adresse-ip-publique-ec2>` par l'IP publique de votre instance

---

### Étapes du Déploiement

#### 1. Création d'une instance Amazon EC2
Création d'une machine virtuelle Ubuntu sur Amazon Web Services 
pour héberger l'application.
Type     : t2.micro (Free Tier)

OS       : Ubuntu Server 22.04 LTS

Stockage : 20 GB SSD

Ports    : 22 (SSH), 5000 (Flask), 80 (HTTP)
#### 2. Configuration de l'environnement serveur
Mise à jour du système et installation des outils nécessaires.
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git python3 python3-pip python3-venv
```

#### 3. Récupération du projet
Clonage du projet depuis GitHub vers l'instance EC2.
```bash
git clone <url_du_projet>
cd PROJET_ML
```

#### 4. Gestion des fichiers volumineux avec Git LFS
Installation de Git LFS pour récupérer les modèles `.pkl` 
et les datasets volumineux.
```bash
sudo apt install git-lfs -y
git lfs install
git lfs pull
```

#### 5. Création de l'environnement virtuel Python
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 6. Installation des dépendances
```bash
pip install -r requirements.txt
```

#### 7. Chargement et vérification des modèles
Vérification du chargement des modèles, encodeurs, scaler et dataset.
```bash
python -c "
import joblib, json
clf = joblib.load('model/best_classifier.pkl')
reg = joblib.load('model/best_regressor.pkl')
print('✅ Classificateur :', type(clf).__name__)
print('✅ Régresseur     :', type(reg).__name__)
"
```

#### 8. Déploiement de l'application Flask
Lancement et vérification via l'adresse IP publique.
```bash
python app.py
# Accès : http://<ip-publique-ec2>:5000
```

#### 9. Déploiement en production avec Gunicorn
Utilisation de Gunicorn pour un environnement stable et production.
```bash
pip install gunicorn
gunicorn --workers 3 --bind 0.0.0.0:5000 app:app
```

#### 10. Configuration d'un service système (systemd)
Service permettant le démarrage automatique et le redémarrage en cas d'arrêt.
```bash
sudo nano /etc/systemd/system/heatwave.service
```
```ini
[Unit]
Description=HeatWave AI Flask App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/PROJET_ML
Environment="PATH=/home/ubuntu/PROJET_ML/venv/bin"
ExecStart=/home/ubuntu/PROJET_ML/venv/bin/gunicorn \
          --workers 3 \
          --bind 0.0.0.0:5000 \
          app:app
Restart=always

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable heatwave
sudo systemctl start heatwave
sudo systemctl status heatwave
```

#### 11. Tests de validation
Vérification de l'ensemble des fonctionnalités.
```bash
# Test API santé
curl http://localhost:5000/api/health

# Test prédiction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "Paris", "temp_max": 38, "humidity": 25}'
```

#### 12. Accès à l'application
🌍 URL publique : http://51.21.149.17:5000
---

## 📄 Pages
1. **Dashboard** : Vue d'ensemble avec KPI, conditions météo par ville,
   prédiction IA temps réel et carte mondiale des températures
2. **Prédiction IA** : Formulaire de prédiction avec jauge circulaire,
   probabilité de canicule, température J+3 et Feature Importance
3. **Analyse des Données** : EDA interactif — distributions, tendances
   climatiques, top villes, saisonnalité, relation Temp × Humidité
4. **Centre d'Alertes** : Alertes actives, notifications Email/SMS/Push,
   historique des alertes par ville et région
5. **Rapports** : Génération et export PDF/CSV des statistiques détaillées

---

## 🔧 Technologies
- **Backend** : Python 3.12 / Flask
- **Frontend** : HTML5, CSS3, JavaScript
- **ML** : scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE)
- **Data** : pandas, numpy, pyarrow (parquet)
- **Interprétabilité** : SHAP, Feature Importance
- **Déploiement** : AWS EC2, Gunicorn, systemd

---

## 🎨 Design
- **Style** : Glassmorphism + Light Mode
- **Couleurs** : Bleu #2563EB, Orange #F59E0B, Rouge #DC2626
- **Animations** : Transitions fluides et modernes
- **Responsive** : Adapté aux écrans de toutes tailles

---

## 👥 Équipe
- **Ouarrak Layla**
- **Outighli Sanae**

**Encadrants** : Pr. BENLAHMAR ELHABIB | Pr. Oussama Kaich

**Année Universitaire** : 2025–2026

---

## 🏗️ Architecture
