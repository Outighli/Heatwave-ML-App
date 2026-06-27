# 🌡️ HeatWave AI - Système Intelligent de Prédiction et d'Alerte Précoce des Canicules

## 📋 Description

Système de Machine Learning pour la prédiction et l'alerte précoce des canicules à partir de données climatiques historiques (2010-2020).

## 🎯 Objectifs

- Prédire l'apparition des canicules
- Alerter les populations vulnérables
- Visualiser les données climatiques
- Expliquer les prédictions du modèle IA (SHAP)

## 📊 Données du Projet

- **Observations** : 343 510
- **Villes** : 101
- **Régions** : 7
- **Période** : 2010-2020
- **Variables explicatives** : 36
- **Variable cible** : `is_heatwave`

## 🏗️ Architecture

```
heatwave_prediction_system/
├── app.py                  # Application Flask principale
├── requirements.txt        # Dépendances Python
├── static/
│   ├── css/
│   │   └── style.css      # Styles Glassmorphism + Light Mode
│   ├── js/                # Scripts JavaScript (si nécessaire)
│   └── images/            # Images et assets
├── templates/
│   ├── base.html          # Template de base (layout)
│   ├── dashboard.html     # Page 1 : Dashboard
│   ├── prediction.html    # Page 2 : Prédiction IA
│   ├── analytics.html     # Page 3 : Analyse des Données
│   └── alerts.html        # Page 4 : Centre d'Alertes
├── model/                 # Modèles ML sauvegardés
└── database/              # Données et datasets
```

## 🚀 Installation

```bash
# 1. Cloner le projet
cd heatwave_prediction_system

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

## 🎨 Design

- **Style** : Glassmorphism + Light Mode
- **Couleurs** : Bleu #2563EB, Orange #F59E0B, Rouge #DC2626
- **Animations** : Modernes avec transitions fluides
- **Responsive** : Adapté aux écrans de toutes tailles

## 📄 Pages

1. **Dashboard** : Vue d'ensemble avec KPI, météo, prédiction IA, carte mondiale, graphiques
2. **Prédiction IA** : Formulaire de prédiction avec jauge circulaire et explication SHAP
3. **Analyse des Données** : Distribution, corrélations, évolution climatique, statistiques régionales
4. **Centre d'Alertes** : Alertes actives, notifications, historique, paramètres

## 🔧 Technologies

- **Backend** : Python / Flask
- **Frontend** : HTML5, CSS3, JavaScript
- **Visualisation** : Plotly.js, Leaflet.js
- **Icons** : Font Awesome 6
- **ML** : scikit-learn, SHAP

## 👨‍💻 Auteur

Projet de fin d'études en Intelligence Artificielle
