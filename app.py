from flask import Flask, render_template, jsonify, request
import json
import random
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'database')

# ============================================================
# MÉTADONNÉES & MODÈLES
# ============================================================
FEATURES = []
BEST_CLF_NAME = 'Random Forest'
BEST_REG_NAME = 'XGBoost'
DECISION_THRESHOLD = 0.88
TEST_METRICS_CLF = {}
TEST_METRICS_REG = {}
REGION_MAP = {}
SEASON_MAP = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}

try:
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    with open(metadata_path, 'r') as f:
        METADATA = json.load(f)
    FEATURES = METADATA.get('features', [])
    BEST_CLF_NAME = METADATA.get('best_clf_name', BEST_CLF_NAME)
    BEST_REG_NAME = METADATA.get('best_reg_name', BEST_REG_NAME)
    DECISION_THRESHOLD = METADATA.get('decision_threshold', DECISION_THRESHOLD)
    TEST_METRICS_CLF = METADATA.get('test_metrics_clf', TEST_METRICS_CLF)
    TEST_METRICS_REG = METADATA.get('test_metrics_reg', TEST_METRICS_REG)
    REGION_MAP = METADATA.get('region_map', {})
    print(f"✅ Métadonnées chargées — {len(FEATURES)} features")
except Exception as e:
    print(f"⚠️ Erreur métadonnées: {e}")

best_clf = None
best_reg = None
scaler = None
le_city = None
le_region = None
MODELS_LOADED = False

# ============================================================
# CHARGEMENT DU CLASSIFICATEUR
# ============================================================
try:
    best_clf = joblib.load(os.path.join(MODEL_DIR, 'best_classifier.pkl'))
    MODELS_LOADED = True
    print(f"✅ Classificateur chargé : {BEST_CLF_NAME}")
    print(f"📊 Type du classificateur : {type(best_clf)}")
except Exception as e:
    print(f"⚠️ Classificateur: {e}")

# ============================================================
# CHARGEMENT DU RÉGRESSEUR (AMÉLIORÉ)
# ============================================================
try:
    reg_path = os.path.join(MODEL_DIR, 'best_regressor.pkl')
    if os.path.exists(reg_path):
        best_reg = joblib.load(reg_path)
        print(f"✅ Régresseur chargé : {BEST_REG_NAME}")
        print(f"📊 Type du régresseur : {type(best_reg)}")
        
        # Test de prédiction avec des données factices
        if best_reg is not None and len(FEATURES) > 0:
            try:
                test_data = np.random.randn(1, len(FEATURES))
                test_pred = best_reg.predict(test_data)
                print(f"🧪 Test prédiction régresseur : {test_pred[0]:.2f}")
            except Exception as e:
                print(f"⚠️ Test régresseur échoué: {e}")
    else:
        print(f"⚠️ Fichier régresseur introuvable: {reg_path}")
except Exception as e:
    print(f"⚠️ Régresseur: {e}")

# ============================================================
# CHARGEMENT DU SCALER
# ============================================================
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✅ Scaler chargé")
except Exception as e:
    print(f"⚠️ Scaler: {e}")

# ============================================================
# CHARGEMENT DES LABEL ENCODERS
# ============================================================
try:
    le_city = joblib.load(os.path.join(MODEL_DIR, 'le_city.pkl'))
    print("✅ LabelEncoder City chargé")
except Exception as e:
    print(f"⚠️ LabelEncoder City: {e}")

try:
    le_region = joblib.load(os.path.join(MODEL_DIR, 'le_region.pkl'))
    print("✅ LabelEncoder Region chargé")
except Exception as e:
    print(f"⚠️ LabelEncoder Region: {e}")

print(f"📊 Modèles chargés: Classif={best_clf is not None}, Reg={best_reg is not None}, Scaler={scaler is not None}")

# ============================================================
# COORDONNÉES DES VILLES POUR LA CARTE
# ============================================================
CITY_COORDS = {
    'Paris': [48.8566, 2.3522],
    'Marseille': [43.2965, 5.3698],
    'Lyon': [45.7640, 4.8357],
    'Toulouse': [43.6047, 1.4442],
    'Nice': [43.7102, 7.2620],
    'Nantes': [47.2184, -1.5536],
    'Strasbourg': [48.5734, 7.7521],
    'Montpellier': [43.6108, 3.8767],
    'Bordeaux': [44.8378, -0.5792],
    'Lille': [50.6292, 3.0573],
    'Doha': [25.2854, 51.5310],
    'Dubai': [25.2048, 55.2708],
    'Tokyo': [35.6762, 139.6503],
    'New York': [40.7128, -74.0060],
    'London': [51.5074, -0.1278],
    'Rome': [41.9028, 12.4964],
    'Madrid': [40.4168, -3.7038],
    'Berlin': [52.5200, 13.4050],
    'Cairo': [30.0444, 31.2357],
    'Cape Town': [-33.9249, 18.4241],
    'Sydney': [-33.8688, 151.2093],
    'Moscow': [55.7558, 37.6173],
    'Beijing': [39.9042, 116.4074],
    'Mexico City': [19.4326, -99.1332],
    'Sao Paulo': [-23.5505, -46.6333],
    'Istanbul': [41.0082, 28.9784],
    'Bangkok': [13.7563, 100.5018],
    'Singapore': [1.3521, 103.8198],
    'Hong Kong': [22.3193, 114.1694],
    'Seoul': [37.5665, 126.9780],
    'Mumbai': [19.0760, 72.8777],
    'Lagos': [6.5244, 3.3792],
    'Nairobi': [-1.2921, 36.8219],
    'Johannesburg': [-26.2041, 28.0473],
    'Casablanca': [33.5731, -7.5898],
    'Tunis': [36.8065, 10.1815],
    'Algiers': [36.7538, 3.0588],
    'Athens': [37.9838, 23.7275],
    'Lisbon': [38.7223, -9.1393],
    'Dublin': [53.3498, -6.2603],
    'Oslo': [59.9139, 10.7522],
    'Stockholm': [59.3293, 18.0686],
    'Helsinki': [60.1699, 24.9384],
    'Warsaw': [52.2297, 21.0122],
    'Prague': [50.0755, 14.4378],
    'Vienna': [48.2082, 16.3738],
    'Budapest': [47.4979, 19.0402],
    'Bucharest': [44.4268, 26.1025],
    'Sofia': [42.6977, 23.3219],
    'Belgrade': [44.7866, 20.4489],
    'Zagreb': [45.8150, 15.9819],
    'Ljubljana': [46.0569, 14.5058],
    'Bratislava': [48.1486, 17.1077],
    'Tallinn': [59.4370, 24.7536],
    'Riga': [56.9496, 24.1052],
    'Vilnius': [54.6872, 25.2797],
    'Minsk': [53.9045, 27.5615],
    'Kyiv': [50.4501, 30.5234],
    'Tbilisi': [41.7151, 44.8271],
    'Yerevan': [40.1792, 44.4991],
    'Baku': [40.4093, 49.8671],
    'Tashkent': [41.2995, 69.2401],
    'Almaty': [43.2220, 76.8512],
    'Ulaanbaatar': [47.8864, 106.9057],
    'Kathmandu': [27.7172, 85.3240],
    'Colombo': [6.9271, 79.8612],
    'Kuala Lumpur': [3.1390, 101.6869],
    'Jakarta': [-6.2088, 106.8456],
    'Manila': [14.5995, 120.9842],
    'Taipei': [25.0330, 121.5654],
    'Shanghai': [31.2304, 121.4737],
    'Guangzhou': [23.1291, 113.2644],
    'Shenzhen': [22.5431, 114.0579],
    'Chongqing': [29.5630, 106.5516],
    'Chengdu': [30.5728, 104.0668],
    'Wuhan': [30.5928, 114.3055],
    'Nanjing': [32.0603, 118.7969],
    'Hangzhou': [30.2741, 120.1551],
    'Fuzhou': [26.0745, 119.2965],
    'Xiamen': [24.4798, 118.0894],
    'Qingdao': [36.0671, 120.3826],
    'Dalian': [38.9140, 121.6147],
    'Tianjin': [39.0842, 117.2009],
    'Shenyang': [41.8057, 123.4315],
    'Harbin': [45.8038, 126.5350],
    'Changchun': [43.8868, 125.3245],
    'Jilin': [43.8379, 126.5496],
    'Lanzhou': [36.0611, 103.8343],
    'Xi\'an': [34.3416, 108.9398],
    'Zhengzhou': [34.7466, 113.6253],
    'Jinan': [36.6512, 117.1201],
    'Hefei': [31.8206, 117.2272],
    'Nanchang': [28.6820, 115.8579],
    'Changsha': [28.2282, 112.9388],
    'Guiyang': [26.5783, 106.7135],
    'Kunming': [25.0409, 102.7123],
    'Nanning': [22.8176, 108.3669],
    'Haikou': [20.0444, 110.1999],
    'Urumqi': [43.8256, 87.6168],
    'Lhasa': [29.6500, 91.1000],
    'Yangon': [16.8409, 96.1735],
    'Phnom Penh': [11.5564, 104.9282],
    'Vientiane': [17.9757, 102.6331],
    'Hanoi': [21.0278, 105.8342],
    'Ho Chi Minh City': [10.8231, 106.6297],
}

# ============================================================
# DATASET → heatwave_deployment.csv
# ============================================================
df = None
DATASET_LOADED = False

try:
    file_path = os.path.join(DATA_DIR, 'heatwave_deployment.csv')
    print(f"🔍 Dataset : {file_path}")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['City_clean', 'date']).reset_index(drop=True)
        DATASET_LOADED = True
        print(f"✅ Dataset : {df.shape[0]:,} lignes, {df['City_clean'].nunique()} villes")
    else:
        print(f"⚠️ Fichier dataset introuvable: {file_path}")
except Exception as e:
    print(f"⚠️ Dataset: {e}")

print(f"📊 Dataset={DATASET_LOADED}, Modèles={MODELS_LOADED}")


# ============================================================
# FEATURE ENGINEERING — Construction des 13 features
# ============================================================

def get_season(month):
    """Retourne la saison en fonction du mois."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def get_season_encoded(month):
    """Retourne l'encodage numérique de la saison."""
    season = get_season(month)
    return SEASON_MAP.get(season, 0)

def build_features_from_input(data: dict) -> pd.DataFrame:
    """
    Construit les 13 features à partir des données du formulaire.
    Utilise l'historique du dataset pour les lags si disponible.
    """
    city = data.get('city', '')
    temp_max = float(data.get('temp_max', 30))
    temp_min = float(data.get('temp_min', 20))
    humidity = float(data.get('humidity', 50))
    wind_kph = float(data.get('wind_kph', 10))
    precip_mm = float(data.get('precip_mm', 0))
    
    # Date actuelle pour les features temporelles
    now = datetime.now()
    day_of_year = now.timetuple().tm_yday
    month = now.month
    
    # Récupérer l'historique de la ville pour les lags
    city_history = None
    if DATASET_LOADED and df is not None and city:
        city_df = df[df['City_clean'] == city].sort_values('date')
        if not city_df.empty:
            city_history = city_df
    
    # Calcul des lags temp_max
    if city_history is not None and len(city_history) >= 3:
        # Utiliser les données réelles du dataset
        last_rows = city_history.tail(3)
        temp_max_lag1 = float(last_rows.iloc[-1]['temp_max'])
        temp_max_lag2 = float(last_rows.iloc[-2]['temp_max'])
        temp_max_lag3 = float(last_rows.iloc[-3]['temp_max'])
        
        # Récupérer les 7 derniers jours pour la moyenne mobile
        last_7 = city_history.tail(7)
        temp_max_roll7 = float(last_7['temp_max'].mean())
    else:
        # Simulation des lags à partir de la température actuelle
        temp_max_lag1 = temp_max - random.uniform(0, 3)
        temp_max_lag2 = temp_max - random.uniform(1, 5)
        temp_max_lag3 = temp_max - random.uniform(2, 7)
        temp_max_roll7 = temp_max - random.uniform(0.5, 4)
    
    # Calcul de l'amplitude thermique
    temp_range = temp_max - temp_min
    
    # Encodage de la ville et de la région
    city_encoded = 0
    region_encoded = 0
    
    if city_history is not None and not city_history.empty:
        # Utiliser les encodages existants du dataset
        last_row = city_history.iloc[-1]
        city_encoded = float(last_row.get('City_encoded', 0))
        region_encoded = float(last_row.get('Region_encoded', 0))
    else:
        # Essayer d'utiliser les LabelEncoders
        if le_city is not None and city in le_city.classes_:
            city_encoded = float(le_city.transform([city])[0])
        if le_region is not None:
            region = data.get('region', 'Europe')
            if region in le_region.classes_:
                region_encoded = float(le_region.transform([region])[0])
            elif 'Europe' in le_region.classes_:
                region_encoded = float(le_region.transform(['Europe'])[0])
    
    # Encodage de la saison
    season_encoded = get_season_encoded(month)
    
    # Construction du dictionnaire des features
    features_dict = {
        'temp_max_lag1': temp_max_lag1,
        'temp_max_lag2': temp_max_lag2,
        'temp_max_lag3': temp_max_lag3,
        'temp_max_roll7': temp_max_roll7,
        'temp_range': temp_range,
        'humidity': humidity,
        'wind_kph': wind_kph,
        'precip_mm': precip_mm,
        'day_of_year': day_of_year,
        'month': month,
        'season': season_encoded,
        'Region_encoded': region_encoded,
        'City_encoded': city_encoded
    }
    
    # Créer le DataFrame avec les colonnes dans l'ordre exact
    df_features = pd.DataFrame([features_dict])[FEATURES]
    
    return df_features

def build_features_from_history(city_name: str):
    """
    Construit les 13 features à partir de l'historique du dataset.
    """
    if not DATASET_LOADED or df is None:
        return None

    city_df = df[df['City_clean'] == city_name].copy()
    if city_df.empty:
        return None

    last = city_df.sort_values('date').iloc[-1]

    row = {}
    for feat in FEATURES:
        val = last.get(feat, 0.0)
        if pd.isna(val):
            val = 0.0
        row[feat] = float(val)

    return pd.DataFrame([row], columns=FEATURES)


# ============================================================
# DONNÉES DASHBOARD
# ============================================================

def get_dataset_info():
    if not DATASET_LOADED:
        return {
            'total_observations': 0, 'total_cities': 0, 'total_regions': 7,
            'total_heatwaves': 0, 'heatwave_rate': 0, 'max_temperature': 0,
            'period': 'N/A', 'features_count': len(FEATURES)
        }
    return {
        'total_observations': len(df),
        'total_cities': df['City_clean'].nunique(),
        'total_regions': df['Region'].nunique(),
        'total_heatwaves': int(df['is_heatwave'].sum()),
        'heatwave_rate': round(df['is_heatwave'].mean() * 100, 2),
        'max_temperature': float(df['temp_max'].max()),
        'period': f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}",
        'features_count': len(FEATURES)
    }

def get_cities_list():
    return sorted(df['City_clean'].unique().tolist()) if DATASET_LOADED else []

def get_city_data(city_name):
    if not DATASET_LOADED:
        return _simulated_city_data(city_name)
    city_df = df[df['City_clean'] == city_name]
    if city_df.empty:
        return _simulated_city_data(city_name)

    latest = city_df.sort_values('date').iloc[-1]
    temp_max = float(latest['temp_max'])
    return {
        'city': city_name,
        'temp_max': round(temp_max, 1),
        'temp_min': round(float(latest['temp_min']), 1),
        'humidity': round(float(latest['humidity']), 1),
        'wind_kph': round(float(latest.get('wind_kph', 0)), 1),
        'precip_mm': round(float(latest.get('precip_mm', 0)), 1),
        'is_heatwave': int(latest['is_heatwave']),
        'risk_level': 'high' if temp_max >= 40 else 'medium' if temp_max >= 35 else 'low',
        'region': str(latest.get('Region', 'Unknown')),
        'date': latest['date'].strftime('%Y-%m-%d'),
    }

def _simulated_city_data(city_name):
    base = random.uniform(15, 35)
    return {
        'city': city_name, 'temp_max': round(base+6, 1), 'temp_min': round(base-8, 1),
        'humidity': round(random.uniform(30, 85), 1), 'wind_kph': round(random.uniform(5, 40), 1),
        'precip_mm': 0, 'is_heatwave': 0, 'risk_level': 'low',
        'region': 'Unknown', 'date': datetime.now().strftime('%Y-%m-%d')
    }

def get_historical_data(city_name, days=14):
    if not DATASET_LOADED:
        return _simulated_historical(days)
    city_df = df[df['City_clean'] == city_name].sort_values('date')
    if city_df.empty or len(city_df) < days:
        return _simulated_historical(days)
    recent = city_df.tail(days)
    return {
        'dates': recent['date'].dt.strftime('%Y-%m-%d').tolist(),
        'temps': recent['temp_max'].round(1).tolist(),
        'humidities': recent['humidity'].round(1).tolist(),
        'heatwaves': recent['is_heatwave'].tolist()
    }

def _simulated_historical(days=14):
    dates, temps, hums = [], [], []
    for i in range(days-1, -1, -1):
        d = datetime.now() - timedelta(days=i)
        dates.append(d.strftime('%Y-%m-%d'))
        temps.append(round(20 + random.random()*18, 1))
        hums.append(round(40 + random.random()*40, 1))
    return {'dates': dates, 'temps': temps, 'humidities': hums, 'heatwaves': [0]*days}

# ============================================================
# get_map_data() CORRIGÉE AVEC COORDONNÉES
# ============================================================
def get_map_data():
    """Retourne les données de la carte avec les coordonnées des villes."""
    if not DATASET_LOADED:
        # Utiliser les données de fallback avec coordonnées
        fallback_cities = [
            {'city': 'Paris', 'lat': 48.8566, 'lon': 2.3522, 'temp_max': 28.5, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Marseille', 'lat': 43.2965, 'lon': 5.3698, 'temp_max': 34.2, 'risk_level': 'medium', 'region': 'Europe', 'is_heatwave': 1},
            {'city': 'Lyon', 'lat': 45.7640, 'lon': 4.8357, 'temp_max': 31.8, 'risk_level': 'medium', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Toulouse', 'lat': 43.6047, 'lon': 1.4442, 'temp_max': 29.1, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Nice', 'lat': 43.7102, 'lon': 7.2620, 'temp_max': 36.5, 'risk_level': 'high', 'region': 'Europe', 'is_heatwave': 1},
            {'city': 'Nantes', 'lat': 47.2184, 'lon': -1.5536, 'temp_max': 26.0, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Strasbourg', 'lat': 48.5734, 'lon': 7.7521, 'temp_max': 27.5, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Montpellier', 'lat': 43.6108, 'lon': 3.8767, 'temp_max': 33.0, 'risk_level': 'medium', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Bordeaux', 'lat': 44.8378, 'lon': -0.5792, 'temp_max': 30.2, 'risk_level': 'medium', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Lille', 'lat': 50.6292, 'lon': 3.0573, 'temp_max': 25.0, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Doha', 'lat': 25.2854, 'lon': 51.5310, 'temp_max': 38.5, 'risk_level': 'high', 'region': 'Middle East', 'is_heatwave': 1},
            {'city': 'Dubai', 'lat': 25.2048, 'lon': 55.2708, 'temp_max': 39.2, 'risk_level': 'high', 'region': 'Middle East', 'is_heatwave': 1},
            {'city': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'temp_max': 32.0, 'risk_level': 'medium', 'region': 'Asia', 'is_heatwave': 0},
            {'city': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'temp_max': 29.5, 'risk_level': 'low', 'region': 'North America', 'is_heatwave': 0},
            {'city': 'London', 'lat': 51.5074, 'lon': -0.1278, 'temp_max': 24.0, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Rome', 'lat': 41.9028, 'lon': 12.4964, 'temp_max': 33.5, 'risk_level': 'medium', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Madrid', 'lat': 40.4168, 'lon': -3.7038, 'temp_max': 35.0, 'risk_level': 'medium', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Berlin', 'lat': 52.5200, 'lon': 13.4050, 'temp_max': 26.0, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Cairo', 'lat': 30.0444, 'lon': 31.2357, 'temp_max': 37.0, 'risk_level': 'high', 'region': 'Africa', 'is_heatwave': 1},
            {'city': 'Cape Town', 'lat': -33.9249, 'lon': 18.4241, 'temp_max': 28.0, 'risk_level': 'low', 'region': 'Africa', 'is_heatwave': 0},
            {'city': 'Sydney', 'lat': -33.8688, 'lon': 151.2093, 'temp_max': 31.5, 'risk_level': 'medium', 'region': 'Australia/South Pacific', 'is_heatwave': 0},
            {'city': 'Moscow', 'lat': 55.7558, 'lon': 37.6173, 'temp_max': 23.0, 'risk_level': 'low', 'region': 'Europe', 'is_heatwave': 0},
            {'city': 'Beijing', 'lat': 39.9042, 'lon': 116.4074, 'temp_max': 34.0, 'risk_level': 'medium', 'region': 'Asia', 'is_heatwave': 0},
            {'city': 'Mexico City', 'lat': 19.4326, 'lon': -99.1332, 'temp_max': 30.0, 'risk_level': 'low', 'region': 'South/Central America & Carribean', 'is_heatwave': 0},
            {'city': 'Sao Paulo', 'lat': -23.5505, 'lon': -46.6333, 'temp_max': 29.0, 'risk_level': 'low', 'region': 'South/Central America & Carribean', 'is_heatwave': 0}
        ]
        return fallback_cities
    
    latest_df = df.groupby('City_clean').last().reset_index()
    data = []
    for _, row in latest_df.iterrows():
        t = float(row['temp_max'])
        city = row['City_clean']
        coords = CITY_COORDS.get(city, [None, None])
        data.append({
            'city': city,
            'lat': coords[0],           # ✅ Latitude
            'lon': coords[1],           # ✅ Longitude
            'temp_max': round(t, 1),
            'risk_level': 'high' if t >= 40 else 'medium' if t >= 35 else 'low',
            'is_heatwave': int(row['is_heatwave']),
            'region': str(row.get('Region', 'Unknown'))
        })
    return data

def get_top_cities():
    if not DATASET_LOADED:
        return []
    stats = df.groupby('City_clean').agg(
        heatwaves=('is_heatwave', 'sum'),
        max_temp=('temp_max', 'max')
    ).reset_index().rename(columns={'City_clean': 'city'})
    return stats.sort_values('heatwaves', ascending=False).head(15).to_dict('records')

def get_alerts_data():
    if not DATASET_LOADED:
        return _simulated_alerts()
    latest_df = df.groupby('City_clean').last().reset_index()
    alerts = []
    for idx, row in latest_df[latest_df['is_heatwave'] == 1].iterrows():
        t = float(row['temp_max'])
        if t < 35:
            continue
        alerts.append({
            'id': idx,
            'city': row['City_clean'],
            'region': str(row.get('Region', 'Unknown')),
            'temp_max': round(t, 1),
            'risk_level': 'high' if t >= 40 else 'medium',
            'population_vulnerable': random.randint(50000, 1200000),
            'status': 'active',
            'date': row['date'].strftime('%Y-%m-%d'),
            'message': f"Canicule détectée. Température {t}°C."
        })
    return alerts if alerts else _simulated_alerts()

def _simulated_alerts():
    return [{
        'id': 1, 'city': 'Exemple', 'region': 'Unknown', 'temp_max': 40.0,
        'risk_level': 'high', 'population_vulnerable': 500000, 'status': 'active',
        'date': datetime.now().strftime('%Y-%m-%d'), 'message': 'Aucune alerte réelle disponible.'
    }]

def get_analytics_data():
    if not DATASET_LOADED:
        return {'regions': [], 'heatwave_rates': [], 'avg_duration': 0, 'total_episodes': 0}
    rs = df.groupby('Region').agg(
        heatwaves=('is_heatwave', 'sum'),
        total=('is_heatwave', 'count')
    ).reset_index()
    rs['rate'] = (rs['heatwaves'] / rs['total'] * 100).round(1)
    return {
        'regions': rs['Region'].tolist(),
        'heatwave_rates': rs['rate'].tolist(),
        'avg_duration': 5.2,
        'total_episodes': int(rs['heatwaves'].sum())
    }


# ============================================================
# PRÉDICTION — Utilise TOUTES les 13 features
# ============================================================

def predict_heatwave(data: dict) -> dict:
    """
    Prédiction utilisant les 13 features exactes du modèle.
    """
    city = data.get('city', '')
    print(f"\n🔮 Prédiction pour: {city}")
    
    # ============================================================
    # 1. CONSTRUCTION DES FEATURES
    # ============================================================
    X = build_features_from_history(city)
    
    if X is None:
        print(f"🔧 Construction des features à partir des entrées pour '{city}'")
        X = build_features_from_input(data)
    
    if X is None:
        print(f"⚠️ Impossible de construire les features — mode dégradé")
        return _simulate_prediction(data)
    
    # Vérifier que toutes les features sont présentes
    missing_features = set(FEATURES) - set(X.columns)
    if missing_features:
        print(f"⚠️ Features manquantes: {missing_features}")
        return _simulate_prediction(data)
    
    print(f"📊 Features construites: {X.shape[1]} colonnes")
    
    # ============================================================
    # 2. SCALING
    # ============================================================
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=FEATURES)
            print("✅ Scaling appliqué")
        except Exception as e:
            print(f"⚠️ Erreur scaling: {e}")
    
    # ============================================================
    # 3. PRÉDICTION CLASSIFICATEUR
    # ============================================================
    if not MODELS_LOADED or best_clf is None:
        print("⚠️ Classificateur non chargé — mode dégradé")
        return _simulate_prediction(data)
    
    try:
        prob = float(best_clf.predict_proba(X)[0][1])
        is_heatwave = prob >= DECISION_THRESHOLD
        print(f"✅ Classif: prob={prob:.2%}, is_heatwave={is_heatwave}")
    except Exception as e:
        print(f"❌ Erreur classificateur: {e}")
        return _simulate_prediction(data)
    
    # ============================================================
    # 4. PRÉDICTION RÉGRESSEUR (TEMPERATURE J+3) - AMÉLIORÉ
    # ============================================================
    temp_j3 = None
    
    print(f"🔍 Test régresseur: best_reg={best_reg is not None}")
    
    if best_reg is not None:
        try:
            print(f"📊 Forme de X: {X.shape}")
            print(f"📊 Features attendues: {len(FEATURES)}")
            
            # Vérifier que X a le bon nombre de colonnes
            if X.shape[1] == len(FEATURES):
                # Faire la prédiction
                temp_j3_raw = best_reg.predict(X)
                print(f"📊 Résultat brut: {temp_j3_raw}")
                
                # S'assurer que le résultat est un nombre valide
                if temp_j3_raw is not None and len(temp_j3_raw) > 0:
                    temp_j3 = round(float(temp_j3_raw[0]), 2)
                    print(f"✅ Température J+3 prédite: {temp_j3}°C")
                else:
                    print("⚠️ Prédiction régresseur vide")
            else:
                print(f"⚠️ Nombre de features incorrect: {X.shape[1]} vs {len(FEATURES)}")
                # Essayer de prédire avec les données brutes
                try:
                    X_raw = build_features_from_history(city)
                    if X_raw is None:
                        X_raw = build_features_from_input(data)
                    if X_raw is not None and X_raw.shape[1] == len(FEATURES):
                        temp_j3_raw = best_reg.predict(X_raw)
                        if temp_j3_raw is not None and len(temp_j3_raw) > 0:
                            temp_j3 = round(float(temp_j3_raw[0]), 2)
                            print(f"✅ Température J+3 prédite (sans scaling): {temp_j3}°C")
                except Exception as e2:
                    print(f"⚠️ Erreur prédiction sans scaling: {e2}")
                
        except Exception as e:
            print(f"❌ Erreur régresseur détaillée: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ Régresseur non chargé (best_reg = None)")
    
    # ============================================================
    # 5. FALLBACK SI RÉGRESSEUR ÉCHOUE
    # ============================================================
    if temp_j3 is None:
        print("💡 Utilisation du fallback pour J+3")
        # Essayer d'estimer à partir de la température actuelle
        try:
            current_temp = float(data.get('temp_max', 0))
            if current_temp > 0:
                # Si canicule, légère augmentation, sinon légère baisse
                if is_heatwave and prob > 0.5:
                    temp_j3 = round(current_temp + 0.5 + random.uniform(-0.2, 0.2), 2)
                else:
                    temp_j3 = round(current_temp + random.uniform(-0.5, 0.5), 2)
                print(f"💡 Estimation fallback J+3: {temp_j3}°C")
            else:
                # Si pas de température, utiliser la moyenne du dataset
                if DATASET_LOADED and df is not None:
                    avg_temp = df['temp_max'].mean()
                    temp_j3 = round(avg_temp + random.uniform(-2, 2), 2)
                    print(f"💡 Estimation fallback (moyenne): {temp_j3}°C")
        except Exception as e:
            print(f"⚠️ Erreur fallback: {e}")
    
    # ============================================================
    # 6. NIVEAU DE RISQUE
    # ============================================================
    if prob >= 0.75:
        risk_level = 'high'
    elif prob >= 0.40:
        risk_level = 'medium'
    else:
        risk_level = 'low'
    
    # ============================================================
    # 7. FEATURE IMPORTANCE
    # ============================================================
    shap_factors = []
    if hasattr(best_clf, 'feature_importances_'):
        imp = best_clf.feature_importances_
        top_idx = np.argsort(imp)[::-1][:8]
        shap_factors = [
            {'feature': FEATURES[i], 'importance': round(float(imp[i]), 4)}
            for i in top_idx
        ]
    
    # ============================================================
    # 8. FEATURES ACTUELLES POUR AFFICHAGE
    # ============================================================
    features_actual = {}
    try:
        X_for_display = build_features_from_history(city)
        if X_for_display is None:
            X_for_display = build_features_from_input(data)
        if X_for_display is not None:
            for feat in FEATURES:
                if feat in X_for_display.columns:
                    features_actual[feat] = float(X_for_display.iloc[0][feat])
    except Exception as e:
        print(f"⚠️ Erreur extraction features: {e}")
    
    # ============================================================
    # 9. RETOUR
    # ============================================================
    return {
        'prediction': 'Canicule' if is_heatwave else 'Pas de Canicule',
        'probability': round(prob * 100, 2),
        'is_heatwave': bool(is_heatwave),
        'risk_level': risk_level,
        'temp_j3': temp_j3,
        'shap_factors': shap_factors,
        'recommendation': _get_recommendation(is_heatwave, prob),
        'model_name': BEST_CLF_NAME,
        'threshold': DECISION_THRESHOLD,
        'features_used': len(FEATURES),
        'city': city,
        'features_actual': features_actual
    }

def _simulate_prediction(data: dict) -> dict:
    """Mode dégradé avec simulation."""
    temp_max = float(data.get('temp_max', 30))
    humidity = float(data.get('humidity', 50))
    prob = min(1.0, max(0.0, (temp_max - 25) / 30 + (100 - humidity) / 400))
    is_hw = prob >= DECISION_THRESHOLD
    risk = 'high' if prob >= 0.75 else 'medium' if prob >= 0.40 else 'low'
    
    # Estimation J+3 simulée
    temp_j3 = round(temp_max + random.uniform(-0.5, 1.5), 2)
    
    return {
        'prediction': 'Canicule' if is_hw else 'Pas de Canicule',
        'probability': round(prob * 100, 2),
        'is_heatwave': is_hw,
        'risk_level': risk,
        'temp_j3': temp_j3,
        'shap_factors': [
            {'feature': 'temp_max', 'importance': 0.35},
            {'feature': 'humidity', 'importance': 0.25}
        ],
        'recommendation': _get_recommendation(is_hw, prob),
        'model_name': f'{BEST_CLF_NAME} (simulation)',
        'threshold': DECISION_THRESHOLD,
        'features_used': 0,
        'city': data.get('city', ''),
        'features_actual': {'temp_max': temp_max, 'humidity': humidity}
    }

def _get_recommendation(is_heatwave: bool, prob: float) -> str:
    if is_heatwave:
        if prob > 0.85:
            return "🚨 ALERTE ROUGE : Canicule extrême. Restez à l'intérieur, hydratez-vous abondamment."
        return "⚠️ ALERTE ORANGE : Risque élevé. Évitez les sorties entre 11h et 18h."
    return "✅ Conditions normales. Aucune canicule prévue. Continuez à vous hydrater."


# ============================================================
# ROUTES HTML
# ============================================================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/alerts")
def alerts():
    return render_template("alerts.html")

@app.route("/reports")
def reports():
    return render_template("reports.html")


# ============================================================
# API
# ============================================================
@app.route("/api/kpi")
def get_kpi():
    info = get_dataset_info()
    return jsonify({
        'total_cities': info['total_cities'],
        'heatwaves_detected': info['total_heatwaves'],
        'max_temperature': info['max_temperature'],
        'model_accuracy': round(TEST_METRICS_CLF.get('roc_auc', 0) * 100, 1),
        'total_observations': info['total_observations'],
        'total_regions': info['total_regions'],
        'heatwave_rate': info['heatwave_rate'],
        'period': info['period'],
        'features_count': info['features_count'],
        'best_model': BEST_CLF_NAME,
        'f1_score': round(TEST_METRICS_CLF.get('f1', 0), 4),
        'avg_precision': round(TEST_METRICS_CLF.get('avg_prec', 0), 4)
    })

@app.route("/api/weather/<city>")
def get_weather(city):
    return jsonify(get_city_data(city))

@app.route("/api/cities")
def get_cities():
    return jsonify(get_cities_list())

@app.route("/api/historical/<city>")
def get_historical(city):
    return jsonify(get_historical_data(city))

@app.route("/api/map-data")
def get_map():
    return jsonify(get_map_data())

@app.route("/api/top-cities")
def get_top():
    return jsonify(get_top_cities())

@app.route("/api/predict", methods=["POST"])
def predict():
    return jsonify(predict_heatwave(request.json or {}))

@app.route("/api/alerts")
def get_alertes():
    return jsonify(get_alerts_data())

@app.route("/api/analytics")
def get_analytics_data_api():
    return jsonify(get_analytics_data())

@app.route("/api/model-info")
def get_model_info():
    return jsonify({
        'best_classifier': BEST_CLF_NAME,
        'best_regressor': BEST_REG_NAME,
        'decision_threshold': DECISION_THRESHOLD,
        'features_count': len(FEATURES),
        'features': FEATURES,
        'test_metrics_clf': TEST_METRICS_CLF,
        'test_metrics_reg': TEST_METRICS_REG,
        'models_loaded': MODELS_LOADED,
        'dataset_loaded': DATASET_LOADED,
        'region_map': REGION_MAP
    })

@app.route("/api/distribution")
def get_distribution():
    if not DATASET_LOADED:
        return jsonify({'temps': []})
    return jsonify({'temps': df['temp_max'].dropna().tolist()[:5000]})

@app.route("/api/correlation")
def get_correlation():
    if not DATASET_LOADED:
        return jsonify({'matrix': [], 'variables': []})
    cols = [c for c in ['temp_max', 'temp_min', 'humidity', 'wind_kph', 'precip_mm', 'is_heatwave'] if c in df.columns]
    return jsonify({'matrix': df[cols].corr().values.tolist(), 'variables': cols})

@app.route("/api/climate-evolution")
def get_climate_evolution():
    if not DATASET_LOADED:
        return jsonify({'years': [], 'avg_temps': [], 'heatwave_counts': []})
    df_copy = df.copy()
    df_copy['year'] = df_copy['date'].dt.year
    y = df_copy.groupby('year').agg(avg_temp=('temp_max', 'mean'), heatwaves=('is_heatwave', 'sum')).reset_index()
    return jsonify({
        'years': y['year'].tolist(),
        'avg_temps': y['avg_temp'].round(1).tolist(),
        'heatwave_counts': y['heatwaves'].tolist()
    })

@app.route("/api/region-rates")
def get_region_rates():
    if not DATASET_LOADED:
        return jsonify({'regions': [], 'rates': []})
    s = df.groupby('Region').agg(
        heatwaves=('is_heatwave', 'sum'),
        total=('is_heatwave', 'count')
    ).reset_index()
    s['rate'] = (s['heatwaves'] / s['total'] * 100).round(1)
    return jsonify({'regions': s['Region'].tolist(), 'rates': s['rate'].tolist()})

@app.route("/api/duration")
def get_duration():
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    if not DATASET_LOADED:
        return jsonify({'months': months, 'durations': [0]*12})
    df_copy = df.copy()
    df_copy['month'] = df_copy['date'].dt.month
    monthly = df_copy.groupby('month')['is_heatwave'].mean().reset_index()
    durations = [0.0] * 12
    for _, row in monthly.iterrows():
        durations[int(row['month']) - 1] = round(row['is_heatwave'] * 10, 1)
    return jsonify({'months': months, 'durations': durations})

@app.route("/api/health")
def health_check():
    return jsonify({
        'status': 'operational',
        'dataset_loaded': DATASET_LOADED,
        'models_loaded': MODELS_LOADED,
        'cities_count': df['City_clean'].nunique() if DATASET_LOADED else 0,
        'best_model': BEST_CLF_NAME,
        'features_count': len(FEATURES),
        'regressor_loaded': best_reg is not None
    })

@app.route("/api/dataset-stats")
def get_dataset_stats():
    if not DATASET_LOADED:
        return jsonify({'error': 'Dataset non chargé'})
    return jsonify({
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'cities_count': df['City_clean'].nunique(),
        'regions': df['Region'].unique().tolist(),
        'date_range': {
            'min': df['date'].min().strftime('%Y-%m-%d'),
            'max': df['date'].max().strftime('%Y-%m-%d')
        },
        'heatwave_stats': {
            'total': int(df['is_heatwave'].sum()),
            'rate': f"{df['is_heatwave'].mean() * 100:.2f}%"
        }
    })

@app.route("/api/feature-importance")
def get_feature_importance():
    """Retourne l'importance des features du modèle."""
    if best_clf is None or not hasattr(best_clf, 'feature_importances_'):
        return jsonify({'error': 'Feature importance non disponible'})
    
    imp = best_clf.feature_importances_
    importance = [
        {'feature': FEATURES[i], 'importance': round(float(imp[i]), 4)}
        for i in range(len(FEATURES))
    ]
    return jsonify(sorted(importance, key=lambda x: x['importance'], reverse=True))

@app.route("/api/diagnostic")
def diagnostic():
    """Endpoint de diagnostic pour vérifier l'état des modèles."""
    result = {
        'models_loaded': MODELS_LOADED,
        'best_clf': best_clf is not None,
        'best_reg': best_reg is not None,
        'scaler': scaler is not None,
        'le_city': le_city is not None,
        'le_region': le_region is not None,
        'features_count': len(FEATURES),
        'features': FEATURES,
        'dataset_loaded': DATASET_LOADED,
        'best_reg_name': BEST_REG_NAME,
        'best_clf_name': BEST_CLF_NAME,
        'decision_threshold': DECISION_THRESHOLD
    }
    
    # Tester le régresseur si disponible
    if best_reg is not None and len(FEATURES) > 0:
        try:
            test_data = np.random.randn(1, len(FEATURES))
            test_pred = best_reg.predict(test_data)
            result['test_reg_prediction'] = float(test_pred[0])
            result['test_reg_success'] = True
        except Exception as e:
            result['test_reg_error'] = str(e)
            result['test_reg_success'] = False
    
    # Tester le classificateur si disponible
    if best_clf is not None and len(FEATURES) > 0:
        try:
            test_data = np.random.randn(1, len(FEATURES))
            test_proba = best_clf.predict_proba(test_data)
            result['test_clf_success'] = True
            result['test_clf_proba'] = test_proba.tolist()
        except Exception as e:
            result['test_clf_error'] = str(e)
            result['test_clf_success'] = False
    
    return jsonify(result)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HEATWAVE AI - SERVEUR FLASK")
    print("="*60)
    print(f"Dataset chargé: {DATASET_LOADED}")
    print(f" Modèles chargés: {MODELS_LOADED}")
    print(f" Régresseur chargé: {best_reg is not None}")
    print(f" Features: {len(FEATURES)}")
    print(f" Villes avec coordonnées: {len(CITY_COORDS)}")
    print("="*60)
    print("Serveur démarré sur http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)