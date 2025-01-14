import os

# Configuration des features
NUMERIC_FEATURES = [
    'age', 'player_height', 'player_weight', 'team_encoded',
    'college_encoded', 'country_encoded', 'gp', 'reb', 'ast',
    'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
]

CATEGORICAL_FEATURES = ['team_abbreviation', 'college', 'country']
ENCODED_FEATURES = ['team_encoded', 'college_encoded', 'country_encoded']

# Chemins des fichiers
DATA_PATH = 'all_seasons.csv'
MODEL_PATH = {
    'rf': 'rf_model.joblib',
    'dnn': 'dnn_model.joblib',
    'scaler': 'scaler.joblib'
}

# Configuration de l'interface
TITLE = "🏀 NBA Points Predictor"
PAGE_CONFIG = {
    "page_title": "NBA Stats Predictor",
    "layout": "wide"
}