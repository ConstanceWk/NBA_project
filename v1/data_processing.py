import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from config import DATA_PATH, NUMERIC_FEATURES

@st.cache_data
def load_data():
    """Load and preprocess NBA player data"""
    try:
        df = pd.read_csv(DATA_PATH)
        df['college'] = df['college'].fillna('None')
        df['country'] = df['country'].fillna('USA')
        st.success("Données chargées avec succès!")
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None

def prepare_features(nba_data):
    """Prepare features for model training"""
    le_team = LabelEncoder()
    le_college = LabelEncoder()
    le_country = LabelEncoder()
    
    nba_data['team_encoded'] = le_team.fit_transform(nba_data['team_abbreviation'])
    nba_data['college_encoded'] = le_college.fit_transform(nba_data['college'])
    nba_data['country_encoded'] = le_country.fit_transform(nba_data['country'])
    
    # Créer le DataFrame avec l'ordre exact des features utilisé dans l'entraînement
    X = nba_data[NUMERIC_FEATURES].copy()
    
    encoders = {
        'team': le_team,
        'college': le_college,
        'country': le_country
    }
    
    return X, encoders