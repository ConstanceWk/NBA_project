import joblib
import streamlit as st
import numpy as np
from config import MODEL_PATH, NUMERIC_FEATURES

@st.cache_resource
def load_trained_models():
    """Load the pre-trained models and scaler"""
    try:
        rf_model = joblib.load(MODEL_PATH['rf'])
        dnn_model = joblib.load(MODEL_PATH['dnn'])
        scaler = joblib.load(MODEL_PATH['scaler'])
        
        st.sidebar.success("Modèles chargés avec succès!")
        return rf_model, dnn_model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        return None, None, None

def predict_points(player_data, rf_model, dnn_model, scaler, encoders):
    """Predict points using both models"""
    try:
        # Créer l'array des features dans le bon ordre
        features = []
        for feature in NUMERIC_FEATURES:
            if feature in ['team_encoded', 'college_encoded', 'country_encoded']:
                # Pour les features encodées
                base_feature = feature.replace('_encoded', '')
                encoded_val = encoders[base_feature].transform([str(player_data[base_feature + '_abbreviation' if base_feature == 'team' else base_feature])])[0]
                features.append(encoded_val)
            else:
                # Pour les features numériques
                features.append(float(str(player_data[feature]).replace(',', '.')))
        
        # Convertir en array numpy et faire les prédictions
        features_array = np.array(features, dtype=np.float64).reshape(1, -1)
        player_scaled = scaler.transform(features_array)
        
        rf_pred = float(rf_model.predict(player_scaled)[0])
        dnn_pred = float(dnn_model.predict(player_scaled)[0])
        
        return rf_pred, dnn_pred
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None, None