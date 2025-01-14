import streamlit as st
import random

from config import PAGE_CONFIG, TITLE
from styles import MAIN_STYLE
from data_processing import load_data, prepare_features
from models import load_trained_models, predict_points
from visualizations import (
    display_player_card, 
    display_predictions, 
    create_radar_chart,
    display_manual_input_form
)

def main():
    # Configuration de la page
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(MAIN_STYLE, unsafe_allow_html=True)
    st.markdown(f"<div class='title'>{TITLE}</div>", unsafe_allow_html=True)

    try:
        # Charger données et modèles
        nba_data = load_data()
        if nba_data is not None:
            X, encoders = prepare_features(nba_data)
            rf_model, dnn_model, scaler = load_trained_models()
            
            if rf_model is None or dnn_model is None or scaler is None:
                st.error("Impossible de charger les modèles. Veuillez vérifier les fichiers .joblib")
                st.stop()
            
            # Mode de prédiction
            prediction_mode = st.radio(
                "Select Mode", 
                ["Manual Input", "Random Player"], 
                horizontal=True
            )
            
            if prediction_mode == "Random Player":
                if st.button("Get Random Player"):
                    # Sélection aléatoire d'un joueur
                    player = nba_data.iloc[random.randint(0, len(nba_data)-1)]
                    
                    # Afficher les informations du joueur
                    display_player_card(player)
                    
                    # Préparer les données
                    player_dict = {
                        'age': player['age'],
                        'player_height': player['player_height'],
                        'player_weight': player['player_weight'],
                        'gp': player['gp'],
                        'reb': player['reb'],
                        'ast': player['ast'],
                        'oreb_pct': player['oreb_pct'],
                        'dreb_pct': player['dreb_pct'],
                        'usg_pct': player['usg_pct'],
                        'ts_pct': player['ts_pct'],
                        'ast_pct': player['ast_pct'],
                        'team_abbreviation': player['team_abbreviation'],
                        'college': player['college'],
                        'country': player['country']
                    }
                    
                    # Faire les prédictions
                    rf_pred, dnn_pred = predict_points(
                        player_dict, 
                        rf_model, 
                        dnn_model, 
                        scaler, 
                        encoders
                    )
                    
                    if rf_pred is not None and dnn_pred is not None:
                        # Afficher les prédictions
                        display_predictions(player, rf_pred, dnn_pred)
                        # Afficher le graphique radar
                        create_radar_chart(player, rf_pred, dnn_pred)
            
            else:  # Manual Input mode
                player_data = display_manual_input_form(nba_data)
                
                if player_data:
                    rf_pred, dnn_pred = predict_points(
                        player_data, 
                        rf_model, 
                        dnn_model, 
                        scaler, 
                        encoders
                    )
                    
                    if rf_pred is not None and dnn_pred is not None:
                        st.markdown(f"""
                            <div class='model-comparison'>
                                <div class='model-card rf-card'>
                                    <div class='model-title'>Random Forest Prediction</div>
                                    <div class='prediction-value'>{rf_pred:.1f}</div>
                                </div>
                                <div class='model-card dnn-card'>
                                    <div class='model-title'>Deep Neural Network Prediction</div>
                                    <div class='prediction-value'>{dnn_pred:.1f}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Créer le graphique radar pour les prédictions manuelles
                        create_radar_chart(player_data, rf_pred, dnn_pred)

    except Exception as e:
        st.error(f"Une erreur s'est produite: {str(e)}")

if __name__ == "__main__":
    main()