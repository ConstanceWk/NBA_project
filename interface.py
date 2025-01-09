import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Configuration constante des features pour assurer la cohérence
NUMERIC_FEATURES = [
    'age', 'player_height', 'player_weight', 'gp', 'reb', 'ast',
    'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct'
]

CATEGORICAL_FEATURES = ['team_abbreviation', 'college', 'country']
ENCODED_FEATURES = ['team_encoded', 'college_encoded', 'country_encoded']
ALL_FEATURES = NUMERIC_FEATURES + ENCODED_FEATURES

st.set_page_config(page_title="NBA Stats Predictor", layout="wide")

# [Style CSS reste identique...]
st.markdown("""
   <style>
       .title {
           text-align: center;
           background: linear-gradient(45deg, #C9082A, #17408B);
           color: white;
           padding: 20px;
           border-radius: 15px;
           margin-bottom: 30px;
           font-size: 48px;
           text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
       }
       .stButton>button {
           background: linear-gradient(45deg, #C9082A, #17408B) !important;
           color: white !important;
           font-weight: bold !important;
           border: none !important;
           padding: 15px 30px !important;
           width: 100%;
           transition: all 0.3s ease !important;
       }
       .stButton>button:hover {
           transform: scale(1.05) !important;
           box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
       }
       .stat-card {
           background: white;
           padding: 20px;
           border-radius: 15px;
           box-shadow: 0 4px 15px rgba(0,0,0,0.1);
           margin: 10px;
           border: 2px solid #17408B;
       }
       .stat-header {
           color: #17408B;
           font-weight: bold;
           font-size: 24px;
           margin-bottom: 15px;
           text-align: center;
       }
       .prediction-card {
           background: linear-gradient(45deg, #C9082A, #17408B);
           color: white;
           padding: 30px;
           border-radius: 15px;
           text-align: center;
           margin: 20px 0;
           box-shadow: 0 5px 20px rgba(0,0,0,0.2);
       }
       .player-card {
           background: white;
           padding: 20px;
           border-radius: 15px;
           margin: 20px 0;
           box-shadow: 0 4px 15px rgba(0,0,0,0.1);
           border: 2px solid #17408B;
           text-align: center;
       }
   </style>
   <div class='title'>🏀 NBA Points Predictor</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess NBA player data"""
    df = pd.read_csv('all_seasons.csv')
    # Remplir les valeurs manquantes de manière cohérente
    df['college'] = df['college'].fillna('None')
    df['country'] = df['country'].fillna('USA')
    return df

def prepare_features(nba_data):
    """Prepare features for model training"""
    le_team = LabelEncoder()
    le_college = LabelEncoder()
    le_country = LabelEncoder()
    
    # Encoder les variables catégorielles
    nba_data['team_encoded'] = le_team.fit_transform(nba_data['team_abbreviation'])
    nba_data['college_encoded'] = le_college.fit_transform(nba_data['college'])
    nba_data['country_encoded'] = le_country.fit_transform(nba_data['country'])
    
    # Sélectionner les colonnes dans l'ordre exact
    X = nba_data[ALL_FEATURES].copy()
    
    # Stocker les encodeurs pour réutilisation
    encoders = {
        'team': le_team,
        'college': le_college,
        'country': le_country
    }
    
    return X, encoders

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model and scaler"""
    try:
        st.sidebar.write("Fichiers disponibles:", os.listdir())
        model = joblib.load('dnn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        st.sidebar.success("Modèle chargé avec succès!")
        return model, scaler
    except Exception as e:
        st.error(f"Erreur détaillée lors du chargement: {str(e)}")
        return None, None

def predict_points(player_data, model, scaler, encoders):
    """Predict points for a given player"""
    try:
        # Créer un DataFrame avec les données du joueur
        player_df = pd.DataFrame([player_data])
        
        # Encoder les variables catégorielles
        player_df['team_encoded'] = encoders['team'].transform([player_data['team_abbreviation']])
        player_df['college_encoded'] = encoders['college'].transform([player_data['college']])
        player_df['country_encoded'] = encoders['country'].transform([player_data['country']])
        
        # Sélectionner et ordonner les features exactement comme lors de l'entraînement
        player_features = player_df[ALL_FEATURES].copy()
        
        # Scaler les features
        player_scaled = scaler.transform(player_features)
        
        # Faire la prédiction
        prediction = model.predict(player_scaled)
        return prediction[0][0] if isinstance(prediction, np.ndarray) else prediction
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}\nFeatures disponibles: {player_df.columns.tolist()}")
        st.error(f"Features attendues: {ALL_FEATURES}")
        return None

try:
    # Charger données et modèles
    nba_data = load_data()
    X, encoders = prepare_features(nba_data)
    model, scaler = load_trained_model()
    
    if model is None or scaler is None:
        st.error("Impossible de charger les modèles. Veuillez vérifier les fichiers .joblib")
        st.stop()
    
    prediction_mode = st.radio("Select Mode", ["Manual Input", "Random Player"], horizontal=True)
    
    if prediction_mode == "Manual Input":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='stat-card'><div class='stat-header'>📊 Physical Stats</div>", unsafe_allow_html=True)
            age = st.number_input("Age", min_value=18, max_value=40, value=25)
            height = st.number_input("Height (cm)", min_value=160, max_value=230, value=198)
            weight = st.number_input("Weight (kg)", min_value=60, max_value=150, value=95)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='stat-card'><div class='stat-header'>🎯 Game Stats</div>", unsafe_allow_html=True)
            games = st.number_input("Games Played", min_value=1, max_value=82, value=82)
            rebounds = st.number_input("Rebounds per Game", min_value=0.0, max_value=20.0, value=5.0)
            assists = st.number_input("Assists per Game", min_value=0.0, max_value=15.0, value=4.0)
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='stat-card'><div class='stat-header'>📈 Advanced Stats</div>", unsafe_allow_html=True)
            usage = st.number_input("Usage %", min_value=0.0, max_value=40.0, value=20.0)
            ts = st.number_input("True Shooting %", min_value=0.0, max_value=100.0, value=55.0)
            oreb_pct = st.number_input("OREB %", min_value=0.0, max_value=100.0, value=10.0)
            dreb_pct = st.number_input("DREB %", min_value=0.0, max_value=100.0, value=20.0)
            ast_pct = st.number_input("AST %", min_value=0.0, max_value=100.0, value=15.0)
            st.markdown("</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            team = st.selectbox("Team", sorted(nba_data['team_abbreviation'].unique()))
        with col2:
            college = st.selectbox("College", sorted(nba_data['college'].unique()))
        with col3:
            country = st.selectbox("Country", sorted(nba_data['country'].unique()))

        if st.button("Predict Points"):
            # Préparer les données du joueur en suivant l'ordre exact des features
            player_data = {
                'age': age,
                'player_height': height,
                'player_weight': weight,
                'gp': games,
                'reb': rebounds,
                'ast': assists,
                'oreb_pct': oreb_pct,
                'dreb_pct': dreb_pct,
                'usg_pct': usage,
                'ts_pct': ts,
                'ast_pct': ast_pct,
                'team_abbreviation': team,
                'college': college,
                'country': country
            }
            
            # Faire la prédiction
            predicted_points = predict_points(player_data, model, scaler, encoders)
            
            if predicted_points is not None:
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h2>Predicted Points Per Game</h2>
                        <h1 style='font-size: 4em; margin: 20px 0;'>{predicted_points:.1f}</h1>
                    </div>
                """, unsafe_allow_html=True)

                # Créer le graphique radar
                categories = ['Scoring', 'Playmaking', 'Rebounds', 'Usage', 'Efficiency']
                values = [
                    predicted_points/30 * 5,
                    assists/10 * 5,
                    rebounds/10 * 5,
                    usage/40 * 5,
                    ts/100 * 5
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    line_color='#17408B'
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    showlegend=False,
                    title={
                        'text': "Player Profile",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 24, 'color': '#17408B'}
                    },
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
    else:  # Mode "Random Player"
        if st.button("Get Random Player"):
            random_idx = random.randint(0, len(nba_data)-1)
            player = nba_data.iloc[random_idx]
            
            st.markdown(f"""
                <div class='player-card'>
                    <h2 style='color: #17408B;'>{player['player_name']}</h2>
                    <h3 style='color: #C9082A;'>{player['team_abbreviation']} | {player['season']}</h3>
                    <div style='margin: 20px 0;'>
                        <p><strong>Age:</strong> {player['age']}</p>
                        <p><strong>Height:</strong> {player['player_height']:.1f} cm</p>
                        <p><strong>Weight:</strong> {player['player_weight']:.1f} kg</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Préparer les données et faire la prédiction
            player_dict = {col: player[col] for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
            predicted_points = predict_points(player_dict, model, scaler, encoders)
            
            if predicted_points is not None:
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h2>Points Comparison</h2>
                        <div style='display: flex; justify-content: space-around; margin: 20px 0;'>
                            <div>
                                <h3>Actual Points</h3>
                                <h1 style='font-size: 3em;'>{player['pts']:.1f}</h1>
                            </div>
                            <div>
                                <h3>Predicted Points</h3>
                                <h1 style='font-size: 3em;'>{predicted_points:.1f}</h1>
                            </div>
                        </div>
                        <h3>Prediction Error: {abs(player['pts'] - predicted_points):.1f} points</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                categories = ['Scoring', 'Playmaking', 'Rebounds', 'Usage', 'Efficiency']
                values = [
                    player['pts']/30 * 5,
                    player['ast']/10 * 5,
                    player['reb']/10 * 5,
                    player['usg_pct']/40 * 5,
                    player['ts_pct']/100 * 5
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    line_color='#17408B'
                ))
                

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    showlegend=False,
                    title={
                        'text': "Player Profile",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 24, 'color': '#17408B'}
                    },
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)

except Exception as e:  
    st.error(f"Une erreur s'est produite: {str(e)}")                                                       