import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import joblib
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
st.markdown("""
    <style>
        .model-comparison {
            display: flex;
            justify-content: space-between;
            margin: 30px 0;
            gap: 24px;
        }
        
        .model-card {
            flex: 1;
            background: white;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid;
        }
        
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(0,0,0,0.12);
        }
        
        .rf-card {
            border-color: #17408B;
        }
        
        .dnn-card {
            border-color: #C9082A;
        }
        
        .actual-card {
            border-color: #2E8B57;
        }
        
        .player-card {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin: 30px 0;
            border-left: 5px solid #17408B;
        }
        
        .player-name {
            color: #17408B;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .player-team {
            color: #C9082A;
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        
        .player-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .stat-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-label {
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }
        
        .stat-value {
            color: #17408B;
            font-size: 20px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)
# Style CSS amélioré pour inclure le DNN
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
       .model-comparison {
           display: flex;
           justify-content: space-between;
           margin: 20px 0;
           gap: 20px;
       }
       .model-card {
           flex: 1;
           background: white;
           padding: 20px;
           border-radius: 15px;
           box-shadow: 0 4px 15px rgba(0,0,0,0.1);
           text-align: center;
           border: 2px solid;
       }
       .rf-card {
           border-color: #17408B;
       }
       .dnn-card {
           border-color: #C9082A;
       }
       .actual-card {
           border-color: #2E8B57;
       }
       .player-card {
           background: white;
           padding: 20px;
           border-radius: 15px;
           box-shadow: 0 4px 15px rgba(0,0,0,0.1);
           margin: 20px 0;
       }
   </style>
   <div class='title'>🏀 NBA Points Predictor</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess NBA player data"""
    try:
        df = pd.read_csv('all_seasons.csv')
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
    
    X = nba_data[ALL_FEATURES].copy()
    
    encoders = {
        'team': le_team,
        'college': le_college,
        'country': le_country
    }
    
    return X, encoders

@st.cache_resource
def load_trained_models():
    """Load the pre-trained models and scaler"""
    try:
        # Vérifier l'existence des fichiers
        required_files = ['rf_model.joblib', 'dnn_model.joblib', 'scaler.joblib']
        for file in required_files:
            if not os.path.exists(file):
                st.error(f"Le fichier {file} n'existe pas dans le répertoire")
                return None, None, None
        
        # Charger les modèles
        rf_model = joblib.load('rf_model.joblib')
        dnn_model = joblib.load('dnn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        st.sidebar.success("Modèles chargés avec succès!")
        return rf_model, dnn_model, scaler
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        st.error(f"Répertoire actuel: {os.getcwd()}")
        st.error(f"Fichiers disponibles: {os.listdir()}")
        return None, None, None

def predict_points(player_data, rf_model, dnn_model, scaler, encoders):
    try:
        # Convertir et valider les données numériques
        features_data = np.array([
            float(player_data['age']),
            float(player_data['player_height']),
            float(player_data['player_weight']),
            float(str(player_data['gp']).replace(',', '.')),  # Gérer les potentiels séparateurs décimaux
            float(str(player_data['reb']).replace(',', '.')),
            float(str(player_data['ast']).replace(',', '.')),
            float(str(player_data['oreb_pct']).replace(',', '.')),
            float(str(player_data['dreb_pct']).replace(',', '.')),
            float(str(player_data['usg_pct']).replace(',', '.')),
            float(str(player_data['ts_pct']).replace(',', '.')),
            float(str(player_data['ast_pct']).replace(',', '.')),
            int(encoders['team'].transform([str(player_data['team_abbreviation'])])[0]),
            int(encoders['college'].transform([str(player_data['college'])])[0]),
            int(encoders['country'].transform([str(player_data['country'])])[0])
        ], dtype=np.float64).reshape(1, -1)
        
        # Debug : afficher les valeurs
        st.write("Données avant prédiction:", features_data.tolist())
        
        # Faire les prédictions
        player_scaled = scaler.transform(features_data)
        rf_pred = float(rf_model.predict(player_scaled)[0])
        dnn_pred = float(dnn_model.predict(player_scaled)[0])
        
        return rf_pred, dnn_pred
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        st.error(f"Type d'erreur: {type(e).__name__}")
        st.error(f"Valeurs tentées de conversion:")
        for key, value in player_data.items():
            st.error(f"{key}: {value} (type: {type(value)})")
        return None, None

try:
    # Charger données et modèles
    nba_data = load_data()
    if nba_data is not None:
        X, encoders = prepare_features(nba_data)
        rf_model, dnn_model, scaler = load_trained_models()
        
        if rf_model is None or dnn_model is None or scaler is None:
            st.error("Impossible de charger les modèles. Veuillez vérifier les fichiers .joblib")
            st.stop()
        
        prediction_mode = st.radio("Select Mode", ["Manual Input", "Random Player"], horizontal=True)
        
        if prediction_mode == "Random Player":
            # Remplacer les deux définitions de player_dict par celle-ci :
            if st.button("Get Random Player"):
                random_idx = random.randint(0, len(nba_data)-1)
                player = nba_data.iloc[random_idx]
                
                # Afficher les informations du joueur
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
                
                # S'assurer que toutes les features sont incluses
                player_dict = {
                    # Features numériques
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
                    # Features catégorielles
                    'team_abbreviation': player['team_abbreviation'],
                    'college': player['college'],
                    'country': player['country']
                }
                
                rf_pred, dnn_pred = predict_points(player_dict, rf_model, dnn_model, scaler, encoders)

                # Après avoir obtenu rf_pred et dnn_pred dans le mode Random Player
                if rf_pred is not None and dnn_pred is not None:
                    # Afficher les prédictions
                    st.markdown(f"""
                        <div class='model-comparison'>
                            <div class='model-card actual-card'>
                                <div style='font-size: 20px; font-weight: 600; margin-bottom: 10px; color: #2E8B57;'>Points Réels</div>
                                <div style='font-size: 36px; font-weight: 700;'>{player['pts']:.1f}</div>
                                <div style='font-size: 14px; color: #666; margin-top: 5px;'>Points par match</div>
                            </div>
                            <div class='model-card rf-card'>
                                <div style='font-size: 20px; font-weight: 600; margin-bottom: 10px; color: #17408B;'>Random Forest</div>
                                <div style='font-size: 36px; font-weight: 700;'>{rf_pred:.1f}</div>
                                <div style='color: #666; margin-top: 5px;'>
                                    <span style='color: {"#2E8B57" if abs(player["pts"] - rf_pred) < 3 else "#C9082A"}'>
                                        Erreur: {abs(player["pts"] - rf_pred):.1f} pts
                                    </span>
                                </div>
                            </div>
                            <div class='model-card dnn-card'>
                                <div style='font-size: 20px; font-weight: 600; margin-bottom: 10px; color: #C9082A;'>Deep Neural Network</div>
                                <div style='font-size: 36px; font-weight: 700;'>{dnn_pred:.1f}</div>
                                <div style='color: #666; margin-top: 5px;'>
                                    <span style='color: {"#2E8B57" if abs(player["pts"] - dnn_pred) < 3 else "#C9082A"}'>
                                        Erreur: {abs(player["pts"] - dnn_pred):.1f} pts
                                    </span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Créer le graphique radar
                    categories = ['Scoring', 'Playmaking', 'Interior Game', 'Usage', 'Efficiency']
                    
                    fig = go.Figure()
                    
                    # Normaliser les valeurs pour le graphique radar
                    values_real = [
                        player['pts']/30 * 5,
                        player['ast']/10 * 5,
                        (player['reb'] + player['oreb_pct'] * 100)/15 * 5,
                        player['usg_pct']/40 * 5,
                        player['ts_pct'] * 5
                    ]
                    
                    values_rf = [
                        rf_pred/30 * 5,
                        player['ast']/10 * 5,
                        (player['reb'] + player['oreb_pct'] * 100)/15 * 5,
                        player['usg_pct']/40 * 5,
                        player['ts_pct'] * 5
                    ]
                    
                    values_dnn = [
                        dnn_pred/30 * 5,
                        player['ast']/10 * 5,
                        (player['reb'] + player['oreb_pct'] * 100)/15 * 5,
                        player['usg_pct']/40 * 5,
                        player['ts_pct'] * 5
                    ]
                    
                    # Ajouter les traces au radar avec des couleurs et styles améliorés
                    fig.add_trace(go.Scatterpolar(
                        r=values_real,
                        theta=categories,
                        fill='toself',
                        name='Réel',
                        line=dict(color='#2E8B57', width=2),
                        fillcolor='rgba(46, 139, 87, 0.2)'
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values_rf,
                        theta=categories,
                        fill='toself',
                        name='Random Forest',
                        line=dict(color='#17408B', width=2),
                        fillcolor='rgba(23, 64, 139, 0.2)'
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values_dnn,
                        theta=categories,
                        fill='toself',
                        name='DNN',
                        line=dict(color='#C9082A', width=2),
                        fillcolor='rgba(201, 8, 42, 0.2)'
                    ))
                    
                    # Mise à jour du layout
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 5],
                                showline=False,
                                color='#666',
                                tickfont=dict(size=10)
                            ),
                            angularaxis=dict(
                                color='#666',
                                tickfont=dict(size=12)
                            )
                        ),
                        showlegend=True,
                        legend=dict(
                            x=0.5,
                            y=-0.1,
                            xanchor='center',
                            orientation='h',
                            font=dict(size=12)
                        ),
                        title={
                            'text': "Profil du Joueur",
                            'y':0.95,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': dict(size=24, color='#17408B')
                        },
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:  # Manual Input mode
            with st.form("manual_input_form"):
                col1, col2, col3 = st.columns(3)
                
                # Basic Info
                with col1:
                    st.subheader("Basic Info")
                    age = st.number_input("Age", min_value=18, max_value=45, value=25)
                    height = st.number_input("Height (cm)", min_value=150, max_value=250, value=200)
                    weight = st.number_input("Weight (kg)", min_value=60, max_value=180, value=100)
                    
                # Game Stats
                with col2:
                    st.subheader("Game Stats")
                    games_played = st.number_input("Games Played", min_value=0, max_value=82, value=70)
                    rebounds = st.number_input("Rebounds per Game", min_value=0.0, max_value=20.0, value=5.0)
                    assists = st.number_input("Assists per Game", min_value=0.0, max_value=15.0, value=3.0)
                    
                # Advanced Stats
                with col3:
                    st.subheader("Advanced Stats")
                    oreb_pct = st.number_input("Offensive Rebound %", min_value=0.0, max_value=100.0, value=10.0)
                    dreb_pct = st.number_input("Defensive Rebound %", min_value=0.0, max_value=100.0, value=20.0)
                    usg_pct = st.number_input("Usage Rate %", min_value=0.0, max_value=100.0, value=20.0)
                    ts_pct = st.number_input("True Shooting %", min_value=0.0, max_value=100.0, value=55.0)
                    ast_pct = st.number_input("Assist %", min_value=0.0, max_value=100.0, value=15.0)
                
                # Team Info
                st.subheader("Team Info")
                col4, col5, col6 = st.columns(3)
                with col4:
                    team = st.selectbox("Team", sorted(nba_data['team_abbreviation'].unique()))
                with col5:
                    college = st.selectbox("College", sorted(nba_data['college'].unique()))
                with col6:
                    country = st.selectbox("Country", sorted(nba_data['country'].unique()))
                    
                submitted = st.form_submit_button("Predict Points")
                
                if submitted:
                    player_data = {
                        'age': age,
                        'player_height': height,
                        'player_weight': weight,
                        'gp': games_played,
                        'reb': rebounds,
                        'ast': assists,
                        'oreb_pct': oreb_pct,
                        'dreb_pct': dreb_pct,
                        'usg_pct': usg_pct,
                        'ts_pct': ts_pct,
                        'ast_pct': ast_pct,
                        'team_abbreviation': team,
                        'college': college,
                        'country': country
                    }
                    
                    rf_pred, dnn_pred = predict_points(player_data, rf_model, dnn_model, scaler, encoders)
              
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
                        
                        # Créer le graphique radar pour les prédictions
                        categories = ['Scoring', 'Playmaking', 'Rebounds', 'Usage', 'Efficiency']
                        
                        fig = go.Figure()
                        
                        # Normaliser les valeurs pour le graphique radar
                        values_rf = [
                            rf_pred/30 * 5,
                            assists/10 * 5,
                            rebounds/10 * 5,
                            usg_pct/40 * 5,
                            ts_pct/100 * 5
                        ]
                        
                        values_dnn = [
                            dnn_pred/30 * 5,
                            assists/10 * 5,
                            rebounds/10 * 5,
                            usg_pct/40 * 5,
                            ts_pct/100 * 5
                        ]
                        
                        # Ajouter les traces au radar
                        fig.add_trace(go.Scatterpolar(
                            r=values_rf,
                            theta=categories,
                            fill='toself',
                            name='Random Forest',
                            line_color='#17408B'
                        ))
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values_dnn,
                            theta=categories,
                            fill='toself',
                            name='DNN',
                            line_color='#C9082A'
                        ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                            showlegend=True,
                            title={
                                'text': "Player Profile Comparison",
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