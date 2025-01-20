import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Configuration de la page
st.set_page_config(
    page_title="NBA Points Predictor",
    page_icon="🏀",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        padding: 20px;
        background: linear-gradient(120deg, #1e3799, #0c2461);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #3498db;
    }
    .player-info {
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre principal
st.markdown("""
    <div class="main-title">
        <h1>🏀 NBA Points Predictor</h1>
        <p>Prédiction des points marqués avec Deep Learning & Random Forest</p>
    </div>
""", unsafe_allow_html=True)

# Fonction pour charger les données et modèles
@st.cache_resource
def load_resources():
    # Chargement des données
    df = pd.read_csv('all_seasons.csv')
    
    # Encodage des variables catégorielles
    le = LabelEncoder()
    df['team_encoded'] = le.fit_transform(df['team_abbreviation'])
    df['college_encoded'] = le.fit_transform(df['college'].fillna('None'))
    df['country_encoded'] = le.fit_transform(df['country'])
    
    # Chargement des modèles
    dnn_model = joblib.load('dnn_model.joblib')
    rf_model = joblib.load('rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    return df, dnn_model, rf_model, scaler

# Chargement des ressources
try:
    df, dnn_model, rf_model, scaler = load_resources()
    
    # Layout principal avec deux colonnes
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("""
            ### 📊 Informations
            Cette application utilise l'intelligence artificielle pour prédire les points marqués par les joueurs NBA :
            - 🤖 Deep Neural Network
            - 🌳 Random Forest
            
            Les modèles ont été entraînés sur des données historiques de la NBA.
        """)
        
        # Métriques des modèles
        st.markdown("### 📈 Performance des modèles")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("DNN R²", "0.85")
        with metrics_col2:
            st.metric("RF R²", "0.87")
    
    with col1:
        tab1, tab2 = st.tabs(["🎲 Prédiction aléatoire", "✍️ Prédiction manuelle"])
        
        with tab1:
            st.markdown("### 🎯 Prédiction aléatoire")
            if st.button("Générer une nouvelle prédiction", key="predict_button"):
                with st.spinner("Génération de la prédiction..."):
                    # Sélection aléatoire d'un joueur
                    random_index = random.randint(0, len(df) - 1)
                    player_data = df.iloc[random_index]
                    
                    # Préparation des features
                    features = [
                        'age', 'player_height', 'player_weight', 'gp', 'reb', 'ast',
                        'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct',
                        'team_encoded', 'college_encoded', 'country_encoded'
                    ]
                    
                    X = player_data[features].values.reshape(1, -1)
                    X_scaled = scaler.transform(X)
                    
                    # Prédictions
                    dnn_pred = float(dnn_model.predict(X_scaled)[0])
                    rf_pred = float(rf_model.predict(X_scaled)[0])
                    real_points = float(player_data['pts'])
                    
                    # Affichage des informations du joueur
                    st.markdown(f"""
                        <div class="player-info">
                            <h3>🏀 {player_data['player_name']}</h3>
                            <p>Saison: {player_data['season']} | Équipe: {player_data['team_abbreviation']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Affichage des prédictions
                    pred_cols = st.columns(3)
                    with pred_cols[0]:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h4>Points réels</h4>
                                <p class="metric-value">{real_points:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_cols[1]:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h4>Prédiction DNN</h4>
                                <p class="metric-value">{dnn_pred:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_cols[2]:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h4>Prédiction RF</h4>
                                <p class="metric-value">{rf_pred:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Graphique de comparaison
                    fig = go.Figure()
                    categories = ['Points réels', 'Prédiction DNN', 'Prédiction RF']
                    values = [real_points, dnn_pred, rf_pred]
                    colors = ['#FFA07A', '#98FB98', '#87CEEB']
                    
                    fig.add_trace(go.Bar(
                        x=categories,
                        y=values,
                        marker_color=colors,
                        text=[f"{v:.1f}" for v in values],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title="Comparaison des prédictions",
                        title_x=0.5,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis_title="Points",
                        height=400,
                        margin=dict(t=50, l=50, r=30, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Entrez les statistiques du joueur")
            st.markdown("""
                <div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
                    ℹ️ Les champs sont pré-remplis avec des valeurs moyennes NBA. 
                    Vous pouvez les ajuster selon les caractéristiques du joueur.
                </div>
            """, unsafe_allow_html=True)
            
            # Création de colonnes pour organiser les champs de saisie
            input_col1, input_col2, input_col3 = st.columns(3)
            
            with input_col1:
                age = st.number_input(
                    "Âge", 
                    min_value=18, 
                    max_value=45, 
                    value=26,
                    help="L'âge moyen en NBA est de 26 ans"
                )
                player_height = st.number_input(
                    "Taille (cm)", 
                    min_value=160, 
                    max_value=230, 
                    value=198,
                    help="La taille moyenne en NBA est de 198cm"
                )
                player_weight = st.number_input(
                    "Poids (kg)", 
                    min_value=60, 
                    max_value=150, 
                    value=98,
                    help="Le poids moyen en NBA est de 98kg"
                )
                gp = st.number_input(
                    "Matchs joués", 
                    min_value=1, 
                    max_value=82, 
                    value=65,
                    help="Une saison NBA compte 82 matchs"
                )
            
            with input_col2:
                reb = st.number_input(
                    "Rebonds par match", 
                    min_value=0.0, 
                    max_value=20.0, 
                    value=6.0,
                    help="La moyenne NBA est d'environ 6 rebonds par match"
                )
                ast = st.number_input(
                    "Passes décisives par match", 
                    min_value=0.0, 
                    max_value=15.0, 
                    value=3.5,
                    help="La moyenne NBA est d'environ 3.5 passes par match"
                )
                oreb_pct = st.number_input(
                    "% Rebonds offensifs", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=8.5,
                    help="Pourcentage des rebonds offensifs disponibles capturés. Moyenne NBA ~8.5%"
                )
                dreb_pct = st.number_input(
                    "% Rebonds défensifs", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=18.0,
                    help="Pourcentage des rebonds défensifs disponibles capturés. Moyenne NBA ~18%"
                )
            
            with input_col3:
                usg_pct = st.number_input(
                    "% Usage", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=20.0,
                    help="Pourcentage des possessions utilisées. Un joueur star a ~30%, role player ~20%"
                )
                ts_pct = st.number_input(
                    "% True Shooting", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=55.0,
                    help="Mesure d'efficacité au tir. La moyenne NBA est ~55%"
                )
                ast_pct = st.number_input(
                    "% Passes décisives", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=15.0,
                    help="Pourcentage des paniers marqués par les coéquipiers assistés par le joueur"
                )

            # Sélection de l'équipe, du collège et du pays
            select_col1, select_col2, select_col3 = st.columns(3)
            
            with select_col1:
                team = st.selectbox(
                    "Équipe", 
                    sorted(df['team_abbreviation'].unique()),
                    help="L'équipe actuelle du joueur"
                )
            
            with select_col2:
                colleges = ['None'] + sorted(df['college'].dropna().unique())
                college = st.selectbox(
                    "Collège",
                    colleges,
                    help="Le collège d'où vient le joueur. 'None' si pas de collège"
                )
            
            with select_col3:
                country = st.selectbox(
                    "Pays",
                    sorted(df['country'].unique()),
                    index=sorted(df['country'].unique()).index('USA') if 'USA' in df['country'].unique() else 0,
                    help="Le pays d'origine du joueur"
                )

            # Bouton pour lancer la prédiction
            # Remplacer la partie de l'encodage des variables catégorielles par :
            if st.button("Prédire les points", key="manual_predict"):
                with st.spinner("Calcul de la prédiction..."):
                    # Créer un DataFrame temporaire pour l'encodage
                    temp_df = df.copy()
                    new_data = pd.DataFrame({
                        'team_abbreviation': [team],
                        'college': [college],
                        'country': [country],
                        'age': [age],
                        'player_height': [player_height],
                        'player_weight': [player_weight],
                        'gp': [gp],
                        'reb': [reb],
                        'ast': [ast],
                        'oreb_pct': [oreb_pct/100],  # Conversion en décimal
                        'dreb_pct': [dreb_pct/100],  # Conversion en décimal
                        'usg_pct': [usg_pct/100],    # Conversion en décimal
                        'ts_pct': [ts_pct/100],      # Conversion en décimal
                        'ast_pct': [ast_pct/100]     # Conversion en décimal
                    })
                    
                    # Ajouter les nouvelles données au DataFrame temporaire
                    temp_df = pd.concat([temp_df, new_data], ignore_index=True)
                    
                    # Encoder toutes les variables catégorielles
                    le = LabelEncoder()
                    temp_df['team_encoded'] = le.fit_transform(temp_df['team_abbreviation'])
                    temp_df['college_encoded'] = le.fit_transform(temp_df['college'].fillna('None'))
                    temp_df['country_encoded'] = le.fit_transform(temp_df['country'])
                    
                    # Récupérer les features encodées pour la dernière ligne (notre nouvelle donnée)
                    features = [
                        age, player_height, player_weight, gp, reb, ast,
                        oreb_pct/100, dreb_pct/100, usg_pct/100, ts_pct/100, ast_pct/100,
                        temp_df['team_encoded'].iloc[-1],
                        temp_df['college_encoded'].iloc[-1],
                        temp_df['country_encoded'].iloc[-1]
                    ]
                    
                    X = np.array(features).reshape(1, -1)
                    X_scaled = scaler.transform(X)
                    
                    # Faire les prédictions
                    dnn_pred = float(dnn_model.predict(X_scaled)[0])
                    rf_pred = float(rf_model.predict(X_scaled)[0])
                    
                    # Afficher les résultats
                    pred_cols = st.columns(2)
                    
                    with pred_cols[0]:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h4>Prédiction DNN</h4>
                                <p class="metric-value">{dnn_pred:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_cols[1]:
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h4>Prédiction RF</h4>
                                <p class="metric-value">{rf_pred:.1f}</p>
                            </div>""", unsafe_allow_html=True)
                    
                    # Graphique de comparaison
                    fig = go.Figure()
                    
                    categories = ['Prédiction DNN', 'Prédiction RF']
                    values = [dnn_pred, rf_pred]
                    colors = ['#98FB98', '#87CEEB']
                    
                    fig.add_trace(go.Bar(
                        x=categories,
                        y=values,
                        marker_color=colors,
                        text=[f"{v:.1f}" for v in values],
                        textposition='auto',
                    ))
                    
                    fig.update_layout(
                        title="Comparaison des prédictions",
                        title_x=0.5,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis_title="Points",
                        height=400,
                        margin=dict(t=50, l=50, r=30, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"""
        ### ⚠️ Erreur lors du chargement des ressources
        
        Assurez-vous que tous les fichiers nécessaires sont présents dans le répertoire :
        - all_seasons.csv
        - dnn_model.joblib
        - rf_model.joblib
        - scaler.joblib
        
        Erreur détaillée : {str(e)}
    """)