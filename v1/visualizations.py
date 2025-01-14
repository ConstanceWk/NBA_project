import plotly.graph_objects as go
import streamlit as st

def display_player_card(player):
    """Display player information card"""
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

def display_predictions(player, rf_pred, dnn_pred):
    """Display prediction cards"""
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

def create_radar_chart(player, rf_pred, dnn_pred):
    """Create and display radar chart"""
    categories = ['Scoring', 'Playmaking', 'Interior Game', 'Usage', 'Efficiency']
    
    fig = go.Figure()
    
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
    
    # Configuration des traces
    traces = [
        ('Réel', values_real, '#2E8B57'),
        ('Random Forest', values_rf, '#17408B'),
        ('DNN', values_dnn, '#C9082A')
    ]
    
    for name, values, color in traces:
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name,
            line=dict(color=color, width=2),
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
        ))
    
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
    
# Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)

def display_manual_input_form(nba_data):
    """Affiche et gère le formulaire de saisie manuelle"""
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Info")
            age = st.number_input("Age", min_value=18, max_value=45, value=25)
            height = st.number_input("Height (cm)", min_value=150, max_value=250, value=200)
            weight = st.number_input("Weight (kg)", min_value=60, max_value=180, value=100)
        
        with col2:
            st.subheader("Game Stats")
            games_played = st.number_input("Games Played", min_value=0, max_value=82, value=70)
            rebounds = st.number_input("Rebounds per Game", min_value=0.0, max_value=20.0, value=5.0)
            assists = st.number_input("Assists per Game", min_value=0.0, max_value=15.0, value=3.0)
        
        with col3:
            st.subheader("Advanced Stats")
            oreb_pct = st.number_input("Offensive Rebound %", min_value=0.0, max_value=100.0, value=10.0)
            dreb_pct = st.number_input("Defensive Rebound %", min_value=0.0, max_value=100.0, value=20.0)
            usg_pct = st.number_input("Usage Rate %", min_value=0.0, max_value=100.0, value=20.0)
            ts_pct = st.number_input("True Shooting %", min_value=0.0, max_value=100.0, value=55.0)
            ast_pct = st.number_input("Assist %", min_value=0.0, max_value=100.0, value=15.0)
        
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
            return {
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
        return None