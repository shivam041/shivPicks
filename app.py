# MODEL/app.py

import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb  # Import XGBoost
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time 
from requests.exceptions import ReadTimeout, ConnectionError

# Set page config
st.set_page_config(
    page_title="NBA Game Predictions",
    page_icon="🏀",
    layout="wide"
)

# Add CSS styling
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">NBA Game Predictions 🏀</p>', unsafe_allow_html=True)

# Common Functions
def get_team_roster(team_abbreviation, retries=3, delay=5):
    for attempt in range(retries):
        try:
            team_info = teams.find_team_by_abbreviation(team_abbreviation)
            if not team_info:
                st.error(f"Team '{team_abbreviation}' not found.")
                return []

            team_id = team_info['id']
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, timeout=60).get_data_frames()[0]
            return roster['PLAYER'].tolist()
        except ReadTimeout:
            if attempt < retries - 1:
                st.warning(f"Timeout, retrying... ({attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                st.error("Failed to fetch team roster after multiple attempts.")
                return []
        except Exception as e:
            st.error(f"Error getting roster: {e}")
            return []

@st.cache_data
def get_player_data(player_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                st.error(f"Player '{player_name}' not found.")
                return None
            
            player_id = player_dict[0]['id']
            
            # Get data for both seasons
            current_season = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season='2024-25',
                timeout=120
            ).get_data_frames()[0]

            time.sleep(1)

            previous_season = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season='2023-24',
                timeout=120
            ).get_data_frames()[0]
            
            # Combine the data
            combined_data = pd.concat([current_season, previous_season], ignore_index=True)
            return combined_data
            
        except (ReadTimeout, ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error(f"Failed to fetch data for {player_name} after {max_retries} attempts")
                return None

def preprocess_game_log(game_log):
    # Specify the format for the date parsing
    game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'], format='%b %d, %Y')  
    game_log['HOME_AWAY'] = np.where(game_log['MATCHUP'].str.contains('@'), 'Away', 'Home')
    
    for col in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'FGM', 'FGA', 'FG_PCT', 
                'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
                'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']:
        game_log[col] = game_log[col].astype(float)

    game_log = game_log.sort_values('GAME_DATE', ascending=True)

    game_log.dropna(inplace=True)

    return game_log

# XGBoost Functions
def train_xgboost(game_log):
    features = game_log[['PTS', 'REB', 'AST', 'BLK', 'STL', 
                         'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
    
    target = game_log['PTS']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', 
                              n_estimators=100, 
                              learning_rate=0.1, 
                              max_depth=5, 
                              min_child_weight=1, 
                              subsample=0.8, 
                              colsample_bytree=0.8, 
                              gamma=0, 
                              random_state=42)  # Initialize XGBoost with hyperparameters
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Test MSE (XGBoost): {mse:.2f} (± {np.sqrt(mse):.2f})')  # Display MSE with ± notation

    return model, mse  # Return model and MSE

def predict_performance_against_team_xgboost(model, game_log, opponent_team):
    opponent_games = game_log[game_log['MATCHUP'].str.contains(opponent_team)]
    
    if opponent_games.empty:
        return None

    avg_stats = opponent_games[['PTS', 'REB', 'AST', 'BLK', 'STL', 
                               'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                               'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 
                               'PF', 'PLUS_MINUS']].mean()

    features = np.array(avg_stats).reshape(1, -1)  # Reshape for XGBoost input
    return model.predict(features)[0]

# LSTM Functions
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))  # Increased LSTM units
    model.add(Dropout(0.3))  # Adjusted dropout rate
    model.add(LSTM(50))
    model.add(Dropout(0.3))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data_for_lstm(game_log):
    features = game_log[['PTS', 'REB', 'AST', 'BLK', 'STL', 
                         'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].values
    
    # Reshape data to be [samples, time steps, features]
    X, y = [], []
    for i in range(len(features) - 1):
        X.append(features[i:i + 1])  # Use the previous game as input
        y.append(features[i + 1, 0])  # Predict the points of the next game
    return np.array(X), np.array(y)

def train_lstm(game_log):
    X, y = prepare_data_for_lstm(game_log)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))  # Input shape for LSTM
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Early stopping
    model.fit(X_train, y_train, epochs=200, batch_size=10, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Test MSE (LSTM): {mse:.2f} (± {np.sqrt(mse):.2f})')  # Display MSE with ± notation

    return model, mse  # Return model and MSE

def predict_performance_against_team_lstm(model, game_log, opponent_team):
    opponent_games = game_log[game_log['MATCHUP'].str.contains(opponent_team)]
    
    if opponent_games.empty:
        return None

    avg_stats = opponent_games[['PTS', 'REB', 'AST', 'BLK', 'STL', 
                               'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                               'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 
                               'PF', 'PLUS_MINUS']].mean()

    features = np.array(avg_stats).reshape(1, 1, -1)  # Reshape for LSTM input
    return model.predict(features)[0][0]

# Monte Carlo Functions
def monte_carlo_simulation(game_log, opponent_team, num_simulations=10000, confidence_level=0.95):
    opponent_games = game_log[game_log['MATCHUP'].str.contains(opponent_team)]
    
    if opponent_games.empty:
        st.warning(f"No previous games found against team: {opponent_team}.")
        return None

    stats = {}
    stat_mapping = {
        'PTS': 'points'
    }
    
    for stat_key, stat_name in stat_mapping.items():
        values = opponent_games[stat_key].values
        mean_val = values.mean()
        std_val = values.std() if len(values) > 1 else 0
        
        simulated_values = np.random.normal(loc=mean_val, scale=std_val, size=num_simulations)
        
        stats[stat_name] = {
            "mean": simulated_values.mean(),
            "std": simulated_values.std(),
            "max": simulated_values.max(),  # Maximum predicted points
            "min": simulated_values.min()   # Minimum predicted points
        }
    
    return stats

def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    home_team = st.selectbox("Select Home Team", options=[team['abbreviation'] for team in teams.get_teams()])
    away_team = st.selectbox("Select Away Team", options=[team['abbreviation'] for team in teams.get_teams()])
    
    model_type = st.selectbox("Select Model Type", ["Monte Carlo Simulation", "LSTM", "XGBoost"])

    if st.button("Generate Predictions"):
        with st.spinner("Generating predictions..."):
            teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
            for team, opponent in teams_to_analyze:
                roster = get_team_roster(team)
                st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")
                
                if model_type == "LSTM":
                    lstm_predictions = {}
                    total_predicted_score = 0  # Initialize total predicted score
                    for player_name in roster:
                        game_log = get_player_data(player_name)
                        if game_log is None or len(game_log) < 5:
                            st.warning(f"Insufficient data for {player_name}")
                            continue
                            
                        try:
                            processed_log = preprocess_game_log(game_log)
                            model, mse = train_lstm(processed_log)  # Get model and MSE
                            lstm_pred = predict_performance_against_team_lstm(model, processed_log, opponent)
                            if lstm_pred is not None:
                                lstm_predictions[player_name] = lstm_pred
                                total_predicted_score += lstm_pred  # Add player's predicted score to total
                        except Exception as e:
                            st.error(f"Error processing {player_name}: {e}")
                            continue
                    
                    # Display results
                    st.write(f"\n{'='*50}")
                    st.write(f"LSTM Predictions for {team} against {opponent}:")
                    st.write(f"{'='*50}")
                    
                    for player_name, lstm_points in lstm_predictions.items():
                        st.write(f"{player_name}: {lstm_points:.1f} points (± {np.sqrt(mse):.2f})")  # Display prediction with ± MSE
                    
                    # Display total predicted score for the team
                    st.write(f"\nTotal Predicted Score for {team}: {total_predicted_score:.1f} points")
                
                elif model_type == "XGBoost":
                    xgboost_predictions = {}
                    total_predicted_score = 0  # Initialize total predicted score
                    for player_name in roster:
                        game_log = get_player_data(player_name)
                        if game_log is None or len(game_log) < 5:
                            st.warning(f"Insufficient data for {player_name}")
                            continue
                            
                        try:
                            processed_log = preprocess_game_log(game_log)
                            model, mse = train_xgboost(processed_log)  # Get model and MSE
                            xgboost_pred = predict_performance_against_team_xgboost(model, processed_log, opponent)
                            if xgboost_pred is not None:
                                xgboost_predictions[player_name] = xgboost_pred
                                total_predicted_score += xgboost_pred  # Add player's predicted score to total
                        except Exception as e:
                            st.error(f"Error processing {player_name}: {e}")
                            continue
                    
                    # Display results
                    st.write(f"\n{'='*50}")
                    st.write(f"XGBoost Predictions for {team} against {opponent}:")
                    st.write(f"{'='*50}")
                    
                    for player_name, xgboost_points in xgboost_predictions.items():
                        st.write(f"{player_name}: {xgboost_points:.1f} points (± {np.sqrt(mse):.2f})")  # Display prediction with ± MSE
                    
                    # Display total predicted score for the team
                    st.write(f"\nTotal Predicted Score for {team}: {total_predicted_score:.1f} points")
                
                elif model_type == "Monte Carlo Simulation":
                    monte_carlo_predictions = {}
                    for player_name in roster:
                        game_log = get_player_data(player_name)
                        if game_log is None or len(game_log) < 5:
                            st.warning(f"Insufficient data for {player_name}")
                            continue
                            
                        try:
                            processed_log = preprocess_game_log(game_log)
                            mc_pred = monte_carlo_simulation(processed_log, opponent)
                            if mc_pred is not None:
                                monte_carlo_predictions[player_name] = mc_pred
                        except Exception as e:
                            st.error(f"Error processing {player_name}: {e}")
                            continue
                    
                    # Display results
                    st.write(f"\n{'='*50}")
                    st.write(f"Monte Carlo Simulation Predictions for {team} against {opponent}:")
                    st.write(f"{'='*50}")
                    
                    for player_name, mc_stats in monte_carlo_predictions.items():
                        st.write(f"{player_name}: Mean: {mc_stats['points']['mean']:.1f}, Max: {mc_stats['points']['max']:.1f}, Min: {mc_stats['points']['min']:.1f} (± {mc_stats['points']['std']:.1f})")

if __name__ == "__main__":
    main()
