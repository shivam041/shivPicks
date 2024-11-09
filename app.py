import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import time
from requests.exceptions import ReadTimeout, ConnectionError
from fuzzywuzzy import process

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
            all_players = players.get_players()
            player_names = [player['full_name'] for player in all_players]

            matched_name, score = process.extractOne(player_name, player_names)

            if score < 80:
                st.warning(f"Player '{player_name}' not found. Did you mean '{matched_name}'?")
                player_name = matched_name

            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                st.error(f"Player '{player_name}' not found.")
                return None
            
            player_id = player_dict[0]['id']
            
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
            
            combined_data = pd.concat([current_season, previous_season], ignore_index=True)
            return combined_data
            
        except (ReadTimeout, ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error(f"Failed to fetch data for {player_name} after {max_retries} attempts")
                return None

def preprocess_game_log(game_log):
    game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'], format='%b %d, %Y')  
    game_log['HOME_AWAY'] = np.where(game_log['MATCHUP'].str.contains('@'), 'Away', 'Home')
    
    for col in ['PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TOV', 'PF', 'MIN']:
        game_log[col] = game_log[col].astype(float)

    game_log['AVG_PTS'] = game_log['PTS'].expanding().mean()
    game_log['AVG_AST'] = game_log['AST'].expanding().mean()
    game_log['AVG_REB'] = game_log['REB'].expanding().mean()
    game_log['MINUTES_PLAYED'] = game_log['MIN']
    game_log['FGM_PCT'] = game_log['FGM'] / game_log['FGA']
    game_log['FTM_PCT'] = game_log['FTM'] / game_log['FTA']
    game_log['FG3M_PCT'] = game_log['FG3M'] / game_log['FG3A']

    game_log = game_log.sort_values('GAME_DATE', ascending=True)
    game_log.dropna(inplace=True)

    return game_log

def train_linear_regression(game_log, target_col):
    features = game_log[['FTA', 'FT_PCT', 'TOV', 'PF', 'FGM', 'FGA', 'FTM', 'FG3M', 'AVG_PTS', 'AVG_AST', 'AVG_REB', 'MINUTES_PLAYED', 'FGM_PCT', 'FTM_PCT', 'FG3M_PCT']]
    target = game_log[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Test MSE ({target_col} - Linear Regression): {mse:.2f} (± {np.sqrt(mse):.2f})')

    return model, mse

def train_polynomial_regression(game_log, target_col, degree=2):
    features = game_log[['FTA', 'FT_PCT', 'TOV', 'PF', 'FGM', 'FGA', 'FTM', 'FG3M', 'AVG_PTS', 'AVG_AST', 'AVG_REB', 'MINUTES_PLAYED', 'FGM_PCT', 'FTM_PCT', 'FG3M_PCT']]
    target = game_log[target_col]
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(features)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Test MSE ({target_col} - Polynomial Regression): {mse:.2f} (± {np.sqrt(mse):.2f})')

    return model, poly, mse

def train_xgboost(game_log, target_col):
    features = game_log[['PTS', 'AST', 'REB', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TOV', 'PF', 'AVG_PTS', 'AVG_AST', 'AVG_REB', 'MINUTES_PLAYED', 'FGM_PCT', 'FTM_PCT', 'FG3M_PCT']]
    
    target = game_log[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(objective='reg:squarederror', 
                              n_estimators=100, 
                              learning_rate=0.1, 
                              max_depth=5, 
                              min_child_weight=1, 
                              subsample=0.8, 
                              colsample_bytree=0.8, 
                              gamma=0, 
                              random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Test MSE ({target_col} - XGBoost): {mse:.2f} (± {np.sqrt(mse):.2f})')

    return model, mse

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Prediction Type", ["Points", "Assists", "Rebounds"])

    home_team = st.selectbox("Select Home Team", options=[team['abbreviation'] for team in teams.get_teams()])
    away_team = st.selectbox("Select Away Team", options=[team['abbreviation'] for team in teams.get_teams()])
    
    model_type = st.selectbox("Select Model Type", ["XGBoost", "Linear Regression", "Polynomial Regression"])

    if st.button("Generate Predictions"):
        with st.spinner("Generating predictions..."):
            teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
            for team, opponent in teams_to_analyze:
                roster = get_team_roster(team)
                st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")
                
                predictions = {}
                total_predicted_score = 0
                for player_name in roster:
                    game_log = get_player_data(player_name)
                    if game_log is None or len(game_log) < 5:
                        st.warning(f"Insufficient data for {player_name}")
                        continue
                        
                    try:
                        processed_log = preprocess_game_log(game_log)
                        if model_type == "XGBoost":
                            model, mse = train_xgboost(processed_log, 'PTS' if page == "Points" else 'AST' if page == "Assists" else 'REB')
                            pred = model.predict(processed_log[['FTA', 'FT_PCT', 'TOV', 'PF', 'FGM', 'FGA', 'FTM', 'FG3M', 'AVG_PTS', 'AVG_AST', 'AVG_REB', 'MINUTES_PLAYED', 'FGM_PCT', 'FTM_PCT', 'FG3M_PCT']].mean().values.reshape(1, -1))
                        elif model_type == "Polynomial Regression":
                            model, poly, mse = train_polynomial_regression(processed_log, 'PTS' if page == "Points" else 'AST' if page == "Assists" else 'REB')
                            pred = model.predict(poly.transform(processed_log[['FTA', 'FT_PCT', 'TOV', 'PF', 'FGM', 'FGA', 'FTM', 'FG3M', 'AVG_PTS', 'AVG_AST', 'AVG_REB', 'MINUTES_PLAYED', 'FGM_PCT', 'FTM_PCT', 'FG3M_PCT']].mean().values.reshape(1, -1)))
                        else:
                            model, mse = train_linear_regression(processed_log, 'PTS' if page == "Points" else 'AST' if page == "Assists" else 'REB')
                            pred = model.predict(processed_log[['FTA', 'FT_PCT', 'TOV', 'PF', 'FGM', 'FGA', 'FTM', 'FG3M', 'AVG_PTS', 'AVG_AST', 'AVG_REB', 'MINUTES_PLAYED', 'FGM_PCT', 'FTM_PCT', 'FG3M_PCT']].mean().values.reshape(1, -1))
                        
                        if pred is not None:
                            predictions[player_name] = pred[0]
                            total_predicted_score += pred[0]
                    except Exception as e:
                        st.error(f"Error processing {player_name}: {e}")
                        continue
                
                st.write(f"\n{'='*50}")
                st.write(f"{model_type} Predictions for {team} against {opponent}:")
                st.write(f"{'='*50}")
                
                for player_name, points in predictions.items():
                    if page == "Assists":
                        st.write(f"{player_name}: {points:.1f} points (MSE: {mse:.2f})")
                    else:
                        st.write(f"{player_name}: {points:.1f} points")
                
                st.write(f"\nTotal Predicted Score for {team}: {total_predicted_score:.1f} points")

if __name__ == "__main__":
    main()
