import pandas as pd
import numpy as np
import os
import io
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import VotingClassifier

# Import tree-based models
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Import Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage

# --- Step 1: Initialize Firebase Admin SDK ---
def initialize_firebase_admin(service_account_json_str):
    """
    Initializes the Firebase Admin SDK from a service account JSON string.
    This is ideal for use with environment variables or cloud secrets.
    """
    try:
        if not firebase_admin._apps:
            # Use io.StringIO to treat the JSON string as a file-like object
            cred = credentials.Certificate(io.StringIO(service_account_json_str))
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'sports-betting-model-fea2f.appspot.com'
            })
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        print("Please ensure your 'FIREBASE_SERVICE_KEY' environment variable is correctly set.")
        exit()

# --- Step 2: Load Data from Firebase Storage ---
def load_data_from_firebase(bucket_name, source_path, filenames_to_load):
    """
    Loads dataframes from a specified Firebase Storage bucket and path.
    """
    print(f"--- Loading data from gs://{bucket_name}/{source_path} for modeling ---")
    dataframes = {}
    bucket = storage.bucket()

    for key, filename in filenames_to_load.items():
        blob_path = os.path.join(source_path, filename)
        blob = bucket.blob(blob_path)

        if blob.exists():
            try:
                file_bytes = blob.download_as_bytes()
                file_buffer = io.BytesIO(file_bytes)
                dataframes[key] = pd.read_parquet(file_buffer)
                print(f"Loaded {filename} (Shape: {dataframes[key].shape})")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                dataframes[key] = pd.DataFrame()
        else:
            print(f"Error: File not found at gs://{bucket_name}/{blob_path}")
            dataframes[key] = pd.DataFrame()
    return dataframes

# --- Step 3: Define and Execute the Backtesting Pipeline ---
def run_backtesting_pipeline(df_final_game_data, df_advanced_pbp_data):
    """
    Performs backtesting on an ensemble model with an expanding window.
    """
    print("\n--- Starting Model Backtesting Pipeline ---")

    if df_final_game_data is None or df_final_game_data.empty:
        print("Final game data is empty. Cannot run modeling pipeline.")
        return
    
    if df_advanced_pbp_data is None or df_advanced_pbp_data.empty:
        print("Advanced PBP data is empty. Cannot run modeling pipeline.")
        # Continue with available data
        
    # Merge the two dataframes
    if not df_advanced_pbp_data.empty:
        df_final_game_data = pd.merge(df_final_game_data, df_advanced_pbp_data, on=['game_id', 'season', 'week'], how='left')
        print("Successfully merged advanced PBP data with game data.")

    # Sort data by date to ensure proper time-series split
    df_final_game_data['gameday'] = pd.to_datetime(df_final_game_data['gameday'])
    df_final_game_data = df_final_game_data.sort_values(by='gameday').reset_index(drop=True)

    # Function to convert American odds to implied probability
    def american_odds_to_implied_probability(odds):
        if pd.isna(odds):
            return np.nan
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    # Function to convert American odds to decimal odds
    def american_to_decimal_odds(odds):
        if pd.isna(odds):
            return np.nan
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    # Apply the function to create new implied probability and decimal odds features
    # These are crucial for the value betting strategy
    if 'home_team_moneyline' in df_final_game_data.columns:
        df_final_game_data['home_implied_moneyline_prob'] = df_final_game_data['home_team_moneyline'].apply(american_odds_to_implied_probability)
        df_final_game_data['home_moneyline_decimal'] = df_final_game_data['home_team_moneyline'].apply(american_to_decimal_odds)
    if 'away_team_moneyline' in df_final_game_data.columns:
        df_final_game_data['away_implied_moneyline_prob'] = df_final_game_data['away_team_moneyline'].apply(american_odds_to_implied_probability)
        df_final_game_data['away_moneyline_decimal'] = df_final_game_data['away_team_moneyline'].apply(american_to_decimal_odds)
    if 'home_team_spread_odds' in df_final_game_data.columns:
        df_final_game_data['home_implied_spread_prob'] = df_final_game_data['home_team_spread_odds'].apply(american_odds_to_implied_probability)
        df_final_game_data['home_spread_decimal'] = df_final_game_data['home_team_spread_odds'].apply(american_to_decimal_odds)
    if 'away_team_spread_odds' in df_final_game_data.columns:
        df_final_game_data['away_implied_spread_prob'] = df_final_game_data['away_team_spread_odds'].apply(american_odds_to_implied_probability)
        df_final_game_data['away_spread_decimal'] = df_final_game_data['away_team_spread_odds'].apply(american_to_decimal_odds)
    if 'total_line' in df_final_game_data.columns and 'total_over_odds' in df_final_game_data.columns:
        df_final_game_data['over_implied_prob'] = df_final_game_data['total_over_odds'].apply(american_odds_to_implied_probability)
        df_final_game_data['over_decimal'] = df_final_game_data['total_over_odds'].apply(american_to_decimal_odds)
    if 'total_line' in df_final_game_data.columns and 'total_under_odds' in df_final_game_data.columns:
        df_final_game_data['under_implied_prob'] = df_final_game_data['total_under_odds'].apply(american_odds_to_implied_probability)
        df_final_game_data['under_decimal'] = df_final_game_data['total_under_odds'].apply(american_to_decimal_odds)

    print("Added implied probability and decimal odds features.")

    # Create target for "home team covering the spread", handling pushes as NaN
    if 'home_team_spread_line' in df_final_game_data.columns and 'score_differential' in df_final_game_data.columns:
        df_final_game_data['home_cover_spread'] = np.where(
            df_final_game_data['score_differential'] + df_final_game_data['home_team_spread_line'] > 0, 1,
            np.where(df_final_game_data['score_differential'] + df_final_game_data['home_team_spread_line'] < 0, 0, np.nan)
        )
        print("Added 'home_cover_spread' target variable (pushes treated as NaN).")

    # Re-calculate 'total_over' to explicitly set pushes to NaN for classification
    if 'total' in df_final_game_data.columns and 'total_line' in df_final_game_data.columns:
        df_final_game_data['total_over'] = np.where(
            df_final_game_data['total'] > df_final_game_data['total_line'], 1,
            np.where(df_final_game_data['total'] < df_final_game_data['total_line'], 0, np.nan)
        )
        print("Recalculated 'total_over' target variable (pushes treated as NaN).")

    # Define Features (X) for the model
    features = [
        'home_rolling_team_score_3_games', 'away_rolling_team_score_3_games',
        'home_rolling_opponent_score_3_games', 'away_rolling_opponent_score_3_games',
        'home_rolling_team_score_differential_3_games', 'away_rolling_team_score_differential_3_games',
        'home_rolling_team_score_5_games', 'away_rolling_team_score_5_games',
        'home_rolling_opponent_score_5_games', 'away_rolling_opponent_score_5_games',
        'home_rolling_team_score_differential_5_games', 'away_rolling_team_score_differential_5_games',
        'home_rolling_team_score_8_games', 'away_rolling_team_score_8_games',
        'home_rolling_opponent_score_8_games', 'away_rolling_opponent_score_8_games',
        'home_rolling_team_score_differential_8_games', 'away_rolling_team_score_differential_8_games',
        'home_team_rest', 'away_team_rest'
    ]

    advanced_pbp_features = [
        'home_rolling_redzone_td_pct_3_games', 'away_rolling_redzone_td_pct_3_games',
        'home_rolling_redzone_td_pct_5_games', 'away_rolling_redzone_td_pct_5_games',
        'home_rolling_redzone_td_pct_8_games', 'away_rolling_redzone_td_pct_8_games',
        'home_rolling_third_down_conv_pct_3_games', 'away_rolling_third_down_conv_pct_3_games',
        'home_rolling_third_down_conv_pct_5_games', 'away_rolling_third_down_conv_pct_5_games',
        'home_rolling_third_down_conv_pct_8_games', 'away_rolling_third_down_conv_pct_8_games',
        'home_rolling_turnover_differential_3_games', 'away_rolling_turnover_differential_3_games',
        'home_rolling_turnover_differential_5_games', 'away_rolling_turnover_differential_5_games',
        'home_rolling_turnover_differential_8_games', 'away_rolling_turnover_differential_8_games',
        'home_rolling_offensive_epa_per_play_3_games', 'away_rolling_offensive_epa_per_play_3_games',
        'home_rolling_offensive_epa_per_play_5_games', 'away_rolling_offensive_epa_per_play_5_games',
        'home_rolling_offensive_epa_per_play_8_games', 'away_rolling_offensive_epa_per_play_8_games',
        'home_rolling_defensive_epa_per_play_allowed_3_games', 'away_rolling_defensive_epa_per_play_allowed_3_games',
        'home_rolling_defensive_epa_per_play_allowed_5_games', 'away_rolling_defensive_epa_per_play_allowed_5_games',
        'home_rolling_defensive_epa_per_play_allowed_8_games', 'away_rolling_defensive_epa_per_play_allowed_8_games',
        'home_rolling_net_yards_per_play_3_games', 'away_rolling_net_yards_per_play_3_games',
        'home_rolling_net_yards_per_play_5_games', 'away_rolling_net_yards_per_play_5_games',
        'home_rolling_net_yards_per_play_8_games', 'away_rolling_net_yards_per_play_8_games',
        'home_rolling_penalty_differential_3_games', 'away_rolling_penalty_differential_3_games',
        'home_rolling_penalty_differential_5_games', 'away_rolling_penalty_differential_5_games',
        'home_rolling_penalty_differential_8_games', 'away_rolling_penalty_differential_8_games',
        'home_rolling_offensive_explosive_play_rate_3_games', 'away_rolling_offensive_explosive_play_rate_3_games',
        'home_rolling_offensive_explosive_play_rate_5_games', 'away_rolling_offensive_explosive_play_rate_5_games',
        'home_rolling_offensive_explosive_play_rate_8_games', 'away_rolling_offensive_explosive_play_rate_8_games',
        'home_rolling_defensive_explosive_play_rate_allowed_3_games', 'away_rolling_defensive_explosive_play_rate_allowed_3_games',
        'home_rolling_defensive_explosive_play_rate_allowed_5_games', 'away_rolling_defensive_explosive_play_rate_allowed_5_games',
        'home_rolling_defensive_explosive_play_rate_allowed_8_games', 'away_rolling_defensive_explosive_play_rate_allowed_8_games',
        'home_rolling_offensive_success_rate_3_games', 'away_rolling_offensive_success_rate_3_games',
        'home_rolling_offensive_success_rate_5_games', 'away_rolling_offensive_success_rate_5_games',
        'home_rolling_offensive_success_rate_8_games', 'away_rolling_offensive_success_rate_8_games',
        'home_rolling_offensive_drive_success_rate_3_games', 'away_rolling_offensive_drive_success_rate_3_games',
        'home_rolling_offensive_drive_success_rate_5_games', 'away_rolling_offensive_drive_success_rate_5_games',
        'home_rolling_offensive_drive_success_rate_8_games', 'away_rolling_offensive_drive_success_rate_8_games',
        'home_rolling_avg_starting_field_position_3_games', 'away_rolling_avg_starting_field_position_3_games',
        'home_rolling_avg_starting_field_position_5_games', 'away_rolling_avg_starting_field_position_5_games',
        'home_rolling_avg_starting_field_position_8_games', 'away_rolling_avg_starting_field_position_8_games'
    ]
    
    features.extend([col for col in advanced_pbp_features if col in df_final_game_data.columns])

    player_performance_features = [
        'home_team_rolling_fantasy_points_3_weeks', 'away_team_rolling_fantasy_points_3_weeks',
        'home_team_rolling_fantasy_points_5_weeks', 'away_team_rolling_fantasy_points_5_weeks',
        'home_team_rolling_fantasy_points_8_weeks', 'away_team_rolling_fantasy_points_8_weeks'
    ]
    features.extend([col for col in player_performance_features if col in df_final_game_data.columns])

    weather_features = ['temperature', 'wind', 'precipitation']
    features.extend([col for col in weather_features if col in df_final_game_data.columns])

    matchup_features = [
        'home_off_vs_away_def_epa_diff',
        'away_off_vs_home_def_epa_diff'
    ]
    if 'home_rolling_offensive_epa_per_play_5_games' in df_final_game_data.columns and \
       'away_rolling_defensive_epa_per_play_allowed_5_games' in df_final_game_data.columns:
        df_final_game_data['home_off_vs_away_def_epa_diff'] = df_final_game_data['home_rolling_offensive_epa_per_play_5_games'] - df_final_game_data['away_rolling_defensive_epa_per_play_allowed_5_games']
        df_final_game_data['away_off_vs_home_def_epa_diff'] = df_final_game_data['away_rolling_offensive_epa_per_play_5_games'] - df_final_game_data['home_rolling_defensive_epa_per_play_allowed_5_games']
        features.extend(matchup_features)

    odds_features_to_add = [
        'home_team_moneyline', 'away_team_moneyline',
        'home_team_spread_line', 'away_team_spread_line',
        'home_team_spread_odds', 'away_team_spread_odds',
        'home_implied_moneyline_prob', 'away_implied_moneyline_prob',
        'home_implied_spread_prob', 'away_implied_spread_prob'
    ]
    existing_odds_features = [col for col in odds_features_to_add if col in df_final_game_data.columns]
    features.extend(existing_odds_features)

    pbp_diff_metrics = [
        'rolling_net_yards_per_play',
        'rolling_penalty_differential',
        'rolling_offensive_explosive_play_rate',
        'rolling_defensive_explosive_play_rate_allowed',
        'rolling_offensive_success_rate',
        'rolling_offensive_drive_success_rate',
        'rolling_avg_starting_field_position'
    ]

    for metric_base in pbp_diff_metrics:
        for window in [3, 5, 8]:
            home_col = f'home_{metric_base}_{window}_games'
            away_col = f'away_{metric_base}_{window}_games'
            if home_col in df_final_game_data.columns and away_col in df_final_game_data.columns:
                df_final_game_data[f'{metric_base}_diff_{window}_games'] = \
                    df_final_game_data[home_col] - df_final_game_data[away_col]
                features.append(f'{metric_base}_diff_{window}_games')

    # Fill any remaining NaNs in features with 0
    for col in features:
        if col in df_final_game_data.columns and df_final_game_data[col].isnull().any():
            df_final_game_data[col] = df_final_game_data[col].fillna(0)

    # AI-Driven Feature Discovery (Interaction and Polynomial Features)
    ai_feature_candidates = [
        'home_team_spread_line', 'away_team_spread_line',
        'home_implied_moneyline_prob', 'away_implied_moneyline_prob',
        'home_rolling_defensive_epa_per_play_allowed_8_games',
        'away_rolling_defensive_epa_per_play_allowed_8_games',
        'home_rolling_turnover_differential_8_games',
        'away_rolling_turnover_differential_8_games'
    ]
    ai_feature_candidates = [f for f in ai_feature_candidates if f in df_final_game_data.columns]

    if ai_feature_candidates:
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        poly_features = poly.fit_transform(df_final_game_data[ai_feature_candidates])
        poly_feature_names = poly.get_feature_names_out(ai_feature_candidates)
        df_poly_features = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_final_game_data.index)
        for col in df_poly_features.columns:
            if col not in df_final_game_data.columns and col not in features:
                df_final_game_data[col] = df_poly_features[col]
                features.append(col)

    # Time-Series Split (Expanding Window) parameters
    TEST_START_YEAR = 2009
    FINAL_TEST_YEAR = 2024
    MAX_REGULAR_SEASON_WEEK = 18

    # Define all target variables to model (keeping them consistent with previous script)
    available_targets = []
    if 'home_win' in df_final_game_data.columns and 'home_implied_moneyline_prob' in df_final_game_data.columns and 'away_implied_moneyline_prob' in df_final_game_data.columns:
        available_targets.append('home_win')

    if 'total_over' in df_final_game_data.columns and 'over_implied_prob' in df_final_game_data.columns and 'under_implied_prob' in df_final_game_data.columns:
        available_targets.append('total_over')
    
    if 'home_cover_spread' in df_final_game_data.columns and 'home_implied_spread_prob' in df_final_game_data.columns and 'away_implied_spread_prob' in df_final_game_data.columns:
        available_targets.append('home_cover_spread')
    
    TARGETS = available_targets

    # --- Backtesting Parameters ---
    BET_UNIT = 10  # Bet $10 on each identified value bet
    VALUE_THRESHOLD = 0.02  # Minimum 2% edge required to place a bet (model prob - implied prob)

    # Dictionary to store overall backtesting results for each target
    overall_backtesting_results = {}

    for TARGET in TARGETS:
        if TARGET not in df_final_game_data.columns or df_final_game_data[TARGET].isnull().all():
            print(f"\nSkipping backtesting for '{TARGET}': Target column not found or contains all NaNs.")
            continue

        print(f"\n--- Starting Backtesting for Target: '{TARGET}' ---")

        total_profit = 0
        total_wagered = 0
        total_bets_placed = 0
        total_wins = 0

        bet_results_list = []

        df_current_target_data = df_final_game_data.copy()

        for test_year in range(TEST_START_YEAR, FINAL_TEST_YEAR + 1):
            train_df = df_current_target_data[df_current_target_data['season'] < test_year].copy()

            for test_week in sorted(df_current_target_data[df_current_target_data['season'] == test_year]['week'].unique()):
                if test_week > MAX_REGULAR_SEASON_WEEK:
                    continue

                test_df_week = df_current_target_data[(df_current_target_data['season'] == test_year) & (df_current_target_data['week'] == test_week)].copy()

                if train_df.empty or test_df_week.empty:
                    continue

                # Drop rows where the target variable is NaN (e.g., for pushes in spread/total)
                valid_train_indices = train_df[train_df[TARGET].notna()].index
                valid_test_indices = test_df_week[test_df_week[TARGET].notna()].index

                X_train = train_df.loc[valid_train_indices, features]
                y_train = train_df.loc[valid_train_indices, TARGET]
                X_test = test_df_week.loc[valid_test_indices, features]
                y_test = test_df_week.loc[valid_test_indices, TARGET]
                
                test_df_week_filtered = test_df_week.loc[valid_test_indices].copy()

                if X_train.empty or X_test.empty:
                    continue

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Define the ensemble of models
                neg_count = (y_train == 0).sum()
                pos_count = (y_train == 1).sum()
                scale_pos_weight_value = neg_count / pos_count if pos_count != 0 else 1

                xgb_model = XGBClassifier(
                    objective='binary:logistic', eval_metric='logloss', enable_categorical=False, random_state=42,
                    n_estimators=900, learning_rate=0.01, max_depth=6, min_child_weight=1,
                    subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=0.0008, reg_lambda=1.2,
                    scale_pos_weight=scale_pos_weight_value
                )
                lr_model = LogisticRegression(
                    solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000
                )
                lgbm_model = LGBMClassifier(
                    objective='binary', metric='binary_logloss', random_state=42,
                    n_estimators=750, learning_rate=0.02, num_leaves=30, max_depth=-1,
                    min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.001, reg_lambda=0.1,
                )

                ensemble_model = VotingClassifier(
                    estimators=[('xgb', xgb_model), ('lr', lr', lr_model), ('lgbm', lgbm_model)],
                    voting='soft', weights=[0.50, 0.05, 0.45]
                )

                ensemble_model.fit(X_train_scaled, y_train)

                y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]

                # --- Betting Simulation for the current week ---
                for i, (idx, row) in enumerate(test_df_week_filtered.iterrows()):
                    model_prob = y_pred_proba[i]

                    implied_prob = np.nan
                    decimal_odds = np.nan
                    bet_side = None
                    true_outcome_for_bet = np.nan

                    if TARGET == 'home_win':
                        if model_prob >= 0.5:
                            if 'home_implied_moneyline_prob' in row and pd.notna(row['home_implied_moneyline_prob']):
                                implied_prob = row['home_implied_moneyline_prob']
                                decimal_odds = row['home_moneyline_decimal']
                                true_outcome_for_bet = row['home_win']
                                bet_side = 'home_moneyline'
                        else:
                            if 'away_implied_moneyline_prob' in row and pd.notna(row['away_implied_moneyline_prob']):
                                implied_prob = row['away_implied_moneyline_prob']
                                decimal_odds = row['away_moneyline_decimal']
                                model_prob = 1 - model_prob
                                true_outcome_for_bet = 1 - row['home_win']
                                bet_side = 'away_moneyline'

                    elif TARGET == 'total_over':
                        if model_prob >= 0.5:
                            if 'over_implied_prob' in row and pd.notna(row['over_implied_prob']):
                                implied_prob = row['over_implied_prob']
                                decimal_odds = row['over_decimal']
                                true_outcome_for_bet = row['total_over']
                                bet_side = 'over_total'
                        else:
                            if 'under_implied_prob' in row and pd.notna(row['under_implied_prob']):
                                implied_prob = row['under_implied_prob']
                                decimal_odds = row['under_decimal']
                                model_prob = 1 - model_prob
                                true_outcome_for_bet = 1 - row['total_over']
                                bet_side = 'under_total'

                    elif TARGET == 'home_cover_spread':
                        if model_prob >= 0.5:
                            if 'home_implied_spread_prob' in row and pd.notna(row['home_implied_spread_prob']):
                                implied_prob = row['home_implied_spread_prob']
                                decimal_odds = row['home_spread_decimal']
                                true_outcome_for_bet = row['home_cover_spread']
                                bet_side = 'home_spread_cover'
                        else:
                            if 'away_implied_spread_prob' in row and pd.notna(row['away_implied_spread_prob']):
                                implied_prob = row['away_implied_spread_prob']
                                decimal_odds = row['away_spread_decimal']
                                model_prob = 1 - model_prob
                                true_outcome_for_bet = 1 - row['home_cover_spread']
                                bet_side = 'away_spread_cover'

                    if bet_side is not None and pd.notna(implied_prob) and pd.notna(decimal_odds) and pd.notna(true_outcome_for_bet):
                        edge = model_prob - implied_prob

                        if edge > VALUE_THRESHOLD:
                            total_bets_placed += 1
                            total_wagered += BET_UNIT
                            profit_loss = -BET_UNIT
                            outcome_str = "Loss"
                            if true_outcome_for_bet == 1:
                                total_wins += 1
                                profit_loss = BET_UNIT * (decimal_odds - 1)
                                outcome_str = "Win"
                            total_profit += profit_loss

                            original_odds_display = np.nan
                            if bet_side == 'home_moneyline':
                                original_odds_display = row.get('home_team_moneyline', np.nan)
                            elif bet_side == 'away_moneyline':
                                original_odds_display = row.get('away_team_moneyline', np.nan)
                            elif bet_side == 'over_total':
                                original_odds_display = row.get('total_over_odds', np.nan)
                            elif bet_side == 'under_total':
                                original_odds_display = row.get('total_under_odds', np.nan)
                            elif bet_side == 'home_spread_cover':
                                original_odds_display = row.get('home_team_spread_odds', np.nan)
                            elif bet_side == 'away_spread_cover':
                                original_odds_display = row.get('away_team_spread_odds', np.nan)

                            bet_results_list.append({
                                'game_id': row['game_id'],
                                'season': row['season'],
                                'week': row['week'],
                                'gameday': row['gameday'],
                                'home_team': row['home_team'],
                                'away_team': row['away_team'],
                                'target': TARGET,
                                'bet_side': bet_side,
                                'model_prob': model_prob,
                                'implied_prob': implied_prob,
                                'edge': edge,
                                'original_odds': original_odds_display,
                                'decimal_odds': decimal_odds,
                                'true_outcome': true_outcome_for_bet,
                                'predicted_outcome': 1,
                                'profit_loss': profit_loss,
                                'outcome_str': outcome_str
                            })

            train_df = pd.concat([train_df, test_df_week_filtered], ignore_index=True)

        roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        win_rate_on_bets = (total_wins / total_bets_placed) * 100 if total_bets_placed > 0 else 0

        overall_backtesting_results[TARGET] = {
            'total_profit': total_profit,
            'total_wagered': total_wagered,
            'total_bets_placed': total_bets_placed,
            'win_rate_on_bets': win_rate_on_bets,
            'roi': roi,
            'bet_details': pd.DataFrame(bet_results_list)
        }

        print(f"\n--- Backtesting Results for '{TARGET}' ---")
        print(f"Total Profit: ${overall_backtesting_results[TARGET]['total_profit']:.2f}")
        print(f"Total Wagered: ${overall_backtesting_results[TARGET]['total_wagered']:.2f}")
        print(f"Total Bets Placed: {overall_backtesting_results[TARGET]['total_bets_placed']}")
        print(f"Win Rate on Placed Bets: {overall_backtesting_results[TARGET]['win_rate_on_bets']:.2f}%")
        print(f"Return on Investment (ROI): {overall_backtesting_results[TARGET]['roi']:.2f}%")

        if not overall_backtesting_results[TARGET]['bet_details'].empty:
            print(f"\nSample of Value Bets Placed for '{TARGET}':")
            print(overall_backtesting_results[TARGET]['bet_details'].head())
        else:
            print(f"No value bets identified for '{TARGET}' with the current threshold.")
    
    # --- Step 4: Train and Export Final Live Model ---
    print("\n--- Training Final Model and Exporting for Live Prediction ---")
    
    # Drop rows with NaN targets for final training
    df_for_final_training = df_final_game_data.copy()
    
    # For live predictions, we're going to use the spread model since that's what was requested
    TARGET_FOR_EXPORT = 'home_cover_spread'
    
    if TARGET_FOR_EXPORT not in df_for_final_training.columns:
        print(f"Error: Target '{TARGET_FOR_EXPORT}' not available for final model training. Cannot export model.")
        return

    df_for_final_training.dropna(subset=[TARGET_FOR_EXPORT], inplace=True)
    
    # Recalculate features to ensure they are based on the latest data
    X_final = df_for_final_training[features].copy()
    y_final = df_for_final_training[TARGET_FOR_EXPORT].copy()
    
    if X_final.empty or y_final.empty:
        print("Final training data is empty. Cannot export model.")
        return
        
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_final)
    
    neg_count = (y_final == 0).sum()
    pos_count = (y_final == 1).sum()
    scale_pos_weight_value = neg_count / pos_count if pos_count != 0 else 1

    final_xgb_model = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', enable_categorical=False, random_state=42,
        n_estimators=900, learning_rate=0.01, max_depth=6, min_child_weight=1,
        subsample=0.7, colsample_bytree=0.7, gamma=0.1, reg_alpha=0.0008, reg_lambda=1.2,
        scale_pos_weight=scale_pos_weight_value
    )
    final_lr_model = LogisticRegression(
        solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000
    )
    final_lgbm_model = LGBMClassifier(
        objective='binary', metric='binary_logloss', random_state=42,
        n_estimators=750, learning_rate=0.02, num_leaves=30, max_depth=-1,
        min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.001, reg_lambda=0.1,
    )
    
    final_ensemble_model = VotingClassifier(
        estimators=[('xgb', final_xgb_model), ('lr', final_lr_model), ('lgbm', final_lgbm_model)],
        voting='soft', weights=[0.50, 0.05, 0.45]
    )
    
    final_ensemble_model.fit(X_final_scaled, y_final)
    
    # Save the trained model and scaler to a file using joblib
    try:
        joblib.dump(final_ensemble_model, 'spread_prediction_model.joblib')
        joblib.dump(final_scaler, 'spread_prediction_scaler.joblib')
        print("Final model and scaler saved successfully to 'spread_prediction_model.joblib' and 'spread_prediction_scaler.joblib'.")
    except Exception as e:
        print(f"Error saving model and scaler: {e}")

# --- Main Script Execution ---

FIREBASE_SERVICE_KEY_JSON_STR = os.getenv('FIREBASE_SERVICE_KEY')

if not FIREBASE_SERVICE_KEY_JSON_STR:
    print("Error: The 'FIREBASE_SERVICE_KEY' environment variable is not set. Exiting.")
    exit()

FIREBASE_STORAGE_BUCKET = "sports-betting-model-fea2f.appspot.com"
BACKTESTING_DATA_PATH = "nfl/backtesting"
FILENAMES_TO_LOAD = {
    "final_game_data": "nfl_final_game_data.parquet",
    "advanced_pbp_data": "nfl_advanced_pbp_data.parquet"
}

initialize_firebase_admin(FIREBASE_SERVICE_KEY_JSON_STR)

loaded_dfs = load_data_from_firebase(FIREBASE_STORAGE_BUCKET, BACKTESTING_DATA_PATH, FILENAMES_TO_LOAD)
df_final_game_data = loaded_dfs.get("final_game_data")
df_advanced_pbp_data = loaded_dfs.get("advanced_pbp_data")

if df_final_game_data is not None and not df_final_game_data.empty:
    run_backtesting_pipeline(df_final_game_data, df_advanced_pbp_data)
else:
    print("Final game data not available or empty. Skipping backtesting.")

print("\n--- Phase 4: Model Evaluation, Backtesting, and Export complete. ---")
print("Analyze the ROI and other metrics. This is where we can start tweaking the VALUE_THRESHOLD and potentially model parameters to optimize profitability.")
