import pandas as pd
import numpy as np
import os
import io
import json

# Import Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage

# --- Step 1: Initialize Firebase Admin SDK ---
def initialize_firebase_admin():
    """
    Initializes the Firebase Admin SDK using the credentials file path
    provided by the GOOGLE_APPLICATION_CREDENTIALS environment variable.
    This is the standard and most reliable method for CI/CD environments.
    """
    try:
        if not firebase_admin._apps:
            # Check for the GOOGLE_APPLICATION_CREDENTIALS environment variable.
            # This is automatically set by the GitHub Actions workflow.
            cred_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            
            if not cred_path:
                raise FileNotFoundError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")

            # Initialize the app using the path to the service account key file.
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'sports-betting-model-fea2f.appspot.com'
            })
            print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        print("Please ensure your 'FIREBASE_SERVICE_KEY' environment variable is correctly set in your secrets.")
        exit(1) # Exit with an error code

# --- Step 2: Load Data from Firebase Storage ---
def load_data_from_firebase(bucket_name, source_path, filenames_to_load):
    """
    Loads dataframes from a specified Firebase Storage bucket and path.

    Args:
        bucket_name (str): The name of your Firebase Storage bucket.
        source_path (str): The sub-folder path within the bucket (e.g., 'nfl/raw_data').
        filenames_to_load (dict): A dictionary mapping a key to the filename.

    Returns:
        dict: A dictionary of loaded DataFrames.
    """
    print(f"--- Loading consolidated DataFrames from gs://{bucket_name}/{source_path} ---")
    dataframes = {}
    bucket = storage.bucket()

    for key, filename in filenames_to_load.items():
        blob_path = os.path.join(source_path, filename)
        blob = bucket.blob(blob_path)

        if blob.exists():
            try:
                # Download the file content into a BytesIO object
                file_bytes = blob.download_as_bytes()
                file_buffer = io.BytesIO(file_bytes)

                # Read the parquet file directly from the buffer
                dataframes[key] = pd.read_parquet(file_buffer)
                print(f"Loaded {filename} (Shape: {dataframes[key].shape})")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                dataframes[key] = pd.DataFrame()
        else:
            print(f"Error: File not found at gs://{bucket_name}/{blob_path}")
            dataframes[key] = pd.DataFrame()
    return dataframes

# --- Step 3: Save Processed Data to Firebase Storage ---
def save_data_to_firebase(df, bucket_name, output_path, filename):
    """
    Saves a DataFrame as a Parquet file to Firebase Storage.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        bucket_name (str): The name of your Firebase Storage bucket.
        output_path (str): The sub-folder path within the bucket (e.g., 'nfl/features').
        filename (str): The name of the file to save.
    """
    print(f"\n--- Saving enhanced DataFrame to gs://{bucket_name}/{output_path}/{filename} ---")
    bucket = storage.bucket()
    blob_path = os.path.join(output_path, filename)
    blob = bucket.blob(blob_path)

    try:
        # Convert DataFrame to Parquet bytes in a buffer
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        # Upload the buffer content to Firebase Storage
        blob.upload_from_file(parquet_buffer, content_type='application/octet-stream')
        print(f"DataFrame saved successfully to: gs://{bucket_name}/{blob_path}")
    except Exception as e:
        print(f"Error saving DataFrame to Firebase: {e}")

# --- Main Script Execution ---

# ⚠️ GITHUB SECRETS IMPLEMENTATION ⚠️
# The script will now retrieve the service account key from an environment variable.
FIREBASE_SERVICE_KEY_JSON_STR = os.getenv('FIREBASE_SERVICE_KEY')

if not FIREBASE_SERVICE_KEY_JSON_STR:
    print("Error: The 'FIREBASE_SERVICE_KEY' environment variable is not set. Exiting.")
    exit()

# 2. Provide your Firebase Storage bucket name.
FIREBASE_STORAGE_BUCKET = "sports-betting-model-fea2f.appspot.com"

# 3. Define the paths within your bucket for raw and processed data.
RAW_DATA_PATH = "nfl/raw_data"
PROCESSED_DATA_PATH = "nfl/features"
PROCESSED_FILENAME = "nfl_game_level_features.parquet"

# Initialize the Firebase Admin SDK with the secret key string
initialize_firebase_admin()

# Define the files the script needs to load from Firebase
files_to_load = {
    "schedules": "nfl_schedules_all_years.parquet",
    "pbp": "nfl_pbp_all_years.parquet"
}

# Load the necessary data for this script
loaded_dfs = load_data_from_firebase(FIREBASE_STORAGE_BUCKET, RAW_DATA_PATH, files_to_load)
df_schedules_all = loaded_dfs.get("schedules")
df_pbp_all = loaded_dfs.get("pbp")

# --- Game-Level Features ---
df_games_teams = pd.DataFrame() # Initialize an empty DataFrame
if df_schedules_all is not None and not df_schedules_all.empty:
    print("\n--- Starting Game-Level Feature Engineering ---")

    # Ensure gameday is datetime and sort for time-series operations
    df_schedules_all['gameday'] = pd.to_datetime(df_schedules_all['gameday'])
    df_schedules_all = df_schedules_all.sort_values(by=['season', 'week', 'gameday']).reset_index(drop=True)

    # 2.1. Basic Game Outcomes / Target Variables
    df_schedules_all['score_differential'] = df_schedules_all['home_score'] - df_schedules_all['away_score']
    df_schedules_all['abs_score_differential'] = abs(df_schedules_all['score_differential'])
    df_schedules_all['home_win'] = (df_schedules_all['score_differential'] > 0).astype(int)
    # The total_over feature calculation is simplified and more robust now
    df_schedules_all['total_over'] = np.where(df_schedules_all['total'] > df_schedules_all['total_line'], 1,
                                             np.where(df_schedules_all['total'] < df_schedules_all['total_line'], 0, 0.5))

    print("Added basic game outcome features (score_differential, abs_score_differential, home_win, total_over).")

    # 2.2. Team-Specific Game Features (for both home and away teams)
    # This step transforms the data so each row represents a team's perspective.
    common_game_cols = ['game_id', 'season', 'week', 'gameday', 'score_differential', 'home_win', 'total', 'total_line', 'total_over']

    odds_cols_home_raw = ['home_moneyline', 'spread_line', 'home_spread_odds']
    odds_cols_away_raw = ['away_moneyline', 'spread_line', 'away_spread_odds']

    existing_odds_cols_home = [col for col in odds_cols_home_raw if col in df_schedules_all.columns]
    existing_odds_cols_away = [col for col in odds_cols_away_raw if col in df_schedules_all.columns]

    df_home = df_schedules_all[common_game_cols + ['home_team', 'home_score', 'away_team', 'away_score', 'home_rest'] + existing_odds_cols_home].copy()
    df_home.rename(columns={
        'home_team': 'team', 'home_score': 'team_score', 'away_team': 'opponent', 'away_score': 'opponent_score',
        'score_differential': 'team_score_differential', 'home_win': 'team_win', 'home_rest': 'team_rest',
        'home_moneyline': 'team_moneyline', 'spread_line': 'team_spread_line', 'home_spread_odds': 'team_spread_odds'
    }, inplace=True)
    df_home['is_home'] = 1

    df_away = df_schedules_all[common_game_cols + ['away_team', 'away_score', 'home_team', 'home_score', 'away_rest'] + existing_odds_cols_away].copy()
    df_away.rename(columns={
        'away_team': 'team', 'away_score': 'team_score', 'home_team': 'opponent', 'home_score': 'opponent_score',
        'score_differential': 'opponent_score_differential', 'home_win': 'opponent_win', 'away_rest': 'team_rest',
        'away_moneyline': 'team_moneyline', 'spread_line': 'opponent_spread_line', 'away_spread_odds': 'team_spread_odds'
    }, inplace=True)

    df_away['team_score_differential'] = -df_away['opponent_score_differential']
    df_away['team_win'] = 1 - df_away['opponent_win']
    df_away['is_home'] = 0
    if 'opponent_spread_line' in df_away.columns:
        df_away['team_spread_line'] = -df_away['opponent_spread_line']

    cols_to_drop_away = ['opponent_score_differential', 'opponent_win']
    df_away.drop(columns=[col for col in cols_to_drop_away if col in df_away.columns], inplace=True)

    df_games_teams = pd.concat([df_home, df_away], ignore_index=True)
    df_games_teams = df_games_teams.sort_values(by=['game_id', 'team']).reset_index(drop=True)

    print(f"Created team-game level DataFrame (df_games_teams). Shape: {df_games_teams.shape}")

    # 2.3. Rolling Averages for Team Performance
    df_games_teams = df_games_teams.sort_values(by=['team', 'gameday']).reset_index(drop=True)
    rolling_metrics = ['team_score', 'opponent_score', 'team_score_differential']
    windows = [3, 5, 8]

    for metric in rolling_metrics:
        for window in windows:
            df_games_teams[f'rolling_{metric}_{window}_games'] = df_games_teams.groupby('team')[metric].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            ).fillna(0)
    print("Calculated rolling averages for team scores and differentials.")

    # 2.4. Advanced Game-Level Features from Play-by-Play Data
    if df_pbp_all is not None and not df_pbp_all.empty:
        print("\n--- Adding Advanced Game-Level Features from Play-by-Play Data ---")

        # Memory Optimization: Explicitly convert to smaller data types
        print("Optimizing PBP data types to save memory...")
        # Fill NaNs in 'yards_gained' before converting to integer type
        df_pbp_all['yards_gained'] = df_pbp_all['yards_gained'].fillna(0)
        df_pbp_all['game_id'] = df_pbp_all['game_id'].astype('category')
        df_pbp_all['posteam'] = df_pbp_all['posteam'].astype('category')
        df_pbp_all['defteam'] = df_pbp_all['defteam'].astype('category')
        df_pbp_all['epa'] = df_pbp_all['epa'].astype('float32')
        # Now convert to integer after handling NaNs
        df_pbp_all['yards_gained'] = df_pbp_all['yards_gained'].astype('int16')

        # --- Create necessary boolean columns for aggregation ---
        df_pbp_all['is_redzone_attempt'] = (df_pbp_all['yardline_100'] <= 20) & (df_pbp_all['play_type'].isin(['pass', 'run']))
        df_pbp_all['is_redzone_td'] = df_pbp_all['is_redzone_attempt'] & (df_pbp_all['touchdown'] == 1)
        df_pbp_all['is_third_down_attempt'] = (df_pbp_all['down'] == 3)
        df_pbp_all['is_third_down_conversion'] = df_pbp_all['is_third_down_attempt'] & (df_pbp_all['first_down'] == 1)
        df_pbp_all['is_explosive_play'] = ((df_pbp_all['play_type'] == 'run') & (df_pbp_all['yards_gained'] >= 10)) | \
                                         ((df_pbp_all['play_type'] == 'pass') & (df_pbp_all['yards_gained'] >= 20))
        df_pbp_all['turnover_committed'] = df_pbp_all['interception'] | df_pbp_all['fumble_lost']

        # Logic for offensive success rate as per your request
        df_pbp_all['is_success'] = 0
        df_pbp_all.loc[(df_pbp_all['down'] == 1) & (df_pbp_all['yards_gained'] >= 0.4 * df_pbp_all['ydstogo']), 'is_success'] = 1
        df_pbp_all.loc[(df_pbp_all['down'] == 2) & (df_pbp_all['yards_gained'] >= 0.6 * df_pbp_all['ydstogo']), 'is_success'] = 1
        df_pbp_all.loc[(df_pbp_all['down'].isin([3, 4])) & (df_pbp_all['first_down'] == 1), 'is_success'] = 1

        # --- A. Common Aggregations grouped by offense (posteam) ---
        pbp_offensive_stats = df_pbp_all.groupby(['game_id', 'posteam']).agg(
            # Offensive stats
            redzone_attempts=('is_redzone_attempt', 'sum'),
            redzone_tds=('is_redzone_td', 'sum'),
            third_down_attempts=('is_third_down_attempt', 'sum'),
            third_down_conversions=('is_third_down_conversion', 'sum'),
            turnovers_committed=('turnover_committed', 'sum'),
            offensive_epa_per_play=('epa', 'mean'),
            total_offensive_yards=('yards_gained', 'sum'),
            total_offensive_plays=('play_id', 'count'),
            offensive_explosive_plays=('is_explosive_play', 'sum'),
            offensive_successes=('is_success', 'sum'),
            penalty_yards_committed=('penalty_yards', 'sum')
        ).reset_index()
        pbp_offensive_stats.rename(columns={'posteam': 'team'}, inplace=True)

        # --- B. Aggregations grouped by defense (defteam) ---
        pbp_defensive_stats = df_pbp_all.groupby(['game_id', 'defteam']).agg(
            # Defensive stats
            turnovers_forced=('turnover_committed', 'sum'),
            defensive_epa_per_play_allowed=('epa', 'mean'),
            total_defensive_yards_allowed=('yards_gained', 'sum'),
            total_defensive_plays_faced=('play_id', 'count'),
            explosive_plays_allowed=('is_explosive_play', 'sum'),
            penalty_yards_forced=('penalty_yards', 'sum')
        ).reset_index()
        pbp_defensive_stats.rename(columns={'defteam': 'team'}, inplace=True)

        # --- C. Merge offensive and defensive stats into a single DataFrame ---
        pbp_game_stats = pd.merge(pbp_offensive_stats, pbp_defensive_stats, on=['game_id', 'team'], how='left')

        # --- D. Drive-level features (only possible if 'drive_id' exists) ---
        if 'drive_id' in df_pbp_all.columns:
            # Drive Success Rate
            df_pbp_all['is_scoring_play'] = df_pbp_all['touchdown'] | (df_pbp_all['field_goal_result'] == 'made')
            scoring_drives = df_pbp_all[df_pbp_all['is_scoring_play'] == 1][['game_id', 'drive_id', 'posteam']].drop_duplicates()
            scoring_drives['is_drive_successful'] = 1
            all_drives = df_pbp_all[['game_id', 'drive_id', 'posteam']].drop_duplicates()
            all_drives.rename(columns={'posteam': 'team'}, inplace=True)
            all_drives = pd.merge(all_drives, scoring_drives, on=['game_id', 'drive_id', 'team'], how='left')
            all_drives['is_drive_successful'] = all_drives['is_drive_successful'].fillna(0)

            pbp_drive_success_rate = all_drives.groupby(['game_id', 'team']).agg(
                offensive_drive_success_rate=('is_drive_successful', 'mean')
            ).reset_index()
            pbp_game_stats = pd.merge(pbp_game_stats, pbp_drive_success_rate, on=['game_id', 'team'], how='left')

            # Average Starting Field Position
            drive_starts = df_pbp_all.groupby(['game_id', 'drive_id', 'posteam']).first().reset_index()
            pbp_avg_start_field_pos = drive_starts.groupby(['game_id', 'posteam']).agg(
                avg_starting_field_position=('yardline_100', 'mean')
            ).reset_index()
            pbp_avg_start_field_pos.rename(columns={'posteam': 'team'}, inplace=True)
            pbp_game_stats = pd.merge(pbp_game_stats, pbp_avg_start_field_pos, on=['game_id', 'team'], how='left')
        else:
            print("Skipping Drive Success Rate and Average Starting Field Position: 'drive_id' column not found in play-by-play data.")

        # --- E. Calculate rates and differentials ---
        pbp_game_stats['redzone_td_pct'] = pbp_game_stats['redzone_tds'] / pbp_game_stats['redzone_attempts']
        pbp_game_stats['third_down_conv_pct'] = pbp_game_stats['third_down_conversions'] / pbp_game_stats['third_down_attempts']
        pbp_game_stats['turnover_differential'] = pbp_game_stats['turnovers_forced'] - pbp_game_stats['turnovers_committed']
        pbp_game_stats['net_yards_per_play'] = (pbp_game_stats['total_offensive_yards'] - pbp_game_stats['total_defensive_yards_allowed']) / \
                                             (pbp_game_stats['total_offensive_plays'] + pbp_game_stats['total_defensive_plays_faced'])
        pbp_game_stats['offensive_explosive_play_rate'] = pbp_game_stats['offensive_explosive_plays'] / pbp_game_stats['total_offensive_plays']
        pbp_game_stats['defensive_explosive_play_rate_allowed'] = pbp_game_stats['explosive_plays_allowed'] / pbp_game_stats['total_defensive_plays_faced']
        pbp_game_stats['offensive_success_rate'] = pbp_game_stats['offensive_successes'] / pbp_game_stats['total_offensive_plays']
        pbp_game_stats['penalty_differential'] = pbp_game_stats['penalty_yards_forced'] - pbp_game_stats['penalty_yards_committed']

        # Identify numeric columns created from the rates and fill NaNs specifically on them.
        rate_cols = [
            'redzone_td_pct', 'third_down_conv_pct', 'net_yards_per_play',
            'offensive_explosive_play_rate', 'defensive_explosive_play_rate_allowed',
            'offensive_success_rate'
        ]

        # Drive-level features are optional, so we'll check if they exist before adding to the list.
        if 'offensive_drive_success_rate' in pbp_game_stats.columns:
            rate_cols.append('offensive_drive_success_rate')
        if 'avg_starting_field_position' in pbp_game_stats.columns:
            rate_cols.append('avg_starting_field_position')

        # Apply fillna only to the identified rate columns
        pbp_game_stats[rate_cols] = pbp_game_stats[rate_cols].fillna(0)

        # --- F. Merge all PBP stats back into df_games_teams and calculate rolling averages ---
        pbp_cols_to_merge = [
            'redzone_td_pct', 'third_down_conv_pct', 'turnover_differential',
            'offensive_epa_per_play', 'defensive_epa_per_play_allowed',
            'net_yards_per_play', 'penalty_differential', 'offensive_explosive_play_rate',
            'defensive_explosive_play_rate_allowed', 'offensive_success_rate',
            'offensive_drive_success_rate', 'avg_starting_field_position'
        ]

        # Filter out columns that might not exist (e.g., drive-level stats)
        existing_pbp_cols = [col for col in pbp_cols_to_merge if col in pbp_game_stats.columns]
        df_games_teams = pd.merge(df_games_teams, pbp_game_stats[['game_id', 'team'] + existing_pbp_cols], on=['game_id', 'team'], how='left')
        df_games_teams.fillna(0, inplace=True) # Fill NaNs from the merge

        # Calculate rolling averages for all the new features
        df_games_teams = df_games_teams.sort_values(by=['team', 'gameday']).reset_index(drop=True)
        for metric in existing_pbp_cols:
            for window in windows:
                df_games_teams[f'rolling_{metric}_{window}_games'] = df_games_teams.groupby('team')[metric].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                ).fillna(0)

        print("Added rolling advanced PBP features.")
    else:
        print("Play-by-play data not available or empty. Skipping advanced PBP features.")

    print("\nInitial game-level feature engineering complete.")

    # Save the enhanced game-level DataFrame
    save_data_to_firebase(df_games_teams, FIREBASE_STORAGE_BUCKET, PROCESSED_DATA_PATH, PROCESSED_FILENAME)

else:
    print("Schedules data not available or empty. Skipping game-level feature engineering.")
