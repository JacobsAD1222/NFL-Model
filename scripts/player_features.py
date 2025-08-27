import pandas as pd
import numpy as np
import os
import io
import json

# Import Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage

# --- Step 1: Initialize Firebase Admin SDK using a standard approach ---
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
                'storageBucket': 'sports-betting-model-fea2f.firebasestorage.app'
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
PROCESSED_FILENAME = "nfl_player_level_features.parquet"

# Initialize the Firebase Admin SDK with the secret key string
initialize_firebase_admin()

# Define the files the script needs to load from Firebase
files_to_load = {
    "weekly_stats": "nfl_weekly_stats_all_years.parquet",
    "schedules": "nfl_schedules_all_years.parquet",
    "pbp": "nfl_pbp_all_years.parquet",
    "game_features": "nfl_game_level_features.parquet"
}

# Load all necessary dataframes from Firebase Storage
loaded_dfs = load_data_from_firebase(FIREBASE_STORAGE_BUCKET, RAW_DATA_PATH, files_to_load)

df_weekly_stats_all = loaded_dfs.get("weekly_stats")
df_schedules_all = loaded_dfs.get("schedules")
df_pbp_all = loaded_dfs.get("pbp")
df_game_level_features = loaded_dfs.get("game_features")

# --- Step 4: Player-Level Feature Engineering (using df_weekly_stats_all) ---
if not df_weekly_stats_all.empty:
    print("\n--- Starting Player-Level Feature Engineering ---")

    # Ensure data is sorted by player and week for time-series operations
    df_weekly_stats_all = df_weekly_stats_all.sort_values(by=['player_id', 'season', 'week']).reset_index(drop=True)

    # 4.1 --- Fix missing home/away and game_id reference ---
    if not df_schedules_all.empty:
        # Create a temporary dataframe with a row for each team in each game.
        df_schedules_teams = pd.concat([
            df_schedules_all[['season', 'week', 'game_id', 'home_team']].rename(columns={'home_team': 'team'}),
            df_schedules_all[['season', 'week', 'game_id', 'away_team']].rename(columns={'away_team': 'team'})
        ], ignore_index=True)
        
        # Now, merge this with weekly stats to correctly get game_id
        df_weekly_stats_all = pd.merge(
            df_weekly_stats_all,
            df_schedules_teams,
            left_on=['season', 'week', 'recent_team'],
            right_on=['season', 'week', 'team'],
            how='left'
        )
        
        # FIX: The previous line had an indexing error. This new logic
        # correctly compares the `recent_team` to the `team` column
        # created by the merge, ensuring the Series are aligned.
        df_weekly_stats_all['is_home'] = (df_weekly_stats_all['recent_team'] == df_weekly_stats_all['team']).astype(int)

        df_weekly_stats_all.rename(columns={'game_id': 'game_id_ref'}, inplace=True)
        df_weekly_stats_all.drop(columns=['team'], inplace=True, errors='ignore')

    else:
        df_weekly_stats_all['is_home'] = np.nan
        df_weekly_stats_all['game_id_ref'] = np.nan
        print("Schedules data not available. 'is_home' and 'game_id_ref' are not available.")

    # 4.2 Calculate FanDuel Fantasy Points
    print("\n--- Calculating FanDuel Fantasy Points ---")

    # Define FanDuel scoring rules
    stats_for_fanduel = [
        'passing_yards', 'passing_tds', 'interceptions',
        'rushing_yards', 'rushing_tds',
        'receiving_yards', 'receiving_tds', 'receptions',
        'fumbles_lost' # Assuming this column exists for fumble lost points
    ]

    for col in stats_for_fanduel:
        if col not in df_weekly_stats_all.columns:
            df_weekly_stats_all[col] = 0 # Add missing columns with 0
            print(f"Warning: '{col}' not found in weekly stats, added with 0s for FanDuel calculation.")
        else:
            df_weekly_stats_all[col] = df_weekly_stats_all[col].fillna(0) # Fill NaNs

    df_weekly_stats_all['fantasy_points_fanduel'] = (
        (df_weekly_stats_all['passing_yards'] * 0.04) + # 1 point per 25 passing yards (0.04 per yard)
        (df_weekly_stats_all['passing_tds'] * 4) +
        (df_weekly_stats_all['interceptions'] * -1) +
        (df_weekly_stats_all['rushing_yards'] * 0.1) +
        (df_weekly_stats_all['rushing_tds'] * 6) +
        (df_weekly_stats_all['receiving_yards'] * 0.1) +
        (df_weekly_stats_all['receiving_tds'] * 6) +
        (df_weekly_stats_all['receptions'] * 0.5) + # Half-PPR
        (df_weekly_stats_all['fumbles_lost'] * -1)
    )
    print("Calculated 'fantasy_points_fanduel' for all players.")
    
    # 4.3 Rolling averages for core player stats
    player_rolling_metrics = [
        'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'carries',
        'rushing_yards', 'rushing_tds', 'receptions', 'targets',
        'receiving_yards', 'receiving_tds', 'fantasy_points_ppr', 'fantasy_points_fanduel'
    ]
    player_windows = [3, 5, 8]

    for metric in player_rolling_metrics:
        if metric in df_weekly_stats_all.columns:
            for window in player_windows:
                df_weekly_stats_all[f'rolling_{metric}_{window}_weeks'] = df_weekly_stats_all.groupby('player_id')[metric].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                ).fillna(0)
        else:
            print(f"Skipping rolling average for '{metric}': Column not found in df_weekly_stats_all.")

    print("Calculated rolling averages for core player statistics.")

    # 4.4 Advanced Player-Level Features ---
    print("\n--- Adding Advanced Player-Level Features ---")

    # 1️⃣ Opponent Defensive Matchup Ranks (avg fantasy points allowed by opponent to player's position)
    if 'opponent_team' in df_weekly_stats_all.columns:
        df_opp_defense = df_weekly_stats_all.groupby(['season', 'week', 'opponent_team', 'position']).agg(
            opp_avg_fantasy_allowed=('fantasy_points_ppr', 'mean')
        ).reset_index()

        df_weekly_stats_all = df_weekly_stats_all.merge(
            df_opp_defense,
            on=['season', 'week', 'opponent_team', 'position'],
            how='left'
        )
    else:
        print("Warning: 'opponent_team' not found in weekly stats. Skipping opponent defense features.")

    # 2️⃣ Player Historical Performance vs Specific Opponent Teams (rolling averages)
    rolling_metrics_vs_opponent = ['fantasy_points_ppr', 'passing_yards', 'rushing_yards', 'receiving_yards']
    if 'opponent_team' in df_weekly_stats_all.columns:
        for metric in rolling_metrics_vs_opponent:
            if metric in df_weekly_stats_all.columns:
                for window in player_windows:
                    df_weekly_stats_all[f'rolling_{metric}_vs_opponent_{window}_weeks'] = df_weekly_stats_all.groupby(
                        ['player_id', 'opponent_team']
                    )[metric].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()).fillna(0)

    # 3️⃣ Rolling averages by home/away
    rolling_metrics_home_away = ['fantasy_points_ppr', 'passing_yards', 'rushing_yards', 'receiving_yards']
    if 'is_home' in df_weekly_stats_all.columns:
        for metric in rolling_metrics_home_away:
            if metric in df_weekly_stats_all.columns:
                for window in player_windows:
                    df_weekly_stats_all[f'rolling_{metric}_home_{window}_weeks'] = (
                        df_weekly_stats_all.groupby(['player_id', 'is_home'])[metric]
                        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
                        .fillna(0)
                    )

    # 4️⃣ Snap Share / Target Share Rolling
    rolling_usage_cols = ['target_share', 'snap_percentage']
    for col in rolling_usage_cols:
        if col in df_weekly_stats_all.columns:
            for window in player_windows:
                df_weekly_stats_all[f'rolling_{col}_{window}_weeks'] = df_weekly_stats_all.groupby(
                    'player_id'
                )[col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()).fillna(0)

    # 5️⃣ Team Pace & Run/Pass Ratio from play-by-play
    if not df_pbp_all.empty:
        if 'posteam' in df_pbp_all.columns and 'play_type' in df_pbp_all.columns:
            team_plays = df_pbp_all.groupby(['season','week','posteam']).agg(
                plays=('play_id','count'),
                run_plays=('play_type', lambda x: (x=='run').sum()),
                pass_plays=('play_type', lambda x: (x=='pass').sum())
            ).reset_index().rename(columns={'posteam':'team'})
            team_plays['run_pass_ratio'] = team_plays['run_plays'] / team_plays['pass_plays'].replace(0,1)
            team_plays['pace'] = team_plays['plays']

            df_weekly_stats_all = df_weekly_stats_all.merge(
                team_plays[['season','week','team','pace','run_pass_ratio']],
                left_on=['season','week','recent_team'],
                right_on=['season','week','team'],
                how='left'
            ).drop(columns=['team'], errors='ignore')
        else:
            print("PBP data missing 'posteam' or 'play_type'. Skipping pace and ratio features.")

    # 6️⃣ Defensive Coverage Tendencies of Opponent
    if not df_pbp_all.empty and 'defense_coverage_type' in df_pbp_all.columns and 'defense_man_zone_type' in df_pbp_all.columns:
        coverage_summary = df_pbp_all.groupby(['season','week','defteam','defense_coverage_type', 'defense_man_zone_type']).agg(
            plays=('play_id','count')
        ).reset_index()
        coverage_pivot = coverage_summary.pivot_table(
            index=['season','week','defteam'],
            columns=['defense_coverage_type', 'defense_man_zone_type'],
            values='plays',
            fill_value=0
        ).reset_index()
        coverage_pivot.columns = [
            '_'.join([str(c) for c in col if c not in ['', None]])
            if isinstance(col, tuple) else col
            for col in coverage_pivot.columns
        ]
        coverage_cols = [c for c in coverage_pivot.columns if c not in ['season','week','defteam']]
        coverage_pivot[coverage_cols] = coverage_pivot[coverage_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        coverage_pivot[coverage_cols] = coverage_pivot[coverage_cols].div(coverage_pivot[coverage_cols].sum(axis=1), axis=0)
        
        # Merge this into df_weekly_stats_all
        if 'opponent_team' in df_weekly_stats_all.columns:
            df_weekly_stats_all = df_weekly_stats_all.merge(
                coverage_pivot,
                left_on=['season','week','opponent_team'],
                right_on=['season','week','defteam'],
                how='left'
            ).drop(columns=['defteam'], errors='ignore')

    # 7️⃣ Player Effectiveness vs Coverage (Receiver, Passer, Rusher)
    if not df_pbp_all.empty:
        def add_role_prefix(df, role):
            metric_cols = [c for c in df.columns
                             if c not in ['player_id','season','week','defense_coverage_type','defense_man_zone_type']]
            rename_map = {c: f"{role}_{c}" for c in metric_cols}
            return df.rename(columns=rename_map)

        receiver_vs_coverage = df_pbp_all.groupby(
            ['receiver_player_id','season','week','defense_coverage_type','defense_man_zone_type']
        ).agg(
            yds_per_target=('yards_gained','mean'),
            catch_rate=('complete_pass','mean'),
            targets=('play_id','count')
        ).reset_index().rename(columns={'receiver_player_id':'player_id'})
        receiver_vs_coverage = add_role_prefix(receiver_vs_coverage, "receiver")
        receiver_vs_coverage['role'] = 'receiver'

        passer_vs_coverage = df_pbp_all.groupby(
            ['passer_player_id','season','week','defense_coverage_type','defense_man_zone_type']
        ).agg(
            yds_per_att=('yards_gained','mean'),
            comp_rate=('complete_pass','mean'),
            attempts=('play_id','count'),
            pass_tds=('pass_touchdown','sum'),
            interceptions=('interception','sum')
        ).reset_index().rename(columns={'passer_player_id':'player_id'})
        passer_vs_coverage = add_role_prefix(passer_vs_coverage, "passer")
        passer_vs_coverage['role'] = 'passer'

        rusher_vs_coverage = df_pbp_all.groupby(
            ['rusher_player_id','season','week','defense_coverage_type','defense_man_zone_type']
        ).agg(
            yds_per_carry=('yards_gained','mean'),
            carries=('play_id','count'),
            rush_tds=('rush_touchdown','sum')
        ).reset_index().rename(columns={'rusher_player_id':'player_id'})
        rusher_vs_coverage = add_role_prefix(rusher_vs_coverage, "rusher")
        rusher_vs_coverage['role'] = 'rusher'
        
        player_vs_coverage = pd.concat(
            [receiver_vs_coverage, passer_vs_coverage, rusher_vs_coverage],
            ignore_index=True
        )
        
        # Melt and Pivot for rolling averages (as in your original logic)
        if 'defense_coverage_type' in player_vs_coverage.columns and 'defense_man_zone_type' in player_vs_coverage.columns:
            coverage_long = player_vs_coverage.melt(
                id_vars=['player_id','season','week','role','defense_coverage_type','defense_man_zone_type'],
                value_vars=[c for c in player_vs_coverage.columns
                             if c not in ['player_id','season','week','role','defense_coverage_type','defense_man_zone_type']],
                var_name='metric',
                value_name='value'
            ).sort_values(['player_id','season','week'])

            coverage_long['last3'] = (
                coverage_long.groupby(['player_id','season','role','metric','defense_coverage_type','defense_man_zone_type'])['value']
                .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            )

            coverage_long['season_to_date'] = (
                coverage_long.groupby(['player_id','season','role','metric','defense_coverage_type','defense_man_zone_type'])['value']
                .transform(lambda x: x.shift(1).expanding().mean())
            )
            
            # Pivot the 'last3' column to create wide features
            coverage_roll_last3 = coverage_long.pivot_table(
                index=['player_id','season','week','role'],
                columns=['metric','defense_coverage_type','defense_man_zone_type'],
                values='last3',
                fill_value=0
            ).reset_index()

            # Pivot the 'season_to_date' column to create wide features
            coverage_roll_s2d = coverage_long.pivot_table(
                index=['player_id','season','week','role'],
                columns=['metric','defense_coverage_type','defense_man_zone_type'],
                values='season_to_date',
                fill_value=0
            ).reset_index()
            
            # --- FIX FOR KEYERROR ---
            # Rename columns directly after reset_index to ensure no keys are lost.
            new_cols_last3 = ['player_id', 'season', 'week', 'role'] + [
                f'last3_{"_".join(col).lower()}' for col in coverage_roll_last3.columns.drop(['player_id','season','week','role'])
            ]
            coverage_roll_last3.columns = new_cols_last3
            
            new_cols_s2d = ['player_id', 'season', 'week', 'role'] + [
                f's2d_{"_".join(col).lower()}' for col in coverage_roll_s2d.columns.drop(['player_id','season','week','role'])
            ]
            coverage_roll_s2d.columns = new_cols_s2d

            # Merge the two wide dataframes together
            coverage_roll_wide = pd.merge(
                coverage_roll_last3,
                coverage_roll_s2d,
                on=['player_id','season','week','role'],
                how='outer'
            )
            
            df_weekly_stats_all = df_weekly_stats_all.merge(
                coverage_roll_wide.drop(columns=['role'], errors='ignore'),
                on=['player_id','season','week'],
                how='left'
            )
        else:
            print("PBP data missing coverage info. Skipping player vs. coverage features.")

    print("Advanced player-level features (opponent defense, pace, run/pass, coverage) added successfully.")

    # --- Step 5: Merge Player Features with Game Context and Opponent Features ---
    print("\n--- Merging Player Features with Game Context and Opponent Features ---")

    if not df_game_level_features.empty:
        df_player_props = pd.merge(
            df_weekly_stats_all,
            df_game_level_features,
            left_on=['game_id_ref', 'season', 'week', 'recent_team'],
            right_on=['game_id', 'season', 'week', 'team'],
            how='left'
        )
        df_player_props.drop(columns=['game_id', 'team'], inplace=True, errors='ignore')
        df_player_props.rename(columns={'game_id_ref': 'game_id'}, inplace=True)
    else:
        print("Game-level features not available. Skipping merge.")
        df_player_props = df_weekly_stats_all
        df_player_props.rename(columns={'game_id_ref': 'game_id'}, inplace=True)

    # Final cleanup and sorting
    df_player_props['gameday'] = pd.to_datetime(df_player_props['gameday'])
    df_player_props = df_player_props.sort_values(by=['player_id', 'gameday']).reset_index(drop=True)

    # --- Step 6: Save enhanced player-level DataFrame to Firebase Storage ---
    save_data_to_firebase(df_player_props, FIREBASE_STORAGE_BUCKET, PROCESSED_DATA_PATH, PROCESSED_FILENAME)

else:
    print("Weekly stats data not available or empty. Skipping player-level feature engineering.")
