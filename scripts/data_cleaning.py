import pandas as pd
import os
import numpy as np # Import numpy for NaN handling
from google.cloud import storage
from io import BytesIO # Import BytesIO for handling in-memory files

# --- Firebase Configuration ---
FIREBASE_KEY_PATH = "firebase_key.json"
BUCKET_NAME = "sports-betting-model-fea2f.firebasestorage.app"
FOLDER_PATH = "nfl/raw_data"  # Subfolder in your bucket

# Initialize Firebase Storage client
client = storage.Client.from_service_account_json(FIREBASE_KEY_PATH)
bucket = client.bucket(BUCKET_NAME)

# --- Step 2: Define Cleaning Functions for each DataFrame type ---

def clean_schedules_data(df):
    """Applies cleaning steps to the schedules DataFrame."""
    print("--- Cleaning Schedules Data ---")
    # 1. Drop Empty Columns: nfl_detail_id, pff, ftn have 0 non-null values
    cols_to_drop_schedules = ['nfl_detail_id', 'pff', 'ftn']
    df = df.drop(columns=[col for col in cols_to_drop_schedules if col in df.columns], errors='ignore')
    # print(f"Dropped empty columns. New shape: {df.shape}") # Removed for cleaner loop output

    # 2. Convert gameday to Datetime
    df['gameday'] = pd.to_datetime(df['gameday'])
    # print("Converted 'gameday' to datetime.")

    # 3. Handle overtime in Schedules: Convert to integer (0 or 1)
    df['overtime'] = df['overtime'].fillna(0).astype(int)
    # print("Converted 'overtime' to integer.")

    # 4. Impute/Handle surface, temp, wind in Schedules
    df['surface'] = df['surface'].fillna('Unknown')

    indoor_games_mask = df['roof'].isin(['Dome', 'Retr. Roof-Closed'])
    df.loc[indoor_games_mask, ['temp', 'wind']] = df.loc[indoor_games_mask, ['temp', 'wind']].fillna(0)

    # Calculate median for outdoor games only
    median_temp_outdoor = df.loc[~indoor_games_mask, 'temp'].median()
    median_wind_outdoor = df.loc[~indoor_games_mask, 'wind'].median()

    df['temp'] = df['temp'].fillna(median_temp_outdoor)
    df['wind'] = df['wind'].fillna(median_wind_outdoor)
    # print("Handled NaN in 'temp' and 'wind' (0 for indoor, median for outdoor).")
    return df

def clean_weekly_stats_data(df):
    """Applies cleaning steps to the weekly_stats DataFrame."""
    print("--- Cleaning Weekly Stats Data ---")
    # 5. Handle Missing EPA/Advanced Metrics in Weekly Stats: Impute NaN with 0
    epa_cols = [
        'passing_epa', 'pacr', 'dakota', 'rushing_epa', 'receiving_epa',
        'racr', 'target_share', 'air_yards_share', 'wopr'
    ]
    for col in epa_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    # print("Filled NaN in EPA/advanced metrics in 'weekly_stats' with 0.")
    return df

def clean_pbp_data(df):
    """Applies cleaning steps to the play-by-play DataFrame."""
    print("--- Cleaning Play-by-Play Data ---")
    # 1. Drop Empty Columns: Identify columns with all NaN values
    cols_to_drop_pbp = df.columns[df.isnull().all()].tolist()
    # Add specific PFF columns if they exist and are empty (based on common observation)
    pff_specific_cols = [col for col in df.columns if 'pff_' in col and df[col].isnull().all()]
    cols_to_drop_pbp.extend(pff_specific_cols)
    cols_to_drop_pbp = list(set(cols_to_drop_pbp)) # Remove duplicates

    df = df.drop(columns=[col for col in cols_to_drop_pbp if col in df.columns], errors='ignore')
    # print(f"Dropped empty columns from PBP. New shape: {df.shape}")

    # REVISED list of columns that are truly numeric in PBP
    # Removed 'desc', 'play_type', and all player_id columns from this list
    numeric_pbp_cols = [
        'qtr', 'down', 'yrdln', 'ydstogo', 'yards_gained',
        'shotgun', 'no_huddle', 'qb_hit', 'sack', 'touchdown', 'pass_attempt',
        'rush_attempt', 'interception', 'fumble_lost', 'complete_pass',
        'epa', 'wpa', 'air_yards', 'yards_after_catch', 'cpoe', 'cp',
        'qb_scramble', 'penalty_yards', 'first_down_pass', 'first_down_rush',
        'first_down_penalty', 'third_down_converted', 'third_down_failed',
        'fourth_down_converted', 'fourth_down_failed', 'fumble_recovery_2pt',
        'safety', 'field_goal_attempt', 'field_goal_result', 'extra_point_attempt',
        'extra_point_result', 'two_point_attempt', 'two_point_result',
        'kick_distance', 'punt_distance', 'kickoff_distance', 'return_yards',
        'lateral_yards', 'lateral_return_yards', 'lateral_rush_yards',
        'lateral_rec_yards', 'lateral_punt_return_yards', 'lateral_kickoff_return_yards',
        'lateral_interception_yards', 'lateral_fumble_recovery_yards',
        'qb_hit_att', 'pass_block_att', 'run_block_att', 'special_teams_play',
        'special_teams_fg_attempt', 'special_teams_punt_attempt',
        'special_teams_kickoff_attempt', 'special_teams_misc_attempt',
        'fg_make_prob', 'td_prob', 'ep', 'vegas_wp', 'vegas_wpa', 'home_wp',
        'away_wp', 'home_wp_post', 'away_wp_post', 'qb_epa', 'xpass',
        'pass_rush_att', 'dropback', 'rpo_snap', 'pass_net_epa', 'rush_net_epa',
        'comp_air_epa', 'comp_yac_epa', 'comp_air_wpa', 'comp_yac_wpa',
        'air_epa', 'yac_epa', 'air_wpa', 'yac_wpa', 'punt_blocked',
        'extra_point_blocked', 'field_goal_blocked', 'fumble_forced',
        'fumble_recovery_fumble', 'fumble_recovery_opponent', 'solo_tackle',
        'tackles_for_loss', 'assist_tackle', 'qb_hit_defense', 'interception_defense',
        'sack_defense', 'fumble_recovery_defense', 'ngs_air_yards', 'time_to_throw',
        'was_pressure' # n_offense/defense handled separately below
    ]

    for col in numeric_pbp_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # print(f"Converted '{col}' to numeric (coerced errors to NaN).") # Removed for cleaner loop output

    cols_to_fill_zero_pbp = [
        'qb_hit', 'sack', 'touchdown', 'pass_attempt', 'rush_attempt',
        'interception', 'fumble_lost', 'complete_pass', 'qb_scramble',
        'penalty_yards', 'first_down_pass', 'first_down_rush', 'first_down_penalty',
        'third_down_converted', 'third_down_failed', 'fourth_down_converted',
        'fourth_down_failed', 'fumble_recovery_2pt', 'safety',
        'field_goal_attempt', 'extra_point_attempt', 'two_point_attempt',
        'punt_blocked', 'extra_point_blocked', 'field_goal_blocked',
        'fumble_forced', 'fumble_recovery_fumble', 'fumble_recovery_opponent',
        'solo_tackle', 'tackles_for_loss', 'assist_tackle', 'qb_hit_defense',
        'interception_defense', 'sack_defense', 'fumble_recovery_defense',
        'special_teams_tds', 'passing_2pt_conversions', 'rushing_2pt_conversions',
        'receiving_2pt_conversions', 'n_offense', 'n_defense', 'was_pressure'
    ]
    for col in cols_to_fill_zero_pbp:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            if pd.api.types.is_float_dtype(df[col]) and (df[col] == df[col].astype(int)).all():
                df[col] = df[col].astype(int)
    # print("Filled specific common NaNs in PBP with 0 and converted to int where applicable.") # Removed for cleaner loop output

    # Convert n_offense and n_defense to int specifically if they are still float
    for col in ['n_offense', 'n_defense']:
        if col in df.columns and pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].fillna(0).astype(int)
            # print(f"Converted '{col}' to int.") # Removed for cleaner loop output
    return df

# --- Helper Functions for Firebase Storage ---
def download_parquet_from_firebase(filename):
    """Download a Parquet file from Firebase Storage and return a DataFrame."""
    blob = bucket.blob(f"{FOLDER_PATH}/{filename}")
    if blob.exists():
        data = blob.download_as_bytes()
        return pd.read_parquet(BytesIO(data))
    else:
        print(f"File not found in Firebase Storage: {filename}")
        return None

def upload_parquet_to_firebase(df, filename):
    """Upload a DataFrame as a Parquet file to Firebase Storage."""
    blob = bucket.blob(f"{FOLDER_PATH}/{filename}")
    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=False)
    blob.upload_from_string(out_buffer.getvalue(), content_type='application/octet-stream')
    print(f"Uploaded {filename} to Firebase Storage at {FOLDER_PATH}")

# --- Main Execution ---
if __name__ == "__main__":
    START_YEAR = 2024
    END_YEAR = 2024

    all_schedules_dfs = []
    all_weekly_stats_dfs = []
    all_pbp_dfs = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\nProcessing year: {year}")

        # Schedules
        df_schedules_year = download_parquet_from_firebase(f"nfl_schedules_{year}.parquet")
        if df_schedules_year is not None:
            df_schedules_year = clean_schedules_data(df_schedules_year)
            all_schedules_dfs.append(df_schedules_year)

        # Weekly Stats
        df_weekly_stats_year = download_parquet_from_firebase(f"nfl_weekly_stats_{year}.parquet")
        if df_weekly_stats_year is not None:
            df_weekly_stats_year = clean_weekly_stats_data(df_weekly_stats_year)
            all_weekly_stats_dfs.append(df_weekly_stats_year)

        # Play-by-Play
        df_pbp_year = download_parquet_from_firebase(f"nfl_pbp_{year}.parquet")
        if df_pbp_year is not None:
            df_pbp_year = clean_pbp_data(df_pbp_year)
            all_pbp_dfs.append(df_pbp_year)

        time.sleep(1)  # Polite pause

    # Concatenate all years
    df_schedules_all = pd.concat(all_schedules_dfs, ignore_index=True) if all_schedules_dfs else pd.DataFrame()
    df_weekly_stats_all = pd.concat(all_weekly_stats_dfs, ignore_index=True) if all_weekly_stats_dfs else pd.DataFrame()
    df_pbp_all = pd.concat(all_pbp_dfs, ignore_index=True) if all_pbp_dfs else pd.DataFrame()

    # Upload concatenated DataFrames to Firebase
    if not df_schedules_all.empty:
        upload_parquet_to_firebase(df_schedules_all, "nfl_schedules_all_years.parquet")
    if not df_weekly_stats_all.empty:
        upload_parquet_to_firebase(df_weekly_stats_all, "nfl_weekly_stats_all_years.parquet")
    if not df_pbp_all.empty:
        upload_parquet_to_firebase(df_pbp_all, "nfl_pbp_all_years.parquet")

    print("\n--- All data cleaned, concatenated, and uploaded to Firebase Storage! ---")
