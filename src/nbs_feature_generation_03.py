"""This generates the features for the next best shot predictor model. We use the long format for this model. Each row corresponds to a shot in a point, and we have a column for the shot_number 1, 2, 3, etc. We will only keep the rows which are either a winner or forced error. """

from config import FINAL_CLEANED_VALIDATED_DATA, NEXT_BEST_SHOT_DATASET
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError
import rally_parser_helpers as rph
from tqdm import tqdm

ground_strokes_mapping = {'f': 0, 'b': 1, 's': 2, 'r': 3, 'v': 4, 'o': 5, 'l': 6, 'm': 7, 'z': 8, 'j': 9, 'q': 10, 't': 11, 'p': 12, 'u': 13, 'y': 14}

position_information_mapping = {'=': 0, '+': 1, '-': 2}

terminal_symbol_mapping = {'*': 0, '#': 1, '@': 2}

error_location_mapping = {'n': 0, 'w': 1, 'd': 2, 'x': 3}



def point_parser(point):
    """This function will parse the point number from the point string. The point string is in the format of '0-15', '15-30', '30-40', '40-40', 'Ad-40', '40-Ad'. ."""
    
    # Handle NaN, None, and non-string values
    if pd.isna(point) or not isinstance(point, str):
        return np.nan, np.nan
    
    if point == '0-0':
        return 0, 0
    elif point == '15-0':
        return 1, 0
    elif point == '30-0':
        return 2, 0
    elif point == '40-0':
        return 3, 0
    elif point == '0-15':   
        return 0, 1
    elif point == '15-15':
        return 1, 1
    elif point == '30-15':
        return 2, 1
    elif point == '40-15':
        return 3, 1
    elif point == '0-30':
        return 0, 2
    elif point == '15-30':
        return 1, 2
    elif point == '30-30':  
        return 2, 2
    elif point == '40-30':
        return 3, 2
    elif point == '0-40':
        return 0, 3
    elif point == '15-40':
        return 1, 3
    elif point == '30-40':
        return 2, 3
    elif point == '40-40':
        return 3, 3
    elif point == 'AD-40':
        return 4, 3
    elif point == '40-AD':
        return 3, 4
    else:
        point_list = point.split("-")
        if len(point_list) == 2:
            try:
                player_1_point = int(point_list[0])
                player_2_point = int(point_list[1])
                return player_1_point, player_2_point
            except ValueError:
                return np.nan, np.nan # return NaN if the point string is not in the mapping

    return np.nan, np.nan # return NaN if the point string is not in the mapping

def parse_serve_rph_live_point_win_probability(dataset):
    """Create long-format dataset where each row is a shot in a rally, including previous shot information and rally outcome flags."""
    
    def safe_map_get(value, mapping_dict):
        """Safely get a value from mapping, handling lists and None"""
        if isinstance(value, np.ndarray):
            return np.nan
        if value is None:
            return np.nan
        if isinstance(value, list):
            value = value[0] if len(value) > 0 else None
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except (ValueError, TypeError):
            pass
        return mapping_dict.get(value, np.nan)
    
    def safe_bool_to_float(value):
        """Convert boolean to float (1.0 or 0.0)"""
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, np.ndarray):
            return np.nan
        if value is None:
            return np.nan
        try:
            if pd.isna(value):
                return np.nan
        except (ValueError, TypeError):
            pass
        return float(value) if value is not None else np.nan
    
    def extract_shot_features(shot):
        """Extract all features from a shot dictionary"""
        if not isinstance(shot, dict):
            return {}
        
        features = {
            'shot_type': safe_map_get(shot.get("shot_type"), ground_strokes_mapping),
            'shot_direction': shot.get("shot_direction"),
            'shot_depth': shot.get("shot_depth"),
            'error_location': safe_map_get(shot.get("error_location"), error_location_mapping),
            'error_type': safe_map_get(shot.get("error_type"), terminal_symbol_mapping),
            'winner': safe_bool_to_float(shot.get("winner")),
            'shank_info': safe_bool_to_float(shot.get("shank_info")),
            'position_info': safe_map_get(shot.get("position_info"), position_information_mapping),
        }
        return features
    
    def get_rally_outcome(rally):
        """Determine rally outcome and shot number it occurred at"""
        last_shot = rally[-1] if isinstance(rally[-1], dict) else {}
        winner = last_shot.get("winner", False)
        error_type = last_shot.get("error_type", None)
        
        if winner:
            return "winner", len(rally)
        elif error_type == "forced_error":
            return "forced_error", len(rally)
        elif error_type == "unforced_error":
            return "unforced_error", len(rally)
        else:
            return None, None
    
    def determine_shot_flag(shot_num, rally_length, outcome):
        """Determine if shot is SETUP or DECISIVE"""
        if outcome is None:
            return None
        
        # For forced errors: make second-to-last DECISIVE, mark leading shots as SETUP
        if outcome == "forced_error":
            winner_shot = rally_length - 1  # Second-to-last is the decisive shot
            if shot_num == winner_shot:
                return "DECISIVE"
            # Mark even shots before the decisive shot as SETUP
            if shot_num % 2 == 0 and shot_num < winner_shot:
                return "SETUP"
            return None
        
        # For winners and unforced errors: standard logic
        # Winner/error occurred at rally_length
        winner_shot = rally_length
        
        if shot_num == winner_shot:
            return "DECISIVE"
        
        # Check if this is an odd-numbered setup shot
        if winner_shot % 2 == 0:  # Even shot number wins (e.g., 6th shot)
            if shot_num % 2 == 0 and shot_num < winner_shot:  # Even shots before winner
                return "SETUP"
        else:  # Odd shot number wins (e.g., 7th shot)
            if shot_num % 2 == 1 and shot_num < winner_shot:  # Odd shots before winner
                return "SETUP"
        
        return None
    
    def process_point(row):
        """Process a single point and generate shot rows"""
        shot_rows = []
        try:
            first_serve = row['first_serve']
            second_serve = row['second_serve']
            
            is_first_serve_in = rph.is_first_serve(first_serve, second_serve)
            shots = rph.make_shots(first_serve, second_serve)
            rally = rph.parse_rally(shots)
            
            rally_length = len(rally)
            outcome, outcome_shot_num = get_rally_outcome(rally)
            
            # For forced errors, exclude the last shot (error shot)
            shots_to_process = rally_length
            if outcome == "forced_error":
                shots_to_process = rally_length - 1
            
            # Create a row for each shot in the rally
            for shot_idx in range(shots_to_process):
                shot_data = {}
                
                # Copy match context columns from original row
                for col in dataset.columns:
                    if col not in ['first_serve', 'second_serve']:
                        shot_data[col] = row[col]
                
                # Add rally info
                shot_data['first_serve_in'] = float(is_first_serve_in)
                shot_data['rally_length'] = rally_length
                shot_data['shot_number'] = shot_idx + 1
                
                # Add current shot features
                current_shot = rally[shot_idx]
                current_features = extract_shot_features(current_shot)
                for key, val in current_features.items():
                    shot_data[f'shot_{key}'] = val
                
                # Add previous shot features (NaN if first shot)
                if shot_idx > 0:
                    prev_shot = rally[shot_idx - 1]
                    prev_features = extract_shot_features(prev_shot)
                    for key, val in prev_features.items():
                        shot_data[f'prev_shot_{key}'] = val
                else:
                    # First shot has no previous shot
                    for key in ['shot_type', 'shot_direction', 'shot_depth', 'error_location', 'error_type', 'winner', 'shank_info', 'position_info']:
                        shot_data[f'prev_shot_{key}'] = np.nan
                
                # Add rally outcome information
                shot_data['rally_outcome'] = outcome
                shot_data['outcome_shot_number'] = outcome_shot_num
                shot_data['shot_flag'] = determine_shot_flag(shot_idx + 1, rally_length, outcome)
                
                shot_rows.append(shot_data)
        
        except Exception as e:
            pass
        
        return shot_rows
    
    # Process all points and expand to shot-level rows
    print("Processing points to shot-level data...")
    all_shot_rows = []
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing points"):
        shot_rows = process_point(row)
        all_shot_rows.extend(shot_rows)
    
    # Create new dataframe from shot rows
    print("Creating shot-level dataset...")
    shot_dataset = pd.DataFrame(all_shot_rows)
    
    # Create y_string column with shot characteristics and flag
    print("Creating y_string target variable...")
    shot_dataset['y_string'] = shot_dataset.apply(y_string_creater, axis=1)
    
    return shot_dataset

def y_string_creater(row):
    """Create target string including shot type, direction, depth, and flag (SETUP/DECISIVE)
    Format: shot_type_direction_depth_flag
    Example: 0_2_8_DECISIVE or 1_1_7_SETUP
    """
    shot_type = row['shot_type']
    shot_direction = row['shot_direction']
    shot_depth = row['shot_depth']
    shot_flag = row['shot_flag']
    
    # Return NaN if any required field is missing
    if pd.isna(shot_type) or pd.isna(shot_direction) or pd.isna(shot_depth):
        return np.nan
    
    # Include shot_flag if it exists (SETUP/DECISIVE), otherwise empty string
    flag_str = f"_{shot_flag}" if pd.notna(shot_flag) else ""
    
    return f"{int(shot_type)}_{shot_direction}_{shot_depth}{flag_str}"

def feature_creation():
    """This function creates the features for the next best shot predictor model."""
    # load the final cleaned validated data
    dataset = pd.read_csv(FINAL_CLEANED_VALIDATED_DATA)
    
    dataset.drop(columns = ["match_id", "player_1", "player_2", "game_number", "tiebreak_set", "notes", "shot_validation"], inplace=True)
    
    # Now we create string to int mapping for the categorical variables. We will use one-hot encoding for the surface. We will use label encoding for the shot type and shot outcome.
    
    # One-hot encoding pl_hand_1 and pl_hand_2
    pl_hand_mapping = {'R': 0, 'L': 1}
    dataset['pl_1_hand'] = dataset['pl_1_hand'].map(pl_hand_mapping)
    dataset['pl_2_hand'] = dataset['pl_2_hand'].map(pl_hand_mapping)
    
    # One-hot encoding surface
    surface_dummies = pd.get_dummies(dataset['surface'], prefix='surface')
    dataset = pd.concat([dataset, surface_dummies], axis=1)
    dataset.drop(columns=['surface'], inplace=True)
    
    #for gender, we will use label encoding. We will map M to 0 and W to 1.
    
    gender_mapping = {'M': 0, 'W': 1}
    dataset['gender'] = dataset['gender'].map(gender_mapping)
    
    dataset[['player_0_point', 'player_1_point']] = dataset['points'].apply(lambda x: pd.Series(point_parser(x)))
    
    dataset.drop(columns=['points'], inplace=True)
    
    dataset.rename(columns={'surface_hard': 'surfacehard', 'surface_clay': 'surfaceclay', 'surface_grass': 'surfacegrass', 'pl_1_hand': 'pl_0_hand', 'pl_2_hand': 'pl_1_hand', 'set1': 'set0', 'set2': 'set1', 'game1': 'game0', 'game2': 'game1', 'Svr': 'svr'}, inplace=True)
    
    dataset['svr'] = dataset['svr'] - 1
    dataset['point_winner'] = dataset['point_winner'] - 1
    
    # Now we will be finally parsing the first_serve and second_serve columns. As this is complicated we will use another function to implement this these columns.
    
    # adding a new column for is_tiebreaker. We see the score in the points as well as in the game score. However it is problematic in the case of next generation finals as the tiebreak starts at 3-3. So we so the following, if we see that the points are eiter 1-0 or 0-1 then we even set the previous 0-0 score row as tiebreaker.
    
    print("Detecting tiebreakers...")
    # Use vectorized operations for performance on large datasets
    p0 = dataset['player_0_point']
    p1 = dataset['player_1_point']
    
    # Create boolean mask for tiebreaker conditions
    is_tiebreaker = (
        ((p0 == 1) & (p1 == 0)) |  # 1-0
        ((p0 == 0) & (p1 == 1)) |  # 0-1
        ((p0 == 2) & (p1 == 0)) |  # 2-0
        ((p0 == 0) & (p1 == 2)) |  # 0-2
        ((p0 == 3) & (p1 == 0)) |  # 3-0
        ((p0 == 0) & (p1 == 3)) |  # 0-3
        ((p0 == 4) & (p1 == 0)) |  # 4-0
        ((p0 == 0) & (p1 == 4)) |  # 0-4
        ((p0 == 5) & (p1 == 0)) |  # 5-0
        ((p0 == 0) & (p1 == 5)) |  # 0-5
        ((p0 == 6) & (p1 == 0)) |  # 6-0
        ((p0 == 0) & (p1 == 6)) |  # 0-6
        ((p0.isin([1, 2, 3, 4, 5, 6])) & (p1.isin([1, 2, 3, 4, 5, 6])))  # deuce points
    )
    
    dataset['is_tiebreaker'] = is_tiebreaker
    
    # Handle previous row marking for (1,0) and (0,1) cases
    previous_row_marker = ((p0.shift(-1) == 1) & (p1.shift(-1) == 0)) | ((p0.shift(-1) == 0) & (p1.shift(-1) == 1))
    dataset.loc[previous_row_marker, 'is_tiebreaker'] = True
    
    dataset = parse_serve_rph_live_point_win_probability(dataset)
    
    # removing the rows with "unforced_error" as the rally outcome as we are only interested in winners and forced errors for the next best shot predictor model.
    dataset = dataset[dataset['rally_outcome'].isin(['winner', 'forced_error'])]
    
    # Then we remove all the rows which do not have either "SETUP" or "DECISIVE" as the shot flag as we are only interested in these shots for the next best shot predictor model.
    dataset = dataset[dataset['shot_flag'].isin(['SETUP', 'DECISIVE'])]
    
    # save the featured data
    dataset.to_csv(NEXT_BEST_SHOT_DATASET, index=False)
    
if __name__ == "__main__":
    feature_creation()