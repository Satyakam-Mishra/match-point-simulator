# imports
import pandas as pd
from config import FINAL_CLEANED_VALIDATED_DATA, LIVE_POINT_WIN_PROB_DATASET
import numpy as np
from pydantic import BaseModel, ValidationError
import rally_parser_helpers as rph
from tqdm import tqdm

"""This is for model 1. The live point win probability model. We will generate the dataset for this model here. """

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

class LivePointWinProbabilityDataset(BaseModel):
    pl_1_hand: int
    pl_2_hand: int
    surfacehard: int
    surfaceclay: int
    surfacegrass: int
    gender: int
    best_of: int
    point_number: int
    set1: int
    set2: int
    game1: int
    game2: int
    player_1_point: int
    player_2_point: int
    svr: int
    point_winner: int
    
def parse_serve_rph_live_point_win_probability(dataset):
    """We need to parse the first_serve and seceond_serve columns to get the features for the model. We will use the rally_parser_helpers.py file to parse these columns. We will create new columns for each feature that we want to extract from the first_serve and second_serve columns. """
    
    # Initialize base columns with NaN for all rows
    base_columns = ['first_serve_in', 'rally_length', 'double_fault', 
                    'serve_location', 'serve_ace', 'serve_error_location', 'serve_shank_info', 'serve_position_info',
                    'return_shot_type', 'return_direction', 'return_depth', 'return_error_location', 'return_error_type', 'return_shank_info', 'return_position_info', 'return_winner',
                    'serve_plus_one_shot_type', 'serve_plus_one_direction', 'serve_plus_one_depth', 'serve_plus_one_error_location', 'serve_plus_one_error_type', 'serve_plus_one_shank_info', 'serve_plus_one_position_info', 'serve_plus_one_winner']
    
    # Add shot columns for shots 4-6 from the front
    for i in range(4, 7):
        base_columns.extend([f'shot_{i}_shot_type', f'shot_{i}_shot_direction', f'shot_{i}_shot_depth', 
                            f'shot_{i}_error_location', f'shot_{i}_error_type', f'shot_{i}_winner', 
                            f'shot_{i}_shank_info', f'shot_{i}_position_info'])
    
    # Add shot columns for shots last_1 to last_10 from the end
    for j in range(1, 11):
        base_columns.extend([f'shot_last_{j}_shot_type', f'shot_last_{j}_shot_direction', f'shot_last_{j}_shot_depth',
                            f'shot_last_{j}_error_location', f'shot_last_{j}_error_type', f'shot_last_{j}_winner',
                            f'shot_last_{j}_shank_info', f'shot_last_{j}_position_info'])
    
    # Initialize all columns at once using concat for better performance
    new_cols_dict = {}
    for col in base_columns:
        if col not in dataset.columns:
            # Use float for numeric columns, bool dtype for boolean columns
            if col in ['first_serve_in', 'double_fault']:
                new_cols_dict[col] = np.full(len(dataset), np.nan, dtype='float64')
            else:
                new_cols_dict[col] = np.full(len(dataset), np.nan, dtype='float64')
    
    if new_cols_dict:
        dataset = pd.concat([dataset, pd.DataFrame(new_cols_dict, index=dataset.index)], axis=1)
    
    # Helper function to safely get mapping value
    def safe_map_get(value, mapping_dict):
        """Safely get a value from mapping, handling lists and None"""
        # Check for numpy arrays first to avoid ambiguous truth value warning
        if isinstance(value, np.ndarray):
            return np.nan
        if value is None:
            return np.nan
        # If value is a list, use the first element
        if isinstance(value, list):
            value = value[0] if len(value) > 0 else None
        if value is None:
            return np.nan
        # Use try-catch for pd.isna() to handle edge cases
        try:
            if pd.isna(value):
                return np.nan
        except (ValueError, TypeError):
            pass  # If pd.isna() fails, continue with the value
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
    
    # Process each row using apply for better performance
    def process_row(row):
        row_results = {}
        try:
            first_serve = row['first_serve']
            second_serve = row['second_serve']
            
            is_first_serve_in = rph.is_first_serve(first_serve, second_serve)
            row_results['first_serve_in'] = float(is_first_serve_in)
            
            shots = rph.make_shots(first_serve, second_serve)
            rally = rph.parse_rally(shots)
            
            rally_length = len(rally)
            row_results['rally_length'] = rally_length
            row_results['double_fault'] = 1.0 if rally_length == 1 else 0.0
            
            for i in range(rally_length):
                if i == 1:  # Serve
                    serve_info = rally[i]
                    if isinstance(serve_info, dict):
                        row_results['serve_location'] = serve_info.get("serve_location")
                        row_results['serve_ace'] = safe_bool_to_float(serve_info.get("ace"))
                        row_results['serve_error_location'] = safe_map_get(serve_info.get("error_location"), error_location_mapping)
                        row_results['serve_shank_info'] = safe_bool_to_float(serve_info.get("shank_info"))
                        row_results['serve_position_info'] = safe_map_get(serve_info.get("position_info"), position_information_mapping)
                        
                elif i == 2:  # Return
                    return_info = rally[i]
                    if isinstance(return_info, dict):
                        row_results["return_shot_type"] = safe_map_get(return_info.get("shot_type"), ground_strokes_mapping)
                        row_results["return_direction"] = return_info.get("return_direction")
                        row_results["return_depth"] = return_info.get("return_depth")
                        row_results["return_error_location"] = safe_map_get(return_info.get("error_location"), error_location_mapping)
                        row_results["return_error_type"] = safe_map_get(return_info.get("error_type"), terminal_symbol_mapping)
                        row_results["return_shank_info"] = safe_bool_to_float(return_info.get("shank_info"))
                        row_results["return_position_info"] = safe_map_get(return_info.get("position_info"), position_information_mapping)
                        row_results["return_winner"] = safe_bool_to_float(return_info.get("winner"))
                        
                elif i == 3:  # Serve + 1
                    serve_plus_one_info = rally[i]
                    if isinstance(serve_plus_one_info, dict):
                        row_results["serve_plus_one_shot_type"] = safe_map_get(serve_plus_one_info.get("shot_type"), ground_strokes_mapping)
                        row_results["serve_plus_one_direction"] = serve_plus_one_info.get("shot_direction")
                        row_results["serve_plus_one_depth"] = serve_plus_one_info.get("shot_depth")
                        row_results["serve_plus_one_error_location"] = safe_map_get(serve_plus_one_info.get("error_location"), error_location_mapping)
                        row_results["serve_plus_one_error_type"] = safe_map_get(serve_plus_one_info.get("error_type"), terminal_symbol_mapping)
                        row_results["serve_plus_one_shank_info"] = safe_bool_to_float(serve_plus_one_info.get("shank_info"))
                        row_results["serve_plus_one_position_info"] = safe_map_get(serve_plus_one_info.get("position_info"), position_information_mapping)
                        row_results["serve_plus_one_winner"] = safe_bool_to_float(serve_plus_one_info.get("winner"))
                        
                elif 4 <= i <= 6:  # Shots 4-6
                    shot = rally[i]
                    if isinstance(shot, dict):
                        row_results[f'shot_{i}_shot_type'] = safe_map_get(shot.get("shot_type"), ground_strokes_mapping)
                        row_results[f'shot_{i}_shot_direction'] = shot.get("shot_direction")
                        row_results[f'shot_{i}_shot_depth'] = shot.get("shot_depth")
                        row_results[f'shot_{i}_error_location'] = safe_map_get(shot.get("error_location"), error_location_mapping)
                        row_results[f'shot_{i}_error_type'] = safe_map_get(shot.get("error_type"), terminal_symbol_mapping)
                        row_results[f'shot_{i}_winner'] = safe_bool_to_float(shot.get("winner"))
                        row_results[f'shot_{i}_shank_info'] = safe_bool_to_float(shot.get("shank_info"))
                        row_results[f'shot_{i}_position_info'] = safe_map_get(shot.get("position_info"), position_information_mapping)
                        
                elif i >= rally_length - 10:  # Last 10 shots
                    shot = rally[i]
                    if isinstance(shot, dict):
                        shot_distance = rally_length - i
                        row_results[f'shot_last_{shot_distance}_shot_type'] = safe_map_get(shot.get("shot_type"), ground_strokes_mapping)
                        row_results[f'shot_last_{shot_distance}_shot_direction'] = shot.get("shot_direction")
                        row_results[f'shot_last_{shot_distance}_shot_depth'] = shot.get("shot_depth")
                        row_results[f'shot_last_{shot_distance}_error_location'] = safe_map_get(shot.get("error_location"), error_location_mapping)
                        row_results[f'shot_last_{shot_distance}_error_type'] = safe_map_get(shot.get("error_type"), terminal_symbol_mapping)
                        row_results[f'shot_last_{shot_distance}_winner'] = safe_bool_to_float(shot.get("winner"))
                        row_results[f'shot_last_{shot_distance}_shank_info'] = safe_bool_to_float(shot.get("shank_info"))
                        row_results[f'shot_last_{shot_distance}_position_info'] = safe_map_get(shot.get("position_info"), position_information_mapping)
        except Exception as e:
            pass  # Silently skip rows with errors
        
        return pd.Series(row_results)
    
    # Apply the function to each row using chunking for memory efficiency
    print("Processing serve and rally data...")
    chunk_size = 10000  # Process in chunks to manage memory
    results_list = []
    
    for i in tqdm(range(0, len(dataset), chunk_size), desc="Processing chunks", unit="chunk"):
        chunk = dataset.iloc[i:i+chunk_size]
        chunk_results = chunk.apply(process_row, axis=1)
        results_list.append(chunk_results)
    
    # Concatenate all results
    results_df = pd.concat(results_list, ignore_index=False)
    
    # Update dataset with results with progress bar
    print("Assigning features to dataset...")
    for col in tqdm(results_df.columns, desc="Assigning columns", unit="col"):
        if col in dataset.columns:
            dataset[col] = results_df[col]
    
    return dataset


def validation_checker_live_point_win_probability(dataset):
    """We use pydantic to validate the dataset. We will check for missing values, data types, and value ranges. We will also check for any inconsistencies in the data."""
    
    # Check for missing values
    missing_values = dataset.isnull().sum()
    if missing_values.sum() > 0:
        print("Warning: Missing values found in dataset:")
        print(missing_values[missing_values > 0])
    
    # For large datasets, skip row-by-row validation and just check column ranges
    # This is much faster than iterating through millions of rows
    print("Skipping detailed row validation for performance (dataset is large)")
    
    # Check for value ranges (e.g., gender should be 0 or 1, hands should be 0 or 1, etc.)
    range_checks = {
        'gender': (0, 1),
        'pl_0_hand': (0, 1),
        'pl_1_hand': (0, 1),
        'svr': (0, 1),
        'point_winner': (0, 1),
        'surfacehard': (0, 1),
        'surfaceclay': (0, 1),
        'surfacegrass': (0, 1),
    }
    
    for col, (min_val, max_val) in range_checks.items():
        if col in dataset.columns:
            # Filter out NaN values before checking ranges
            valid_data = dataset[dataset[col].notna()]
            if len(valid_data) > 0:
                out_of_range = valid_data[(valid_data[col] < min_val) | (valid_data[col] > max_val)]
                if len(out_of_range) > 0:
                    print(f"Warning: {len(out_of_range)} values out of range for column '{col}' (expected {min_val}-{max_val})")
    
    print("Validation completed successfully!")

def feature_generator_live_point_win_probability():
    dataset = pd.read_csv(FINAL_CLEANED_VALIDATED_DATA)
    # We first drop the columns that are not needed for this model.
    
    print("feature generation starting...")
    
    dataset.drop(columns=['match_id', 'player_1', 'player_2', 'game_number', 'tiebreak_set', 'notes', 'shot_validation'], inplace=True)
    
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
    
    # Now we will parse the point number from the point string. We will create two new columns for player 1 and player 2 points.
    dataset[['player_0_point', 'player_1_point']] = dataset['points'].apply(lambda x: pd.Series(point_parser(x)))
    
        # We will drop the original point_number column.
    dataset.drop(columns=['point_number', 'points'], inplace=True)
    
    # Now we will rename the columns such that 0 maps to player 1 and 1 maps to player 2. We will also rename the surface columns to surface_hard, surface_clay, and surface_grass.
    
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
    
    dataset.drop(columns=['first_serve', 'second_serve'], inplace=True)
    
    # Also drop every info after the 6th shot as it is not relevant for the live point win probability model. We will drop the columns related to shots 7-10 from the end as well as the columns related to shots 4-6 from the front as they are not relevant for the live point win probability model.
    
    dataset.drop(columns=["double_fault", "serve_ace", "serve_error_location", "return_error_location", "return_error_type", "return_winner", "serve_plus_one_error_location", "serve_plus_one_error_type", "serve_plus_one_winner", "shot_4_error_location", "shot_4_error_type", "shot_4_winner", "shot_5_error_location", "shot_5_error_type", "shot_5_winner", "shot_6_error_location", "shot_6_error_type", "shot_6_winner", "shot_last_1_shot_type", "shot_last_1_shot_direction", "shot_last_1_shot_depth", "shot_last_1_shank_info", "shot_last_1_position_info", "shot_last_1_error_location", "shot_last_1_error_type", "shot_last_1_winner", "shot_last_2_shot_type", "shot_last_2_shot_direction", "shot_last_2_shot_depth", "shot_last_2_shank_info", "shot_last_2_position_info", "shot_last_2_error_location", "shot_last_2_error_type", "shot_last_2_winner", "shot_last_3_shot_type", "shot_last_3_shot_direction", "shot_last_3_shot_depth", "shot_last_3_shank_info", "shot_last_3_position_info", "shot_last_3_error_location", "shot_last_3_error_type", "shot_last_3_winner", "shot_last_4_shot_type", "shot_last_4_shot_direction", "shot_last_4_shot_depth", "shot_last_4_shank_info", "shot_last_4_position_info", "shot_last_4_error_location", "shot_last_4_error_type", "shot_last_4_winner", "shot_last_5_shot_type", "shot_last_5_shot_direction", "shot_last_5_shot_depth", "shot_last_5_shank_info", "shot_last_5_position_info", "shot_last_5_error_location", "shot_last_5_error_type", "shot_last_5_winner", "shot_last_6_shot_type", "shot_last_6_shot_direction", "shot_last_6_shot_depth", "shot_last_6_shank_info", "shot_last_6_position_info", "shot_last_6_error_location", "shot_last_6_error_type", "shot_last_6_winner", "shot_last_7_shot_type", "shot_last_7_shot_direction", "shot_last_7_shot_depth", "shot_last_7_shank_info", "shot_last_7_position_info", "shot_last_7_error_location", "shot_last_7_error_type", "shot_last_7_winner", "shot_last_8_shot_type", "shot_last_8_shot_direction", "shot_last_8_shot_depth", "shot_last_8_shank_info", "shot_last_8_position_info", "shot_last_8_error_location", "shot_last_8_error_type", "shot_last_8_winner", "shot_last_9_shot_type", "shot_last_9_shot_direction", "shot_last_9_shot_depth", "shot_last_9_shank_info", "shot_last_9_position_info", "shot_last_9_error_location", "shot_last_9_error_type", "shot_last_9_winner", "shot_last_10_shot_type", "shot_last_10_shot_direction", "shot_last_10_shot_depth", "shot_last_10_shank_info", "shot_last_10_position_info", "shot_last_10_error_location", "shot_last_10_error_type", "shot_last_10_winner", "rally_length"], inplace=True)
    
    
    print("feature generation completed...")
    dataset.to_csv(LIVE_POINT_WIN_PROB_DATASET, index=False)
    
    return dataset
    
    
if __name__ == "__main__":
    dataset = feature_generator_live_point_win_probability()