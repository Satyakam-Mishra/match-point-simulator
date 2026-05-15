"""This id tehn feature generating code for the Best Return Based on Serve model"""

# imports
import pandas as pd
from config import FINAL_CLEANED_VALIDATED_DATA, BEST_RETURN_BASED_ON_SERVE_DATASET
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
    
def parse_serve_rph_live_point_win_probability(dataset):
    """We need to parse the first_serve and seceond_serve columns to get the features for the model. We will use the rally_parser_helpers.py file to parse these columns. We will create new columns for each feature that we want to extract from the first_serve and second_serve columns. """
    
    # Initialize base columns with NaN for all rows - only serve and return data
    base_columns = ['first_serve_in', 'rally_length', 'double_fault', 
                    'serve_location', 'serve_ace', 'serve_error_location', 'serve_shank_info', 'serve_position_info',
                    'return_shot_type', 'return_direction', 'return_depth', 'return_error_location', 'return_error_type', 'return_shank_info', 'return_position_info', 'return_winner']
    
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
                if i == 0:  # Serve
                    serve_info = rally[i]
                    if isinstance(serve_info, dict):
                        row_results['serve_location'] = serve_info.get("serve_location")
                        row_results['serve_ace'] = safe_bool_to_float(serve_info.get("ace"))
                        row_results['serve_error_location'] = safe_map_get(serve_info.get("error_location"), error_location_mapping)
                        row_results['serve_shank_info'] = safe_bool_to_float(serve_info.get("shank_info"))
                        row_results['serve_position_info'] = safe_map_get(serve_info.get("position_info"), position_information_mapping)
                        
                elif i == 1:  # Return
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

def feature_generator_brbs():
    """This function will generate the features for the Best Return Based on Serve model. We will read the final cleaned validated data and then parse the first_serve and second_serve columns to get the features for the model. We will then save the dataset with the new features to a csv file. """
    
    # Read the final cleaned validated data
    dataset = pd.read_csv(FINAL_CLEANED_VALIDATED_DATA)
    
    # Now we dorp all the columns not needed by us that is all the columns which are not useful
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
    
    # We keep the data only relevant to our model.
    
    dataset.drop(columns=['best_of', 'point_number', 'set0', 'set1', 'game0', 'game1', 'first_serve', 'second_serve', 'rally_length'], inplace=True)
    
    # We remove all the rows in which the returner won due to a double fault.
    dataset = dataset[~((dataset['double_fault'] == 1.0))]
    # We remove all the rows in which the point is won by the server as we are only interested in the points won by the returner.
    dataset = dataset[dataset['point_winner'] != dataset['svr']]
    
    
    # Save the dataset with the new features to a csv file
    dataset.to_csv(BEST_RETURN_BASED_ON_SERVE_DATASET, index=False)
    
if __name__ == "__main__":
    feature_generator_brbs()