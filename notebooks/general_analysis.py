"""This is the general analysis of the dataset."""

# Import necessary libraries
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel
from tqdm import tqdm

try:
    from src.config import FINAL_CLEANED_VALIDATED_DATA
    import src.rally_parser_helpers as rph
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from config import FINAL_CLEANED_VALIDATED_DATA
    import rally_parser_helpers as rph

# importing some necessary functions for analysis


# Load the dataset
dataset = pd.read_csv(FINAL_CLEANED_VALIDATED_DATA)

# Helper functions for analysis

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
                        
                elif i == 2:  # Serve + 1
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
                        
                elif 3 <= i <= 5:  # Shots 4-6 (indices 3-5 in rally array)
                    shot = rally[i]
                    if isinstance(shot, dict):
                        shot_num = i + 1  # Convert to shot number (1-indexed)
                        row_results[f'shot_{shot_num}_shot_type'] = safe_map_get(shot.get("shot_type"), ground_strokes_mapping)
                        row_results[f'shot_{shot_num}_shot_direction'] = shot.get("shot_direction")
                        row_results[f'shot_{shot_num}_shot_depth'] = shot.get("shot_depth")
                        row_results[f'shot_{shot_num}_error_location'] = safe_map_get(shot.get("error_location"), error_location_mapping)
                        row_results[f'shot_{shot_num}_error_type'] = safe_map_get(shot.get("error_type"), terminal_symbol_mapping)
                        row_results[f'shot_{shot_num}_winner'] = safe_bool_to_float(shot.get("winner"))
                        row_results[f'shot_{shot_num}_shank_info'] = safe_bool_to_float(shot.get("shank_info"))
                        row_results[f'shot_{shot_num}_position_info'] = safe_map_get(shot.get("position_info"), position_information_mapping)
                        
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

def men_vs_women_number_of_points(dataset):
    """Compare the number of points played by men and women."""
    num_men = dataset[dataset["gender"] == "M"].shape[0]
    num_women = dataset[dataset["gender"] == "W"].shape[0]
    
    return num_men, num_women

def clay_vs_grass_vs_hard_number_of_points(dataset):
    """Compare the number of points played on clay, grass, and hard surfaces."""
    num_clay = dataset[dataset["surface"] == "Clay"].shape[0]
    num_grass = dataset[dataset["surface"] == "Grass"].shape[0]
    num_hard = dataset[dataset["surface"] == "Hard"].shape[0]
    
    return num_clay, num_grass, num_hard

def Left_handed_points_vs_Right_handed_points(dataset):
    """Compare the number of points played by left-handed and right-handed players."""
    num_left_handed = dataset[dataset["pl_0_hand"] == "L"].shape[0] + dataset[dataset["pl_1_hand"] == "L"].shape[0]
    num_right_handed = dataset[dataset["pl_0_hand"] == "R"].shape[0] + dataset[dataset["pl_1_hand"] == "R"].shape[0]
    
    return num_left_handed, num_right_handed

def total_points(dataset):
    """Calculate the total number of points in the dataset."""
    return dataset.shape[0]

def average_rally_length_per_point(dataset):
    """Calculate the average number of shots per point."""
    total_shots = sum(dataset["rally_length"])
    total_points = dataset.shape[0]
    return total_shots / total_points if total_points > 0 else 0

def average_rally_length_by_surface(dataset):
    """Calculate the average rally length for each surface type."""
    return dataset.groupby("surface")["rally_length"].mean()

def winner_vs_unforced_error_vs_forced_error(dataset):
    pass

def percentage_of_points_won_by_first_serve(dataset):
    """Calculate the percentage of points won by the player who served first."""
    total_first_serve_points = dataset[dataset["first_serve_in"] == True].shape[0]
    points_won_by_first_serve = dataset[dataset["point_winner"] == dataset["svr"]].shape[0]
    return (points_won_by_first_serve / total_first_serve_points) * 100 if total_first_serve_points > 0 else 0

def percentage_of_points_won_by_second_serve(dataset):
    """Calculate the percentage of points won by the player who served second."""
    total_second_serve_points = dataset[dataset["first_serve_in"] == False].shape[0]
    points_won_by_second_serve = dataset[dataset["point_winner"] == dataset["svr"]].shape[0]
    return (points_won_by_second_serve / total_second_serve_points) * 100 if total_second_serve_points > 0 else 0

def ratio_wide_serve_body_serve_T_serve(dataset):
    """Calculate the ratio of wide serves, body serves, and T serves."""
    total_serves = dataset.shape[0]
    wide_serves = dataset[dataset["serve_location"] == 4].shape[0]
    body_serves = dataset[dataset["serve_location"] == 5].shape[0]
    t_serves = dataset[dataset["serve_location"] == 6].shape[0]
    
    return {
        "wide_serve_ratio": wide_serves / total_serves if total_serves > 0 else 0,
        "body_serve_ratio": body_serves / total_serves if total_serves > 0 else 0,
        "t_serve_ratio": t_serves / total_serves if total_serves > 0 else 0
    }

def points_ending_by_serve_vs_by_return_vs_by_serve_plus_one_vs_others(dataset):
    """Calculate the percentage of points ending by serve, return, serve+1, and others."""
    total_points = dataset.shape[0]
    points_ending_by_serve = dataset[dataset["rally_length"] == 1].shape[0]
    points_ending_by_return = dataset[dataset["rally_length"] == 2].shape[0]
    points_ending_by_serve_plus_one = dataset[dataset["rally_length"] == 3].shape[0]
    points_ending_by_others = total_points - (points_ending_by_serve + points_ending_by_return + points_ending_by_serve_plus_one)
    
    return {
        "points_ending_by_serve_ratio": points_ending_by_serve / total_points if total_points > 0 else 0,
        "points_ending_by_return_ratio": points_ending_by_return / total_points if total_points > 0 else 0,
        "points_ending_by_serve_plus_one_ratio": points_ending_by_serve_plus_one / total_points if total_points > 0 else 0,
        "points_ending_by_others_ratio": points_ending_by_others / total_points if total_points > 0 else 0
    }

if __name__ == "__main__":
    dataset = parse_serve_rph_live_point_win_probability(dataset)
    
    num_men, num_women = men_vs_women_number_of_points(dataset)
    num_clay, num_grass, num_hard = clay_vs_grass_vs_hard_number_of_points(dataset)
    avg_rally_length_per_point = average_rally_length_per_point(dataset)
    avg_rally_length_by_surface = average_rally_length_by_surface(dataset)
    percentage_first_serve_points_won = percentage_of_points_won_by_first_serve(dataset)
    percentage_second_serve_points_won = percentage_of_points_won_by_second_serve(dataset)
    serve_ratios = ratio_wide_serve_body_serve_T_serve(dataset)
    points_ending_ratios = points_ending_by_serve_vs_by_return_vs_by_serve_plus_one_vs_others(dataset)