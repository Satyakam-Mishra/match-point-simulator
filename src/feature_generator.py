import pandas as pd
import numpy as np
import re
from collections import defaultdict
from config import FINAL_CLEANED_DATA, FEATURED_DATA

ground_strokes = ['f', 'b', 's', 'r', 'v', 'o', 'l', 'm', 'z', 'j', 'q', 't', 'p', 'u', 'y']

shot_direction = ['1', '2', '3']

serve_direction = ['4', '5', '6']

return_depth = ['7', '8', '9']

position_information = ['=', '+', '-']

terminal_symbols = ['*', '#', '@']

error_location = ['n', 'w', 'd', 'x']

def is_first_serve(first_serve, second_serve) -> bool:
    """Checks if the point starts with a second serve or a first serve."""
    # Treat NaN and empty strings as missing data
    first_serve_empty = pd.isna(first_serve) or first_serve == ""
    second_serve_empty = pd.isna(second_serve) or second_serve == ""
    
    if first_serve_empty and second_serve_empty:
        return np.nan  # Missing data  
    elif second_serve_empty:
        return True  # Only first serve info available, assume first serve
    else:
        return False # Second serve info available, so it's not a first serve
    
def make_shots(first_serve, second_serve) -> list:
    """Extracts shot types from the first and second serve strings."""
    shots = []
    is_first = is_first_serve(first_serve, second_serve)
    if is_first:
        sequence = first_serve
    else:
        sequence = second_serve
        
    new_shot = ""
    for char in sequence:
        if char in ground_strokes:
            if new_shot:  # Only append if not empty
                shots.append(new_shot)
            new_shot = char
        else:
            new_shot += char
    
    # Append the final shot
    if new_shot:
        shots.append(new_shot)
            
    return shots

def parse_rally(first_serve, second_serve) -> list:
    pass
            
    
    
            
        
            
        

def create_features(final_cleaned_data, verbose = False):
    """
    Main feature engineering pipeline. Applies all feature extraction functions
    and returns enriched dataframe with 1 row per point (no explosions).
    
    Args:
        df: Input dataframe with columns:
            match_id, player_1, player_2, pl_1_hand, pl_2_hand, surface, best_of,
            gender, point_number, set1, set2, game1, game2, points, game_number,
            tiebreak_set, Svr, first_serve, second_serve, notes, point_winner
        verbose: If True, print progress messages
        
    Returns:
        Enriched dataframe with additional feature columns
    """
    pass
    
    

        
    
    
    

def run_feature_generation(verbose = False):
    if verbose:
        print("Loading cleaned data...")
        
    final_cleaned_data = pd.read_csv(FINAL_CLEANED_DATA)
    
    enriched_df = create_features(final_cleaned_data, verbose = verbose)
    
    enriched_df.to_csv(FEATURED_DATA, index=False)
    
    if verbose:
        print(f"Features saved to {FEATURED_DATA}")
        
if __name__ == "__main__":
    run_feature_generation(verbose=True)