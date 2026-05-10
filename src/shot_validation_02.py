# Do the necessary imports
import pandas as pd
import numpy as np
from src.rally_parser_helpers import make_shots, is_first_serve
from config import FINAL_CLEANED_DATA, FINAL_CLEANED_VALIDATED_DATA

ground_strokes = ['f', 'b', 's', 'r', 'v', 'o', 'l', 'm', 'z', 'j', 'q', 't', 'p', 'u', 'y']

shot_direction = ['1', '2', '3']

serve_direction = ['4', '5', '6']

return_depth = ['7', '8', '9']

shank = ['!']

position_information = ['=', '+', '-']

terminal_symbols = ['*', '#', '@']

error_location = ['n', 'w', 'd', 'x']

def validate_shots(first_serve, second_serve, verbose=False) -> bool:
    """Validates the shot sequences for a point. Cleans invalid characters and allows terminal symbols in first shot.
    Handles double faults where second serve may not have a terminal symbol."""
    # Check if both serves are missing
    if pd.isna(first_serve) and pd.isna(second_serve):
        if verbose:
            print("FAIL: Both serves missing")
        return False
    
    # Check if only one serve is present
    if pd.isna(first_serve) and not pd.isna(second_serve):
        if verbose:
            print("FAIL: Second serve without first serve")
        return False
    
    shots = make_shots(first_serve, second_serve)
    
    if not shots:
        if verbose:
            print("FAIL: No shots extracted")
        return False
    
    # Valid characters set
    valid_chars = set(ground_strokes + shot_direction + serve_direction + return_depth + position_information + error_location + shank + terminal_symbols)
    
    # Clean shots: remove invalid characters and filter out empty shots
    cleaned_shots = []
    for shot in shots:
        cleaned_shot = ''.join(char for char in shot if char in valid_chars)
        if cleaned_shot:  # Only keep non-empty shots
            cleaned_shots.append(cleaned_shot)
    
    if not cleaned_shots:
        if verbose:
            print("FAIL: No shots remain after cleaning invalid characters")
        return False
    
    # First shot must start with a serve direction (4, 5, or 6)
    if cleaned_shots[0][0] not in serve_direction:
        if verbose:
            print(f"FAIL: First shot '{cleaned_shots[0]}' doesn't start with serve direction")
        return False
    
    # Check for double fault case: second serve only with error location, no terminal symbol
    # Example: first_serve="4*", second_serve="5d" (2 chars)
    is_double_fault = (
        len(cleaned_shots) == 1 and 
        len(cleaned_shots[0]) == 2 and 
        cleaned_shots[0][0] in serve_direction and 
        cleaned_shots[0][1] in error_location
    )
    
    # Last shot must contain a terminal symbol (unless it's a double fault)
    if not is_double_fault:
        if not any(char in terminal_symbols for char in cleaned_shots[-1]):
            if verbose:
                print(f"FAIL: Last shot '{cleaned_shots[-1]}' has no terminal symbol")
            return False
    
    # Only non-last shots should not contain terminal symbols in the middle
    for shot in cleaned_shots[:-1]:
        if any(char in terminal_symbols for char in shot):
            if verbose:
                print(f"FAIL: Non-terminal shot '{shot}' contains terminal symbol")
            return False
    
    # Last shot validation: check terminal symbol rules (skip for double fault)
    if not is_double_fault:
        last_shot = cleaned_shots[-1]
        if last_shot[-1] in terminal_symbols:
            if last_shot[-1] == '*':
                # Winner - should not have error location before terminal symbol
                if len(last_shot) > 1 and last_shot[-2] in error_location:
                    if verbose:
                        print(f"FAIL: Winner '{last_shot}' has error location before *")
                    return False
            elif last_shot[-1] in ['#', '@']:
                # Error - must have error location before terminal symbol
                if len(last_shot) <= 1 or last_shot[-2] not in error_location:
                    if verbose:
                        print(f"FAIL: Error shot '{last_shot}' missing error location before terminal")
                    return False
    
    return True

def run_shot_validation(verbose=False, diagnostic=False):
    """Runs the shot validation process on the dataset and prints out the results."""
    
    final_cleaned_data = pd.read_csv(FINAL_CLEANED_DATA)
    rows_before = len(final_cleaned_data)
    
    if verbose:
        print(f"Rows before validation: {rows_before}")
    
    if diagnostic:
        # Run validation with detailed feedback for first 100 invalid rows
        print("\n=== DIAGNOSTIC MODE: Analyzing first 100 invalid rows ===\n")
        invalid_count = 0
        for idx, row in final_cleaned_data.iterrows():
            result = validate_shots(row['first_serve'], row['second_serve'], verbose=True)
            if not result:
                invalid_count += 1
                if invalid_count >= 100:
                    break
        print(f"\nStopped after {invalid_count} invalid rows to analyze patterns")
        return
    
    final_cleaned_data['shot_validation'] = final_cleaned_data.apply(lambda row: validate_shots(row['first_serve'], row['second_serve']), axis=1)
    
    # Count failures by type
    validation_results = final_cleaned_data['shot_validation']
    rows_valid = validation_results.sum()
    rows_invalid = len(validation_results) - rows_valid
    
    final_cleaned_data = final_cleaned_data[final_cleaned_data['shot_validation'] == True]
    rows_after = len(final_cleaned_data)
    
    if verbose:
        print(f"Rows after validation: {rows_after}")
        print(f"Rows removed: {rows_before - rows_after} ({100 * (rows_before - rows_after) / rows_before:.1f}%)")
        print(f"Rows kept: {rows_after} ({100 * rows_after / rows_before:.1f}%)")
    
    final_cleaned_data.to_csv(FINAL_CLEANED_VALIDATED_DATA, index=False)
    
    if verbose:
        print(f"Cleaned data saved to {FINAL_CLEANED_VALIDATED_DATA}")
    
if __name__ == "__main__":
    run_shot_validation(verbose=True, diagnostic=False)