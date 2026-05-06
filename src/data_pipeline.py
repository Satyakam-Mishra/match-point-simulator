# Imports
import pandas as pd
from typing import Optional
from pydantic import BaseModel, ValidationError
from config import (
    CHARTING_M_MATCHES,
    CHARTING_M_POINTS_2010S,
    CHARTING_M_POINTS_2020S,
    CHARTING_M_POINTS_TO_2009,
    CHARTING_W_MATCHES,
    CHARTING_W_POINTS_2010S,
    CHARTING_W_POINTS_2020S,
    CHARTING_W_POINTS_TO_2009,
) 

# This function checks for missing values in the dataframe and drops rows with nulls. It also prints the number of missing values if verbose is True.
def check_missing_values(df, name, verbose=False):
    """Check for null/missing values in dataframe and drop rows with nulls."""
    missing = df.isnull().sum()
    if missing.any():
        if verbose:
            print(f"Missing values in {name}:")
            print(missing[missing > 0])
        df = df.dropna()
        if verbose:
            print(f"Dropped rows with missing values. New shape: {df.shape}")
    return df


def check_missing_values_selective(df, name, critical_cols, verbose=False):
    """Only check critical columns for missing values, ignore optional ones."""
    missing = df[critical_cols].isnull().sum()
    if missing.any():
        if verbose:
            print(f"Missing values in {name} (critical columns):")
            print(missing[missing > 0])
        df = df.dropna(subset=critical_cols)
        if verbose:
            print(f"Dropped rows with missing critical values. New shape: {df.shape}")
    return df

# This function validates the dataframe rows against a pydantic schema and returns a list of invalid rows. If verbose is True, it also prints the number of invalid rows and shows the first 5 errors.
def validate_dataframe(df, schema, verbose=False):
    """Validate dataframe rows against pydantic schema."""
    invalid_rows = []
    for index, row in df.iterrows():
        try:
            # Convert NaN to None for proper Pydantic validation
            row_dict = row.to_dict()
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
            schema(**row_dict)
        except ValidationError as e:
            invalid_rows.append((index, str(e)))
    
    if invalid_rows and verbose:
        print(f"Found {len(invalid_rows)} invalid rows")
        for idx, error in invalid_rows[:5]:  # Show first 5
            print(f"  Row {idx}: {error}")
    
    return invalid_rows


class MatchMetadata(BaseModel):
    """Pydantic model for match metadata validation."""
    match_id: str
    player_1: str
    player_2: str
    pl_1_hand: str
    pl_2_hand: str
    surface: str
    best_of: Optional[int] = None
    
class PointData(BaseModel):
    """Pydantic model for point data validation."""
    match_id: str
    point_number: Optional[int] = None
    set1: Optional[int] = None
    set2: Optional[int] = None
    game1: Optional[int] = None
    game2: Optional[int] = None
    points: Optional[str] = None
    game_number: Optional[int] = None
    tiebreak_set: Optional[int] = None
    server: Optional[int] = None
    first_serve: Optional[str] = None
    second_serve: Optional[str] = None
    notes: Optional[str] = None
    point_winner: int


# The validation of the data. 
def validate_and_clean_metadata(verbose = False):
    if verbose:
        print("Starting validation...")
        print()
        
    """Men Metadata"""
        
    if verbose:
        print("cleaning charting-m-matches.csv")
    men_matches_metadata = pd.read_csv(CHARTING_M_MATCHES)
    #dropping the columns that are not needed for the analysis.
    men_matches_metadata.drop(columns = ["Date", "Tournament", "Time", "Court", "Round", "Umpire", "Charted by", "Final TB?"], inplace = True)
    #renaming the columns to be more descriptive.
    men_matches_metadata.rename(columns = {"match_id": "match_id", "Player 1": "player_1", "Player 2": "player_2", "Pl 1 hand": "pl_1_hand", "Pl 2 hand": "pl_2_hand", "Surface": "surface", "Best of": "best_of"}, inplace = True)
    # Check for missing values
    men_matches_metadata = check_missing_values(men_matches_metadata, "men_matches_metadata", verbose)
    
    """Women Metadata"""
    
    if verbose:
        print("cleaning charting-w-matches.csv")
    women_matches_metadata = pd.read_csv(CHARTING_W_MATCHES)
    #dropping the columns that are not needed for the analysis.
    women_matches_metadata.drop(columns = ["Date", "Tournament", "Time", "Court", "Round", "Umpire", "Charted by", "Final TB?"], inplace = True)
    #renaming the columns to be more descriptive.
    women_matches_metadata.rename(columns = {"match_id": "match_id", "Player 1": "player_1", "Player 2": "player_2", "Pl 1 hand": "pl_1_hand", "Pl 2 hand": "pl_2_hand", "Surface": "surface", "Best of": "best_of"}, inplace = True)
    # Check for missing values
    women_matches_metadata = check_missing_values(women_matches_metadata, "women_matches_metadata", verbose)
    
    #Validation the men_matches_metadata and women_matches_metadata dataframes to ensure that the match_id column is unique.
    if verbose:
        print("Validating the men_matches_metadata and women_matches_metadata dataframes...")
        print(f"Number of rows in men_matches_metadata: {men_matches_metadata.shape[0]}")
        print(f"Number of rows in women_matches_metadata: {women_matches_metadata.shape[0]}")
        print(f"Unique match_ids in men_matches_metadata: {men_matches_metadata['match_id'].nunique()}")
        print(f"Unique match_ids in women_matches_metadata: {women_matches_metadata['match_id'].nunique()}")
        
    if verbose:
        print("Removing duplicate match_ids from men_matches_metadata and women_matches_metadata dataframes...")
        
    men_matches_metadata.drop_duplicates(subset = ["match_id"], inplace = True)
    women_matches_metadata.drop_duplicates(subset = ["match_id"], inplace = True)
    
    # Validate dataframes against schema
    if verbose:
        print("Validating the men_matches_metadata dataframe against the MatchMetadata schema...")
    men_invalid = validate_dataframe(men_matches_metadata, MatchMetadata, verbose)
    
    if verbose:
        print("Validation of men_matches_metadata completed.")
    
    if verbose:
        print("Validating the women_matches_metadata dataframe against the MatchMetadata schema...")
    women_invalid = validate_dataframe(women_matches_metadata, MatchMetadata, verbose)
    
    if verbose:
        print("Validation of women_matches_metadata completed.")
        
def validate_and_clean_points(verbose = False):
    if verbose:
        print("Starting point validation and cleaning...")
        
    men_points_2010s = pd.read_csv(CHARTING_M_POINTS_2010S)
    men_points_2020s = pd.read_csv(CHARTING_M_POINTS_2020S)
    men_points_2009 = pd.read_csv(CHARTING_M_POINTS_TO_2009)
    
    women_points_2010s = pd.read_csv(CHARTING_W_POINTS_2010S)
    women_points_2020s = pd.read_csv(CHARTING_W_POINTS_2020S)
    women_points_2009 = pd.read_csv(CHARTING_W_POINTS_TO_2009)
    
    # Renaming the columns to be more descriptive.
    men_points_2010s.rename(columns = {"match_id": "match_id", "Pt": "point_number", "Set1": "set1", "Set2": "set2", "Gm1": "game1", "Gm2": "game2", "Pts": "points", "Gm#": "game_number", "TbSet": "tiebreak_set", "Srv": "server", "1st": "first_serve", "2nd": "second_serve", "Notes": "notes", "PtWinner": "point_winner"}, inplace = True)
    men_points_2020s.rename(columns = {"match_id": "match_id", "Pt": "point_number", "Set1": "set1", "Set2": "set2", "Gm1": "game1", "Gm2": "game2", "Pts": "points", "Gm#": "game_number", "TbSet": "tiebreak_set", "Srv": "server", "1st": "first_serve", "2nd": "second_serve", "Notes": "notes", "PtWinner": "point_winner"}, inplace = True)
    men_points_2009.rename(columns = {"match_id": "match_id", "Pt": "point_number", "Set1": "set1", "Set2": "set2", "Gm1": "game1", "Gm2": "game2", "Pts": "points", "Gm#": "game_number", "TbSet": "tiebreak_set", "Srv": "server", "1st": "first_serve", "2nd": "second_serve", "Notes": "notes", "PtWinner": "point_winner"}, inplace = True)
    
    """For Women dataframes"""
    
    women_points_2010s.rename(columns = {"match_id": "match_id", "Pt": "point_number", "Set1": "set1", "Set2": "set2", "Gm1": "game1", "Gm2": "game2", "Pts": "points", "Gm#": "game_number", "TbSet": "tiebreak_set", "Srv": "server", "1st": "first_serve", "2nd": "second_serve", "Notes": "notes", "PtWinner": "point_winner"}, inplace = True)
    women_points_2020s.rename(columns = {"match_id": "match_id", "Pt": "point_number", "Set1": "set1", "Set2": "set2", "Gm1": "game1", "Gm2": "game2", "Pts": "points", "Gm#": "game_number", "TbSet": "tiebreak_set", "Srv": "server", "1st": "first_serve", "2nd": "second_serve", "Notes": "notes", "PtWinner": "point_winner"}, inplace = True)
    women_points_2009.rename(columns = {"match_id": "match_id", "Pt": "point_number", "Set1": "set1", "Set2": "set2", "Gm1": "game1", "Gm2": "game2", "Pts": "points", "Gm#": "game_number", "TbSet": "tiebreak_set", "Srv": "server", "1st": "first_serve", "2nd": "second_serve", "Notes": "notes", "PtWinner": "point_winner"}, inplace = True)
    
    if verbose:
        print("Checking for missing values in points dataframes...")
        # Define critical columns that shouldn't have missing values
        critical_cols = ["match_id", "point_number", "set1", "set2", "game1", "game2", "points", "game_number", "point_winner"]
        men_points_2010s = check_missing_values_selective(men_points_2010s, "men_points_2010s", critical_cols, verbose)
        men_points_2020s = check_missing_values_selective(men_points_2020s, "men_points_2020s", critical_cols, verbose)
        men_points_2009 = check_missing_values_selective(men_points_2009, "men_points_2009", critical_cols, verbose)
        women_points_2010s = check_missing_values_selective(women_points_2010s, "women_points_2010s", critical_cols, verbose)
        women_points_2020s = check_missing_values_selective(women_points_2020s, "women_points_2020s", critical_cols, verbose)
        women_points_2009 = check_missing_values_selective(women_points_2009, "women_points_2009", critical_cols, verbose)
        
    # Now validate the points dataframes against the PointData schema.
    if verbose:
        print("Validating the men_points_2010s dataframe against the PointData schema...")
    men_points_2010s_invalid = validate_dataframe(men_points_2010s, PointData, verbose)
    if verbose:
        print("Validation of men_points_2010s completed.")
        
    if verbose:
        print("Validating the men_points_2020s dataframe against the PointData schema...")
    men_points_2020s_invalid = validate_dataframe(men_points_2020s, PointData, verbose)
    if verbose:
        print("Validation of men_points_2020s completed.")
        
    if verbose:
        print("Validating the men_points_2009 dataframe against the PointData schema...")
    men_points_2009_invalid = validate_dataframe(men_points_2009, PointData, verbose)
    if verbose:
        print("Validation of men_points_2009 completed.")
        
    if verbose:
        print("Validating the women_points_2010s dataframe against the PointData schema...")
    women_points_2010s_invalid = validate_dataframe(women_points_2010s, PointData, verbose)
    if verbose:
        print("Validation of women_points_2010s completed.")
        
    if verbose:
        print("Validating the women_points_2020s dataframe against the PointData schema...")
    women_points_2020s_invalid = validate_dataframe(women_points_2020s, PointData, verbose)
    if verbose:
        print("Validation of women_points_2020s completed.")
        
    if verbose:
        print("Validating the women_points_2009 dataframe against the PointData schema...")
    women_points_2009_invalid = validate_dataframe(women_points_2009, PointData, verbose)
    if verbose:
        print("Validation of women_points_2009 completed.")
        
    if verbose:
        print("Point validation and cleaning completed.")
        
    if verbose:
        print("Summary of valid rows in points dataframes:")
        print("men_points_2010s valid rows:", men_points_2010s.shape[0])
        print("men_points_2020s valid rows:", men_points_2020s.shape[0])
        print("men_points_2009 valid rows:", men_points_2009.shape[0])
        print("women_points_2010s valid rows:", women_points_2010s.shape[0])
        print("women_points_2020s valid rows:", women_points_2020s.shape[0])
        print("women_points_2009 valid rows:", women_points_2009.shape[0])

def run_full_pipeline(verbose = False):
    if verbose:
        print("Starting the pipeline......")
        print() 
    
    validate_and_clean_metadata(verbose)
    
    validate_and_clean_points(verbose)
    
if __name__ == "__main__":
    run_full_pipeline(verbose = True)