# Doing necessary imports
import pandas as pd
import numpy as np
from config import FINAL_CLEANED_DATA

def analyse_dataframe(df):
    print("Dataframe Info:")
    print(df.info())
    print("\nDataframe Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # printing unique values for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nUnique values in {col}:")
        print(df[col].unique())
        
    # printing value counts for categorical columns
    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())
        
    # printing correlation matrix for numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print("\nCorrelation Matrix:")
    print(df[numerical_cols].corr())    
    

def create_features(final_cleaned_data, verbose = False):
    
    
    
    pass
    
def run_feature_creation(verbose = False):
    if verbose:
        print("Starting feature creation...")
        
    final_cleaned_data = pd.read_csv(FINAL_CLEANED_DATA)
    
    create_features(final_cleaned_data, verbose)
    
    

if __name__ == "__main__":
    run_feature_creation()