import os

# Raw data file paths
CHARTING_M_MATCHES = os.path.join("data", "charting-m-matches.csv")
CHARTING_M_POINTS_2010S = os.path.join("data", "charting-m-points-2010s.csv")
CHARTING_M_POINTS_2020S = os.path.join("data", "charting-m-points-2020s.csv")
CHARTING_M_POINTS_TO_2009 = os.path.join("data", "charting-m-points-to-2009.csv")
CHARTING_W_MATCHES = os.path.join("data", "charting-w-matches.csv")
CHARTING_W_POINTS_2010S = os.path.join("data", "charting-w-points-2010s.csv")
CHARTING_W_POINTS_2020S = os.path.join("data", "charting-w-points-2020s.csv")
CHARTING_W_POINTS_TO_2009 = os.path.join("data", "charting-w-points-to-2009.csv")
GENERAL_ANALYSIS_PY = os.path.join("notebooks", "general_analysis.py")

# Cleaned data file paths (saved to dataset folder)
CLEANED_METADATA = os.path.join("dataset", "cleaned_metadata.csv")
CLEANED_POINTS = os.path.join("dataset", "cleaned_points.csv")
FINAL_CLEANED_DATA = os.path.join("dataset", "final_cleaned_data.csv")
FEATURED_DATA = os.path.join("dataset", "featured_data.csv")
FINAL_CLEANED_VALIDATED_DATA = os.path.join("dataset", "final_cleaned_validated_data.csv")
LIVE_POINT_WIN_PROB_DATASET = os.path.join("dataset", "live_point_win_probability_dataset.csv")
NEXT_BEST_SHOT_DATASET = os.path.join("dataset", "next_best_shot_dataset.csv")
BEST_RETURN_BASED_ON_SERVE_DATASET = os.path.join("dataset", "best_return_based_on_serve_dataset.csv")
NEXT_BEST_SHOT_DATASET = os.path.join("dataset", "next_best_shot_dataset.csv")

#model paths
RANDOM_FOREST_LPWP_MODEL = os.path.join("models", "random_forest_lpwp_model.pkl")
FEATURE_COLUMNS_LPWP = os.path.join("models", "feature_columns_lpwp.pkl")
NBS_MODEL = os.path.join("models", "nbs_lightgbm_model.pkl")
FEATURE_COLUMNS_NBS = os.path.join("models", "feature_columns_nbs.pkl")
LABEL_ENCODER_NBS = os.path.join("models", "nbs_label_encoder.pkl")
XGBOOST_BRBS_MODEL = os.path.join("models", "brbs_xgboost_model.pkl")
FEATURE_COLUMNS_BRBS = os.path.join("models", "feature_columns_brbs.pkl")
LABEL_ENCODERS_BRBS = os.path.join("models", "brbs_label_encoders.pkl")
