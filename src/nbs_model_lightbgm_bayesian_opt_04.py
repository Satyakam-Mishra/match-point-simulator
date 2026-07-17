import pandas as pd
import xgboost as xgb
import optuna
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from config import NEXT_BEST_SHOT_DATASET

# 1. LOAD AND CLEAN
print("Loading dataset...")
df = pd.read_csv(NEXT_BEST_SHOT_DATASET)
features = [
    "pl_0_hand","pl_1_hand","best_of","gender","point_number","svr","surface_Clay","surface_Grass","surface_Hard","player_0_point","player_1_point","is_tiebreaker","point_diff","game_diff","set_diff","is_deuce","is_breakpoint","first_serve_in","shot_number","prev_shot_type","prev_shot_direction","prev_shot_depth","prev_shot_shank_info","prev_shot_position_info"

]
targets = ["shot_type" ,"shot_direction" ,"shot_depth"]

df = df.dropna(subset=targets)
df.fillna(-1, inplace=True)

df = df.sample(frac=0.1, random_state=42, replace=False, ignore_index=True) # Use only 10% of the data for faster experimentation

X = df[features].copy()
y = df[targets].copy()

label_encoders = {}
y_list = []
for col in targets:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y[col].astype(str))
    y_list.append(y_encoded)
    label_encoders[col] = le

# Convert to numpy array for multi-output classification
y = np.column_stack(y_list)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split train into train/validation for optimization
X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Calculate class weights for direction (y_train[:, 1]) to handle class imbalance
# This helps balance the heavily skewed direction distribution
direction_classes = np.unique(y_train_opt[:, 1])
direction_weights = compute_class_weight('balanced', classes=direction_classes, y=y_train_opt[:, 1])
direction_weight_dict = dict(zip(direction_classes, direction_weights))

# Create sample weights for all samples based on direction class weights
train_sample_weights = np.array([direction_weight_dict[label] for label in y_train_opt[:, 1]])

# 3. Custom accuracy function for multi-output classification
def multi_output_accuracy_score(y_true, y_pred):
    """Average accuracy across all output columns."""
    accuracies = []
    for i in range(y_true.shape[1]):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies.append(acc)
    return np.mean(accuracies)

# 4. Define the Optuna Objective Function
def objective(trial):
    # Define the hyperparameters to optimize
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
    }

    # Initialize the base XGBoost model with the suggested params
    base_model = xgb.XGBClassifier(**param, random_state=42, eval_metric='mlogloss', verbosity=0)
    
    # Wrap it to handle the 3 output columns simultaneously
    multi_target_model = MultiOutputClassifier(base_model)
    
    # Train on train set and evaluate on validation set with class weights
    try:
        multi_target_model.fit(X_train_opt, y_train_opt, sample_weight=train_sample_weights)
        y_pred = multi_target_model.predict(X_val_opt)
        score = multi_output_accuracy_score(y_val_opt, y_pred)
        return score
    except Exception as e:
        return -1.0  # Return low score if training fails

# 5. Run the Optimization
print("Starting Bayesian Optimization...")
study = optuna.create_study(direction='maximize') # We want to maximize accuracy
study.optimize(objective, n_trials=30) # Increase n_trials for better tuning

# 6. Output the results
print("\nBest Trial:")
trial = study.best_trial
print(f"  Value (Accuracy): {trial.value:.4f}")
print("  Best Hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")