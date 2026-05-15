"""This is the model for the Live Point winning probability (LPWP) model.
The dataset used is live_point_win_probability_dataset.csv, which is created in the lpwp_feature_generation_03.py"""

"""We will use a Random Forest for this task, as it is a binary classification problem (win or lose)."""

"""We will be using multiple test train splits to evaluate the model, and we will be using the average accuracy, precision, recall, and F1 score as our evaluation metrics. We wil use cross validation to evaluatett the model."""

import pandas as pd
from config import LIVE_POINT_WIN_PROB_DATASET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
import optuna
import trackio
from tqdm import tqdm


random_seed = 42

print("Running Random Forest for Live Point Winning Probability...")
print("Loading dataset...")
X = pd.read_csv(LIVE_POINT_WIN_PROB_DATASET)
print(f"Dataset loaded: {len(X)} rows")
print("Sampling 10% of data for faster experimentation...")
X = X.sample(frac=0.1, random_state=random_seed)  # Use only 10% of the data for faster experimentation
print(f"Sampled data: {len(X)} rows")
print("Filling missing values...")
X.fillna(-1, inplace=True)

y = X["point_winner"]
X = X.drop(columns=["point_winner"])

# Determine appropriate n_splits based on minimum class count
min_class_count = y.value_counts().min()
n_splits = max(2, min(5, min_class_count))  # Use at most 5 splits, but not more than smallest class
print(f"Class distribution in point_winner: {y.value_counts().to_dict()}")
print(f"Minimum class count: {min_class_count}, using n_splits={n_splits}")

trackio.init(
    project="match-point-simulator",
    name="random_forest_lpwp"
)

def objective(trial):
    max_depth = trial.suggest_int("max_depth", 10, 30)
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    min_samples_split = trial.suggest_int("min_samples_split", 5, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=4, random_state=random_seed)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    mean_accuracy = scores.mean()
    
    trackio.log({
        "trail_number": trial.number,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "cv_accuracy": float(mean_accuracy)
        })
    return mean_accuracy

print("Starting Optuna optimization...")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_depth = study.best_params["max_depth"]
best_accuracy = study.best_value

print("Optuna optimization completed.")
print(f"Best Max Depth: {best_depth}")
print(f"Best CV Accuracy: {best_accuracy * 100:.2f}%")

trackio.log({
    "final_best_max_depth": best_depth, 
    "final_best_accuracy": float(best_accuracy)
})
trackio.finish()


    
    