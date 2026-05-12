"""This is the model for the Next Best Shot (NBS) model.
The dataset used is next_best_shot_dataset.csv, which is created in the nbs_feature_generation_03.py"""

"""We will use a Random Forest for this task"""

"""We will be using multiple test train splits to evaluate the model, and we will be using the average accuracy, precision, recall, and F1 score as our evaluation metrics. We wil use cross validation to evaluatett the model."""

import pandas as pd
from config import NEXT_BEST_SHOT_DATASET
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
import optuna
import trackio
from tqdm import tqdm


random_seed = 42

print("Running Random Forest for Next Best Shot...")
print("Loading dataset...")
X = pd.read_csv(NEXT_BEST_SHOT_DATASET)
print(f"Dataset loaded: {len(X)} rows")
print("Sampling 20% of data for faster experimentation...")
X = X.sample(frac=0.2, random_state=random_seed)  # Use only 20% of the data for faster experimentation
print(f"Sampled data: {len(X)} rows")
print("Filling missing values...")
X.fillna(-1, inplace=True)

y = X["point_winner"]
X = X.drop(columns=["point_winner"])

trackio.init(
    project="match-point-simulator",
    name="random_forest_nbs"
)

def objective(trial):
    max_depth = trial.suggest_int("max_depth", 10, 60)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=max_depth, n_jobs=4, random_state=random_seed)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    
    scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    mean_accuracy = scores.mean()
    
    trackio.log({
        "trail_number": trial.number,
        "max_depth": max_depth,
        "cv_accuracy": float(mean_accuracy)
        })
    return mean_accuracy

print("Starting Optuna optimization...")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

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