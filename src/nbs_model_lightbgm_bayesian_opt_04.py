import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from config import NEXT_BEST_SHOT_DATASET

# 1. LOAD AND CLEAN
print("Loading dataset...")
df = pd.read_csv(NEXT_BEST_SHOT_DATASET)

X = df.drop(columns=["y_string"]).sample(frac=0.1, random_state=42)
y_raw = df.loc[X.index, "y_string"]
X.fillna(-1, inplace=True)

# Encode strings to numbers for LightGBM
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 2. DEFINE OBJECTIVE
def objective(trial):
    param = {
        'objective': 'multiclass',
        'metric': 'multi_error', # multi_error is (1 - accuracy)
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_class': len(np.unique(y)),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2), # Higher LR for accuracy
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'n_estimators': 200,
        'random_state': 42,
        'n_jobs': -1
    }

    # Accuracy usually needs 5 folds to be reliable
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    clf = lgb.LGBMClassifier(**param)
    
    # We use 'accuracy' directly here
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

# 3. RUN OPTIMIZATION
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, show_progress_bar=True)

print(f"\nBest Trial Accuracy: {study.best_value * 100:.2f}%")
print("Best Params:", study.best_params)