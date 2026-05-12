"""This is the model for the Live Point winning probability (LPWP) model.
The dataset used is live_point_win_probability_dataset.csv, which is created in the lpwp_feature_generation_03.py"""

# importing the model with bayesian optimization, which is in lpwp_model_bayesian_opt.py, and we will be using the best hyperparameters found in that model to train our final model and evaluate it on the test set. We will also be using the same evaluation metrics as before (accuracy, precision, recall, F1 score) to evaluate our final model.

import pandas as pd
from config import LIVE_POINT_WIN_PROB_DATASET, RANDOM_FOREST_LPWP_MODEL, FEATURE_COLUMNS_LPWP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import trackio
import pickle

random_seed = 42

print("Running Random Forest for Live Point Winning Probability...")
print("Loading dataset...")
X = pd.read_csv(LIVE_POINT_WIN_PROB_DATASET)
print(f"Dataset loaded: {len(X)} rows")
print("Filling missing values...")
X.fillna(-1, inplace=True)

y = X["point_winner"]
X = X.drop(columns=["point_winner"])

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
print("Training Random Forest with best hyperparameters from Optuna...")
best_max_depth = 18  # This should be the best max_depth found from the Optuna optimization
rf = RandomForestClassifier(n_estimators=50, max_depth=best_max_depth, n_jobs=4, random_state=random_seed)
rf.fit(X_train, y_train)
print("Evaluating model on test set...")
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# saving the model and feature columns through pickle
with open(RANDOM_FOREST_LPWP_MODEL, "wb") as f:
    pickle.dump(rf, f)

with open(FEATURE_COLUMNS_LPWP, "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print(f"Model saved to {RANDOM_FOREST_LPWP_MODEL}")
print(f"Feature columns saved to {FEATURE_COLUMNS_LPWP}")