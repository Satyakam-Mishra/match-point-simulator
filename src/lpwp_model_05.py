"""This is the model for the Live Point winning probability (LPWP) model.
The dataset used is live_point_win_probability_dataset.csv, which is created in the lpwp_feature_generation_03.py"""

# importing the model with bayesian optimization, which is in lpwp_model_bayesian_opt.py, and we will be using the best hyperparameters found in that model to train our final model and evaluate it on the test set. We will also be using the same evaluation metrics as before (accuracy, precision, recall, F1 score) to evaluate our final model.

import pandas as pd
from config import LIVE_POINT_WIN_PROB_DATASET, RANDOM_FOREST_LPWP_MODEL, FEATURE_COLUMNS_LPWP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
# import trackio
import pickle

random_seed = 42

print("Running Random Forest for Live Point Winning Probability...")
print("Loading dataset...")
X = pd.read_csv(LIVE_POINT_WIN_PROB_DATASET)# Use only 10% of the data for faster experimentation
print(f"Dataset loaded: {len(X)} rows")
print("Filling missing values...")
X.fillna(-1, inplace=True)

y = X["point_winner"]
X = X.drop(columns=["point_winner"])

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
print("Training Random Forest with improved hyperparameters...")
print("  - n_estimators=290 (increased from 250)")
print("  - max_depth=23 (increased from 16)")
print("  - min_samples_split=17 (decreased from 20)")
rf = RandomForestClassifier(
    n_estimators=290,
    max_depth=23,
    min_samples_split=17,
    min_samples_leaf=3,
    n_jobs=4,
    random_state=random_seed
)
rf.fit(X_train, y_train)
print("Evaluating model on test set...")
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

report = classification_report(y_test, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n{'='*60}")
print("MODEL PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1-Score (Macro):   {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")
print(f"{'='*60}")
print("\nDetailed Classification Report:")
print(report)

# Print top 10 most important features
print(f"\n{'='*60}")
print("TOP 10 MOST IMPORTANT FEATURES")
print(f"{'='*60}")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# saving the model and feature columns through pickle
with open(RANDOM_FOREST_LPWP_MODEL, "wb") as f:
    pickle.dump(rf, f)

with open(FEATURE_COLUMNS_LPWP, "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print(f"Model saved to {RANDOM_FOREST_LPWP_MODEL}")
print(f"Feature columns saved to {FEATURE_COLUMNS_LPWP}")

# Trial 16 finished with value: 0.7634751120112931 and parameters: {'max_depth': 23, 'n_estimators': 290, 'min_samples_split': 17, 'min_samples_leaf': 3}. Best is trial 16 with value: 0.7634751120112931.