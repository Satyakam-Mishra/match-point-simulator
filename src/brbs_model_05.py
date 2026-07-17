"""This is the final model for BRBS. We use the parameters found from the Bayesian Optimization in brbs_XGBoost_Bayesian_Opt.py to train a final XGBoost model on the entire training set. We then evaluate it on the test set and save the model for future use."""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from config import XGBOOST_BRBS_MODEL, FEATURE_COLUMNS_BRBS, LABEL_ENCODERS_BRBS, BEST_RETURN_BASED_ON_SERVE_DATASET


def multi_output_accuracy_score(y_true, y_pred):
    """Average accuracy across all output columns."""
    accuracies = []
    for i in range(y_true.shape[1]):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies.append(acc)
    return np.mean(accuracies)

# 1. Load the featured dataset
dataset = pd.read_csv(BEST_RETURN_BASED_ON_SERVE_DATASET)

features = [
    'pl_0_hand', 'pl_1_hand', 'gender', 'svr', 'surface_Clay', 
    'surface_Grass', 'surface_Hard', 'player_0_point', 'player_1_point', 
    'is_tiebreaker', 'first_serve_in', 'serve_location', 
    'serve_shank_info'
]
targets = ['return_shot_type', 'return_direction', 'return_depth']

# Drop rows where target values are missing (do this BEFORE fillna)
dataset = dataset.dropna(subset=targets)

# Handle missing values in other features
dataset.fillna(-1, inplace=True)

# Extract features and targets
X = dataset[features]
y = dataset[targets]

# Encode target variables
label_encoders = {}
y_encoded_list = []
for col in targets:
    le = LabelEncoder()
    y_encoded = le.fit_transform(y[col].astype(str))
    y_encoded_list.append(y_encoded)
    label_encoders[col] = le

# Combine encoded targets into a single array
y_encoded = np.column_stack(y_encoded_list)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

param = {
        'n_estimators': 161,
        'max_depth': 9,
        'learning_rate': 0.012611484729323668,
        'subsample': 0.9870123963087756,
        'colsample_bytree': 0.9950864141325877,
        'min_child_weight': 4
    }

best_base_model = xgb.XGBClassifier(**param, random_state=42, eval_metric='mlogloss')
final_model = MultiOutputClassifier(best_base_model)

# Calculate class weights for direction (y_train[:, 1]) to handle class imbalance
# This helps balance the heavily skewed direction distribution
direction_classes = np.unique(y_train[:, 1])
direction_weights = compute_class_weight('balanced', classes=direction_classes, y=y_train[:, 1])
direction_weight_dict = dict(zip(direction_classes, direction_weights))

# Create sample weights for all samples based on direction class weights
sample_weights = np.array([direction_weight_dict[label] for label in y_train[:, 1]])

# Fit the model with sample weights to handle class imbalance
final_model.fit(X_train, y_train, sample_weight=sample_weights)

# Saving the model, feature columns, and label encoders for future use in the app
joblib.dump(final_model, XGBOOST_BRBS_MODEL)
joblib.dump(features, FEATURE_COLUMNS_BRBS)
joblib.dump(label_encoders, LABEL_ENCODERS_BRBS)

# Accuracy prediction and reporting
y_pred = final_model.predict(X_test)
accuracy = multi_output_accuracy_score(y_test, y_pred)
print(f"Final BRBS Model Accuracy (with balanced class weights): {accuracy:.4f}")
print("Evaluating model on test set...")
for i, target in enumerate(targets):
    f1_macro = f1_score(
        y_test[:, i],
        y_pred[:, i],
        average='macro',
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test[:, i],
        y_pred[:, i],
        average='weighted',
        zero_division=0
    )

    print(f"{target}")
    print(f"Macro F1    : {f1_macro:.4f}")
    print(f"Weighted F1 : {f1_weighted:.4f}")
    print()

# Print top 10 most important features
print(f"\n{'='*60}")
print("TOP 10 MOST IMPORTANT FEATURES")
print(f"{'='*60}")
for i, target in enumerate(targets):
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.estimators_[i].feature_importances_
    }).sort_values("importance", ascending=False)

    print("=" * 60)
    print(f"Top features for {target}")
    print(importance.head(10))


"""Best Trial:
  Value (Accuracy): 0.5699
  Best Hyperparameters:
    n_estimators: 161
    max_depth: 9
    learning_rate: 0.012611484729323668
    subsample: 0.9870123963087756
    colsample_bytree: 0.9950864141325877
    min_child_weight: 4"""
    
"""Best Trial:
  Value (Accuracy): 0.5450
  Best Hyperparameters:
    n_estimators: 214
    max_depth: 9
    learning_rate: 0.029039620115295265
    subsample: 0.9973276900124967
    colsample_bytree: 0.7690981087553185
    min_child_weight: 3"""
    
"""
Final BRBS Model Accuracy (with balanced class weights): 0.5571
Evaluating model on test set...
return_shot_type
Macro F1    : 0.1394
Weighted F1 : 0.6749

return_direction
Macro F1    : 0.4628
Weighted F1 : 0.4778

return_depth
Macro F1    : 0.2204
Weighted F1 : 0.3126


============================================================
TOP 10 MOST IMPORTANT FEATURES
============================================================
============================================================
Top features for return_shot_type
           feature  importance
10  first_serve_in    0.214834
8   player_1_point    0.157144
7   player_0_point    0.118584
11  serve_location    0.110733
1        pl_1_hand    0.097245
0        pl_0_hand    0.092905
3              svr    0.061861
2           gender    0.061839
4     surface_Clay    0.053528
5    surface_Grass    0.015267
============================================================
Top features for return_direction
           feature  importance
10  first_serve_in    0.262446
11  serve_location    0.180291
8   player_1_point    0.157874
7   player_0_point    0.154130
2           gender    0.056000
0        pl_0_hand    0.055460
1        pl_1_hand    0.054749
5    surface_Grass    0.019549
3              svr    0.019547
4     surface_Clay    0.018081
============================================================
Top features for return_depth
           feature  importance
2           gender    0.447876
10  first_serve_in    0.209073
5    surface_Grass    0.103920
6     surface_Hard    0.039811
4     surface_Clay    0.035445
11  serve_location    0.026804
0        pl_0_hand    0.026359
1        pl_1_hand    0.024374
7   player_0_point    0.023880
8   player_1_point    0.023310
"""