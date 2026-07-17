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
from config import NEXT_BEST_SHOT_DATASET, NBS_MODEL, LABEL_ENCODER_NBS, FEATURE_COLUMNS_NBS

def multi_output_accuracy_score(y_true, y_pred):
    """Average accuracy across all output columns."""
    accuracies = []
    for i in range(y_true.shape[1]):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        accuracies.append(acc)
    return np.mean(accuracies)
# 1. LOAD DATA
print("Loading full dataset...")
df = pd.read_csv(NEXT_BEST_SHOT_DATASET)

features = [
    "pl_0_hand","pl_1_hand","best_of","gender","point_number","svr","surface_Clay","surface_Grass","surface_Hard","player_0_point","player_1_point","is_tiebreaker","point_diff","game_diff","set_diff","is_deuce","is_breakpoint","first_serve_in","shot_number","prev_shot_type","prev_shot_direction","prev_shot_depth","prev_shot_shank_info","prev_shot_position_info"

]
targets = ["shot_type" ,"shot_direction" ,"shot_depth"]

df = df.dropna(subset=targets)

# Handle missing values in other features
df.fillna(-1, inplace=True)

X = df[features]
y = df[targets]

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
        'n_estimators': 202,
        'max_depth': 7,
        'learning_rate': 0.007486268010903502,
        'subsample': 0.950099259631429,
        'colsample_bytree': 0.8631257960047692,
        'min_child_weight': 6
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
joblib.dump(final_model, NBS_MODEL)
joblib.dump(features, FEATURE_COLUMNS_NBS)
joblib.dump(label_encoders, LABEL_ENCODER_NBS)

# Accuracy prediction and reporting
y_pred = final_model.predict(X_test)
accuracy = multi_output_accuracy_score(y_test, y_pred)
print(f"Final NBS Model Accuracy (with balanced class weights): {accuracy:.4f}")
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


"""
Best Trial:
  Value (Accuracy): 0.5200
  Best Hyperparameters:
    n_estimators: 202
    max_depth: 7
    learning_rate: 0.007486268010903502
    subsample: 0.950099259631429
    colsample_bytree: 0.8631257960047692
    min_child_weight: 6
"""

"""
Loading full dataset...
Final NBS Model Accuracy (with balanced class weights): 0.5225
Evaluating model on test set...
shot_type
Macro F1    : 0.2352
Weighted F1 : 0.6167

shot_direction
Macro F1    : 0.4296
Weighted F1 : 0.4435

shot_depth
Macro F1    : 0.3524
Weighted F1 : 0.3747


============================================================
TOP 10 MOST IMPORTANT FEATURES
============================================================
============================================================
Top features for shot_type
                    feature  importance
20      prev_shot_direction    0.410788
21          prev_shot_depth    0.139999
0                 pl_0_hand    0.093329
19           prev_shot_type    0.089893
1                 pl_1_hand    0.075289
18              shot_number    0.038019
23  prev_shot_position_info    0.031928
5                       svr    0.029362
3                    gender    0.026680
22     prev_shot_shank_info    0.014319
============================================================
Top features for shot_direction
                    feature  importance
19           prev_shot_type    0.241224
21          prev_shot_depth    0.180611
20      prev_shot_direction    0.169572
3                    gender    0.058811
23  prev_shot_position_info    0.044906
18              shot_number    0.041251
17           first_serve_in    0.038732
12               point_diff    0.031503
0                 pl_0_hand    0.024399
1                 pl_1_hand    0.022849
============================================================
Top features for shot_depth
                    feature  importance
20      prev_shot_direction    0.187369
21          prev_shot_depth    0.182964
19           prev_shot_type    0.163380
23  prev_shot_position_info    0.124533
18              shot_number    0.064282
3                    gender    0.043517
6              surface_Clay    0.030087
8              surface_Hard    0.019270
7             surface_Grass    0.015462
22     prev_shot_shank_info    0.014675
"""