import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving the model and encoder
from config import NEXT_BEST_SHOT_DATASET, LIGHTGBM_NBS_MODEL, LABEL_ENCODER_NBS

# 1. LOAD DATA
print("Loading full dataset...")
df = pd.read_csv(NEXT_BEST_SHOT_DATASET)

# Pre-processing: Filter rare classes as we discussed
# (Adjust the threshold '100' based on your previous findings)

X = df.drop(columns=["y_string"])
y_raw = df["y_string"]

# Fill missing values exactly as you did during tuning
X.fillna(-1, inplace=True)

# 2. ENCODE TARGET
# We save the LabelEncoder so we can decode predictions in the simulator
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 3. TRAIN-TEST SPLIT (90% train, 10% test)
print("Splitting data: 90% train, 10% test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}\n")

# 4. YOUR BEST HYPERPARAMETERS
# Replace the values below with the exact output from your Optuna study
best_params = {
    'objective': 'multiclass',
    'metric': 'multi_error',
    'verbosity': -1,
    'num_class': len(le.classes_),
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'random_state': 42,
    
    # --- Paste your Optuna results here ---
    'n_estimators': 200,        
    'learning_rate': 0.09322762674116546,      
    'num_leaves': 32,         
    'max_depth': 5,           
    'feature_fraction': 0.7042239991443616,  
    'min_data_in_leaf': 99,    
    # --------------------------------------
}

# 5. TRAIN FINAL MODEL
print(f"Training final LightGBM model on {len(X_train)} rows...")
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train)

# 6. EVALUATE MODEL
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Training accuracy
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Test accuracy
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\n" + "-"*50)
print("Test Set Classification Report:")
print("-"*50)
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# 7. SAVE FOR DEPLOYMENT
# Saving both the model and the encoder is vital for your simulator
joblib.dump(model, LIGHTGBM_NBS_MODEL)
joblib.dump(le, LABEL_ENCODER_NBS)

print("Model and Encoder saved successfully!")

# Best Trial Accuracy: 31.38%
# Best Params: {'learning_rate': 0.09322762674116546, 'num_leaves': 32, 'max_depth': 5, 'min_data_in_leaf': 99, 'feature_fraction': 0.7042239991443616}