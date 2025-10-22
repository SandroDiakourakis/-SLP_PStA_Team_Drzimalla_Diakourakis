#!/usr/bin/env python3
"""
STUFE 3: Ensemble verschiedener Modelle
Kombiniert XGBoost, LightGBM und SVM
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import pickle
from datetime import datetime
import json

print("🎭 STUFE 3: ENSEMBLE LEARNING")
print("=" * 60)

# Load features
print("\n📂 Lade Features...")
feature_dir = Path("features/mfcc")  # Oder improved_mfcc

train_manifest = feature_dir / "train" / "manifest.csv"
test_manifest = feature_dir / "test" / "manifest.csv"

train_df = pd.read_csv(train_manifest)
test_df = pd.read_csv(test_manifest)

# Load training data
X_train = []
y_train = []
for idx, row in train_df.iterrows():
    X_train.append(np.load(row['feature_path']))
    y_train.append(row['label'])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Load test data
X_test = []
y_test = []
for idx, row in test_df.iterrows():
    X_test.append(np.load(row['feature_path']))
    y_test.append(row['label'])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"✓ Train: {X_train.shape}")
print(f"✓ Test:  {X_test.shape}")

# Define models
print("\n🤖 Erstelle Ensemble-Modelle...")

# Model 1: XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    max_depth=10,
    learning_rate=0.01,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42
)

# Model 2: LightGBM
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=5,
    max_depth=10,
    learning_rate=0.01,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42
)

# Model 3: SVM (mit RBF Kernel)
svm_model = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=42
)

# Create Voting Ensemble
print("\n🎯 Erstelle Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('svm', svm_model)
    ],
    voting='soft',  # Use probabilities
    weights=[2, 2, 1]  # XGBoost und LightGBM höher gewichtet
)

# Compute sample weights
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train_encoded)

# Train ensemble
print("\n🚀 Trainiere Ensemble...")
print("⏱️  Dies kann 10-15 Minuten dauern...\n")

ensemble.fit(X_train, y_train_encoded, sample_weight=sample_weights)

print("\n✅ Training abgeschlossen!")

# Evaluate
print("\n📊 Evaluiere Ensemble...")
y_pred = ensemble.predict(X_test)
y_pred_proba = ensemble.predict_proba(X_test)

# Compute metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report

accuracy = accuracy_score(y_test_encoded, y_pred)
macro_f1 = f1_score(y_test_encoded, y_pred, average='macro')

print("\n" + "=" * 60)
print("🎯 ENSEMBLE ERGEBNISSE:")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print("\nPer-Class Scores:")
print(classification_report(
    y_test_encoded,
    y_pred,
    target_names=label_encoder.classes_,
    digits=4
))

# Save results
output_dir = Path("runs") / f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir.mkdir(parents=True, exist_ok=True)

# Save ensemble
with open(output_dir / "ensemble_model.pkl", 'wb') as f:
    pickle.dump(ensemble, f)

# Save label encoder
with open(output_dir / "label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'macro_f1': float(macro_f1)
}
with open(output_dir / "metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n💾 Modell gespeichert in: {output_dir}")
#!/usr/bin/env python3
"""
STUFE 2: GridSearch für Hyperparameter-Tuning
Findet die besten Hyperparameter für XGBoost
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import json
from datetime import datetime
import pickle

print("🔍 STUFE 2: HYPERPARAMETER TUNING (GridSearch)")
print("=" * 60)

# Load features
print("\n📂 Lade Features...")
feature_dir = Path("features/mfcc")  # Oder improved_mfcc wenn Stufe 1 abgeschlossen

train_manifest = feature_dir / "train" / "manifest.csv"
train_df = pd.read_csv(train_manifest)

# Load features and labels
X_train = []
y_train = []

print(f"Lade {len(train_df)} Training Samples...")
for idx, row in train_df.iterrows():
    feature = np.load(row['feature_path'])
    X_train.append(feature)
    y_train.append(row['label'])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

print(f"\n✓ Features geladen: {X_train.shape}")
print(f"✓ Labels encodiert: {len(np.unique(y_train_encoded))} Klassen")

# Define parameter grid
print("\n🎯 Definiere Parameter-Grid...")
param_grid = {
    'max_depth': [8, 10, 12],
    'learning_rate': [0.01, 0.03, 0.05],
    'n_estimators': [500, 1000, 1500],
    'min_child_weight': [1, 3, 5],
    'gamma': [0.1, 0.2, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

print(f"Parameter-Kombinationen: {np.prod([len(v) for v in param_grid.values()])}")
print("⏱️  Geschätzte Dauer: 1-2 Stunden")

# Create base model
base_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train_encoded)),
    eval_metric='mlogloss',
    random_state=42,
    tree_method='hist',
    device='cpu'
)

# Compute sample weights for imbalance
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train_encoded)

# Setup GridSearch
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,  # Use all CPU cores
    verbose=2,
    return_train_score=True
)

# Run GridSearch
print("\n🚀 Starte GridSearch...")
print("Dies kann 1-2 Stunden dauern. Bitte warten...\n")

grid_search.fit(X_train, y_train_encoded, sample_weight=sample_weights)

# Results
print("\n" + "=" * 60)
print("✅ GRIDSEARCH ABGESCHLOSSEN!")
print("=" * 60)

print(f"\n🏆 Beste Parameter:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📊 Beste CV Macro F1: {grid_search.best_score_:.4f}")

# Save results
output_dir = Path("runs") / f"gridsearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
output_dir.mkdir(parents=True, exist_ok=True)

# Save best parameters
with open(output_dir / "best_params.json", 'w') as f:
    json.dump(grid_search.best_params_, f, indent=2)

# Save best model
best_model = grid_search.best_estimator_
with open(output_dir / "best_model.pkl", 'wb') as f:
    pickle.dump(best_model, f)

# Save all results
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv(output_dir / "cv_results.csv", index=False)

# Save top 10 configs
top10 = results_df.nlargest(10, 'mean_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
top10.to_csv(output_dir / "top10_configs.csv", index=False)

print(f"\n💾 Ergebnisse gespeichert in: {output_dir}")
print("\n📋 Top 5 Konfigurationen:")
print(top10.head())

print("\n🎯 Nächster Schritt:")
print(f"   Verwende diese Parameter für finales Training:")
print(f"   Bearbeite conf/improved_features.yaml und übernehme die besten Parameter")

