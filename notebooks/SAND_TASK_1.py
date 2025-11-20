# Process audio
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
model_name = "facebook/wav2vec2-base"  # You can also try "facebook/wav2vec2-large" for better performance
processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)

def extract_features(audio_path):
    audio, sr = sf.read(audio_path)
    inputs = processor(
            audio, 
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
    return inputs

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check on mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # check on mac
# 
# def extract_features(batch, wav2vec_model):
#     """Extract features using frozen Wav2Vec 2.0 model"""
#     with torch.no_grad():
#         outputs = wav2vec_model(batch['input_values'].to(device))
#         features = outputs.last_hidden_state
#     return features

# Load a sample audio file from your dataset
sample = metadata.iloc[0]
audio_file = sample['audio_path']

wav, sr = sf.read(audio_file)
print(f"Original audio shape: {wav.shape}")
wav = np.mean(wav, axis=1) if wav.ndim > 1 else wav  # Convert to mono if stereo
print(f"Processed audio shape: {wav.shape}")

import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
from tqdm import tqdm
from scipy.signal import resample


# Load pre-trained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
wav2vec_model = wav2vec_model.to(device)
wav2vec_model.eval()

print("="*70)
print("INFO: Wav2Vec2 Audio Processing")
print("="*70)
print("Wav2Vec2 expects 16kHz audio sampling rate")
print("Your audio files are 8kHz - they will be upsampled to 16kHz")
print("This is normal and works well for speech/voice tasks")
print("="*70 + "\n")

def extract_features_from_file(audio_path, processor, wav2vec_model, device):
    """
    Extract Wav2Vec2 features from a single audio file.
    Automatically handles 8kHz → 16kHz upsampling.
    """
    try:
        # Load audio file
        audio, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # IMPORTANT: Handle 8kHz audio by upsampling to 16kHz
        target_sr = 16000
        
        if sr != target_sr:
            # Resample from source sr to 16kHz
            num_samples = int(len(audio) * target_sr / sr)
            audio = resample(audio, num_samples)
            # print(f"ℹ️  Resampled {audio_path.split('/')[-1]}: {sr}Hz → {target_sr}Hz")
        
        # Normalize audio to [-1, 1]
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Process audio with Wav2Vec2 processor
        inputs = processor(
            audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True
        )

        # Extract features using the Wav2Vec2 model
        with torch.no_grad():
            outputs = wav2vec_model(inputs.input_values.to(device))
            # Mean pooling over time dimension
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return features.squeeze(0)  # Return as a 1D NumPy array
        
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None

def extract_features_from_files(file_list, processor, wav2vec_model, device):
    """Extract features for all files"""
    features_list = []
    for file_path in tqdm(file_list, desc="Extracting features"):
        features = extract_features_from_file(file_path, processor, wav2vec_model, device)
        if features is not None:
            features_list.append(features)
    return features_list

# =====================================================================
# VERIFICATION: Check if all files for each user are present
# =====================================================================
print("Verifying data integrity...")

# Group audio files by user
grouped_by_user = metadata.groupby('ID')

# Check file counts per user
file_counts = metadata.groupby('ID').size()
expected_files = 8  # 5 phonation + 3 rhythm

print(f"\nFile counts per user:")
print(f"Expected: {expected_files} files per user")
print(f"Actual range: {file_counts.min()} - {file_counts.max()} files")

users_with_all_files = (file_counts == expected_files).sum()
users_total = len(file_counts)
print(f"Users with all {expected_files} files: {users_with_all_files}/{users_total}")

if users_with_all_files < users_total:
    print("\n⚠️  WARNING: Some users have missing files!")
    missing_files_users = file_counts[file_counts != expected_files]
    print(missing_files_users)

# =====================================================================
# LATE FUSION: Group features by Sample (8 files per sample)
# =====================================================================
print("\n" + "="*70)
print("Building dataset with Late Fusion strategy...")
print("="*70)

# Group audio files by user (each user has 8 files: 5 phonation + 3 rhythm)
# IMPORTANT: Use sorted() to ensure consistent ordering
grouped_by_user = metadata.groupby('ID', sort=True)

X_fused = []  # Features after fusion (one vector per user)
y_fused = []  # Labels (one per user)
user_ids = []
skipped_users = []

for user_id, group in tqdm(grouped_by_user, desc="Processing users"):
    # IMPORTANT: Sort files to ensure consistent ordering
    audio_files = sorted(group['audio_path'].tolist())
    
    # Check if user has all expected files
    if len(audio_files) != expected_files:
        skipped_users.append((user_id, len(audio_files)))
        continue
    
    # Extract features for each file
    user_features = []
    for audio_file in audio_files:
        features = extract_features_from_file(audio_file, processor, wav2vec_model, device)
        if features is not None:
            user_features.append(features)
    
    # Only use if all files processed successfully
    if len(user_features) == expected_files:
        # Late Fusion: Average all 8 feature vectors
        fused_features = np.mean(user_features, axis=0)
        
        X_fused.append(fused_features)
        # Get label from first row (same for all rows of same user)
        label = group['Class'].iloc[0]
        y_fused.append(label)
        user_ids.append(user_id)

X = np.array(X_fused)
y = np.array(y_fused)

print(f"\n{'='*70}")
print(f"✓ Processed {len(X)} users successfully")
if skipped_users:
    print(f"⚠️  Skipped {len(skipped_users)} users with incomplete data:")
    for uid, count in skipped_users:
        print(f"   - User {uid}: {count}/{expected_files} files")
print(f"✓ Feature dimensionality: {X.shape[1]}")
print(f"✓ Class distribution:\n{pd.Series(y).value_counts().sort_index()}")
print(f"{'='*70}")

# =====================================================================
# DATA PREPARATION: Split with proper label encoding (Wav2Vec2 features)
# =====================================================================

# Load training and validation IDs from Excel (reuse if already loaded)
if 'train_ids' not in locals() or 'test_ids' not in locals():
    train_sheet = pd.read_excel(METADATA_PATH, sheet_name='Training Baseline - Task 1')
    test_sheet = pd.read_excel(METADATA_PATH, sheet_name='Validation Baseline - Task 1')
    train_ids = set(train_sheet['ID'].tolist())
    test_ids = set(test_sheet['ID'].tolist())

# Encode labels starting from 0
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split based on user IDs from Excel
X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []

for idx, user_id in enumerate(user_ids):
    if user_id in train_ids:
        X_train_list.append(X[idx])
        y_train_list.append(y_encoded[idx])
    elif user_id in test_ids:
        X_test_list.append(X[idx])
        y_test_list.append(y_encoded[idx])

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)
X_test = np.array(X_test_list)
y_test = np.array(y_test_list)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Feature dimension: {X_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nTest set class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# =====================================================================
# TASK 2: Hyperparameter Tuning with Late Fusion Features
# =====================================================================

# Create pipeline WITHOUT scaler (Wav2Vec2 already normalized)
svm_pipeline_wav2vec = Pipeline([
    ('svm', SVC(random_state=42, probability=True))
])

# Define parameter grid for SVM with Wav2Vec2 features
svm_param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': ['scale', 'auto']
}

print("Performing Grid Search for SVM with Wav2Vec2 + Late Fusion...")
svm_grid_search = GridSearchCV(
    svm_pipeline_wav2vec,
    svm_param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

svm_grid_search.fit(X_train, y_train)

print(f"\nBest SVM parameters: {svm_grid_search.best_params_}")
print(f"Best SVM CV F1-Weighted: {svm_grid_search.best_score_:.4f}")

# Get predictions
svm_pred = svm_grid_search.predict(X_test)
svm_score = svm_grid_search.score(X_test, y_test)

print(f"\nTest Set Performance:")
print(f"Accuracy: {svm_score:.4f}")
print(f"F1-Weighted: {f1_score(y_test, svm_pred, average='weighted'):.4f}")
print(f"F1-Macro: {f1_score(y_test, svm_pred, average='macro'):.4f}")

# =====================================================================
# ADDITIONAL MODELS: Logistic Regression & XGBoost with Wav2Vec2
# =====================================================================

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Logistic Regression
print("\n" + "="*70)
print("Training Logistic Regression with Wav2Vec2...")
print("="*70)

lr_pipeline_wav2vec = Pipeline([
    ('lr', LogisticRegression(random_state=42, max_iter=1000))
])

lr_param_grid = {
    'lr__C': [0.01, 0.1, 1, 10, 100],
    'lr__penalty': ['l2'],
    'lr__solver': ['lbfgs', 'liblinear']
}

lr_grid_search = GridSearchCV(
    lr_pipeline_wav2vec,
    lr_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

lr_grid_search.fit(X_train, y_train)
lr_pred = lr_grid_search.predict(X_test)
lr_score = lr_grid_search.score(X_test, y_test)

print(f"\nBest LR parameters: {lr_grid_search.best_params_}")
print(f"Test F1-Macro: {f1_score(y_test, lr_pred, average='macro'):.4f}")

# XGBoost
print("\n" + "="*70)
print("Training XGBoost with Wav2Vec2...")
print("="*70)

xgb_pipeline_wav2vec = Pipeline([
    ('xgb', xgb.XGBClassifier(random_state=42, n_jobs=-1))
])

xgb_param_grid = {
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__n_estimators': [100, 200]
}

xgb_grid_search = GridSearchCV(
    xgb_pipeline_wav2vec,
    xgb_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

xgb_grid_search.fit(X_train, y_train)
xgb_pred = xgb_grid_search.predict(X_test)
xgb_score = xgb_grid_search.score(X_test, y_test)

print(f"\nBest XGBoost parameters: {xgb_grid_search.best_params_}")
print(f"Test F1-Macro: {f1_score(y_test, xgb_pred, average='macro'):.4f}")

# =====================================================================
# MODEL COMPARISON
# =====================================================================

print("\n" + "="*70)
print("FINAL MODEL COMPARISON (Wav2Vec2 + Late Fusion)")
print("="*70)

comparison_data = {
    'Model': ['SVM', 'Logistic Regression', 'XGBoost'],
    'Accuracy': [svm_score, lr_score, xgb_score],
    'F1-Weighted': [
        f1_score(y_test, svm_pred, average='weighted'),
        f1_score(y_test, lr_pred, average='weighted'),
        f1_score(y_test, xgb_pred, average='weighted')
    ],
    'F1-Macro': [
        f1_score(y_test, svm_pred, average='macro'),
        f1_score(y_test, lr_pred, average='macro'),
        f1_score(y_test, xgb_pred, average='macro')
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

best_model_idx = comparison_df['F1-Macro'].idxmax()
print(f"\n✓ Best Model: {comparison_df.loc[best_model_idx, 'Model']} (F1-Macro: {comparison_df.loc[best_model_idx, 'F1-Macro']:.4f})")

# =====================================================================
# LATE FUSION ANALYSIS & VISUALIZATION
# =====================================================================

print("\n" + "="*70)
print("RESULTS")
print("="*70)

# Confusion matrix for best model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

best_model_predictions = {
    0: svm_pred,
    1: lr_pred,
    2: xgb_pred
}

best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_pred = best_model_predictions[best_model_idx]

print(f"\n✓ Classification Report ({best_model_name}):")
print(classification_report(y_test, best_pred, target_names=[str(cls) for cls in label_encoder.classes_]))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name} (Wav2Vec2 + Late Fusion)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()