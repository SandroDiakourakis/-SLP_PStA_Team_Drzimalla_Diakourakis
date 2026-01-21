# Task 2: ALS Progression Prediction - Complete Methodology & Results
## 5-Fold Cross-Validation - Baseline vs Sample Weights vs Ensemble

**Dokumentation des kompletten Vorgehen vom 20. Januar 2026**

---

## 1. Problem Definition & Setup

### 1.1 Objective
- **Task**: Predict ALSFRS-R score at last follow-up visit (disease progression)
- **Input Data**: 
  - 8 audio recordings per patient (5 vowels: A, E, I, O, U + 3 syllables: KA, PA, TA)
  - Metadata: Age, Sex, Months between assessments, Initial ALSFRS-R score
- **Target**: ALSFRS-R_end (progression outcome)
- **Classes**: 4 levels (1=Severe, 2=Moderate, 3=Mild, 4=No dysarthria) → Transformed to [0,1,2,3]
- **Evaluation Metric**: F1-Macro Score (critical for imbalanced classes)
- **Competitive Context**:
  - Challenge Baseline (PART Algorithm): **F1 = 0.5830**
  - Competition Leader (ISDS Team): **F1 = 0.5794**

### 1.2 Evaluation Methodology
- **CV Strategy**: 5-fold stratified cross-validation
- **Data**: Combined train + validation sets (132 samples total)
- **Reporting**: Mean F1 ± Std across 5 folds
- **Data Integrity**: Feature extraction performed in CV-loop (no leakage)

### 1.3 Computational Setup
- **Device**: GPU (for acceleration) with fallback to MPS/CPU
- **Random Seed**: 42 (global seed for reproducibility)
- **Packages**: PyTorch, Transformers, XGBoost, Scikit-learn
- **Estimated Runtime**: 
  - Baseline: ~22 minutes
  - Sample Weights: ~45 minutes
  - Ensemble: ~48 minutes
  - **Total: ~2 hours**

---

## 2. Data Exploration & Analysis

### 2.1 Dataset Characteristics
- **Total Samples**: 132
- **Audio Files**: 8 per patient × 132 = 1,056 total audio files
- **Metadata Features**: 4 (Age, Sex, Months, ALSFRS-R_start)
- **Target Distribution**:
  ```
  Class 0 (ALSFRS-R=1, Severe dysarthria):    18 samples (13.6%) ← Rare
  Class 1 (ALSFRS-R=2, Moderate):             27 samples (20.5%)
  Class 2 (ALSFRS-R=3, Mild):                 37 samples (28.0%)
  Class 3 (ALSFRS-R=4, No dysarthria):        50 samples (37.9%) ← Common
  ```

### 2.2 Class Imbalance Analysis
- **Imbalance Ratio**: 2.8:1 (Class 3 vs Class 0)
- **Key Challenge**: Rare class (0) has only 13.6% of data
- **Impact on F1-Macro**: Standard training tends to favor common classes
- **Mitigation Strategy**: Test with sample weights (Section 6)

### 2.3 Label Transformation
- **Original Labels**: ALSFRS-R levels [1, 2, 3, 4]
- **Transformed Labels**: [0, 1, 2, 3] (for XGBoost/sklearn compatibility)
- **Reverse Mapping**: Applied in final results for interpretation

---

## 3. Feature Extraction Strategy

### 3.1 Multi-Model Architecture Rationale
**Hypothesis**: Combining complementary pre-trained models captures diverse acoustic properties (speech quality, phoneme clarity, rhythm patterns)

#### Model 1: Wav2Vec2 (Self-Supervised Contrastive Learning)
- **Model**: `facebook/wav2vec2-large-960h`
- **Architecture**: Contrastive learning-based model trained on unlabeled audio
- **Strengths**: Captures speaker-independent acoustic patterns
- **Configuration**:
  - **Layers Extracted**: [6, 9, 12, 15, 18] (5 intermediate layers)
  - **Rationale**: Mix of low-level (layer 6) to high-level (layer 18) representations
  - **Pooling**: Mean temporal pooling (averages across time dimension)
  - **Layer Fusion**: Concatenation
  - **Hidden Size**: 1,024 per layer
  - **Feature Contribution**: 1,024 × 5 layers × 8 files = **40,960 features**

#### Model 2: HuBERT (Clustering-Based Masked Prediction)
- **Model**: `facebook/hubert-large-ls960-ft`
- **Architecture**: Clustering-based masked prediction model
- **Strengths**: Different learning objective captures different acoustic aspects
- **Configuration**:
  - **Layers Extracted**: [6, 9, 12, 15, 18] (matching Wav2Vec2 for consistency)
  - **Pooling**: Mean temporal pooling
  - **Layer Fusion**: Concatenation
  - **Hidden Size**: 1,024 per layer
  - **Feature Contribution**: 1,024 × 5 layers × 8 files = **40,960 features**

### 3.2 Feature Fusion Strategy
```
Audio Features (Wav2Vec2 + HuBERT): 81,920 features
Metadata Features:                       4 features (Age, Sex, Months, ALSFRS-R_start)
──────────────────────────────────────────────────────
TOTAL INPUT FEATURES:               81,924 features
```

**Fusion Method**: Concatenation
- **Rationale**: Preserves all information from both models
- **Alternative Considered**: Mean across models (not tested, would reduce dimensionality)

### 3.3 Preprocessing Pipeline
1. **Audio Resampling**: Standardize to 16kHz sample rate
2. **Mono Conversion**: Average stereo channels to mono if needed
3. **Feature Extraction**: Per-layer hidden states for both models
4. **Temporal Pooling**: Mean across time dimension
5. **Standardization**: StandardScaler per fold (no information leakage)

---

## 4. Model Selection & Baseline Evaluation

### 4.1 Model Comparison (Initial Screening)
Multiple models were screened to identify the strongest baseline:

| Model | Status | Rationale |
|-------|--------|-----------|
| **XGBoost** | ⭐ **Selected** | Best performance, widely-used for tabular data |
| Gradient Boosting | Good | Considered for ensemble (Section 7) |
| Random Forest | Weak | Lower performance |
| SVM | Weak | Lower performance |

**Decision**: XGBoost selected as primary model for baseline and ensemble components.

### 4.2 XGBoost Baseline Configuration
```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',          # CPU-based (GPU='gpu_hist')
    random_state=42,
    eval_metric='mlogloss',      # Multi-class log loss
    early_stopping_rounds=20,    # Stop if no improvement
    n_jobs=-1                    # Parallel jobs
)
```

### 4.3 Training Strategy
- **Train/Validation Split**: 80/20 (stratified)
- **Validation Purpose**: Early stopping to prevent overfitting
- **Scaling**: StandardScaler applied per fold

---

## 5. EXPERIMENT 1: XGBoost Baseline

### 5.1 Methodology
- **Approach**: Train XGBoost on full scaled features without modification
- **CV Strategy**: 5-fold stratified cross-validation
- **Per-Fold Process**:
  1. Extract features for train/test splits
  2. StandardScale features
  3. Train XGBoost with early stopping
  4. Evaluate on test fold
  5. Record F1-Macro score

### 5.2 Results (5-Fold CV)

**Per-Fold Scores**:
- Fold 1: **F1 = 0.5586**
- Fold 2: **F1 = 0.4139**
- Fold 3: **F1 = 0.4118**
- Fold 4: **F1 = 0.7004** ← High variance observed
- Fold 5: **F1 = 0.5582**

**Aggregate Results**:
```
F1-Macro: 0.5286 ± 0.1078
```

### 5.3 Analysis

**Performance vs. Challenge**:
- vs Challenge Baseline (0.5830): **-0.0544** ❌
- vs ISDS Leader (0.5794): **-0.0508** ❌

**Observations**:
1. **High Variance**: Std of 0.1078 indicates model sensitivity to fold composition
2. **Fold 4 Outlier**: F1 = 0.7004 is significantly higher (possible favorable class distribution)
3. **Below Baseline**: Not competitive with existing approaches
4. **Need for Improvement**: Original approach insufficient

**Interpretation**:
- Baseline XGBoost underperforms PART algorithm
- High variance suggests instability or data sensitivity
- Sample weights and ensemble approaches warranted

---

## 6. EXPERIMENT 2: Sample Weights for Class Imbalance

### 6.1 Motivation & Hypothesis
**Problem**: Severe class imbalance (13.6% rare class vs 37.9% common class)

**Hypothesis**: Applying balanced sample weights will:
- Increase importance of rare class (Class 0) during training
- Improve macro F1-score (which treats all classes equally)
- Reduce bias toward common classes

### 6.2 Implementation
```python
# Per-fold sample weight calculation
fold_weights = compute_sample_weight('balanced', y_train)

# Effect on class weights:
# Class 0 (Severe, rare):    weight ≈ 1.8x
# Class 1 (Moderate):        weight ≈ 1.0x
# Class 2 (Mild):            weight ≈ 0.9x
# Class 3 (None, common):    weight ≈ 0.7x

# Applied during training:
model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,              # Training weights
    eval_set=[(X_val, y_val)],
    sample_weight_eval_set=[w_val],  # Validation weights
    verbose=False
)
```

### 6.3 Results (5-Fold CV)

**Per-Fold Scores**:
- Fold 1: **F1 = 0.5599**
- Fold 2: **F1 = 0.5099**
- Fold 3: **F1 = 0.4653**
- Fold 4: **F1 = 0.5589**
- Fold 5: **F1 = 0.5512**

**Aggregate Results**:
```
F1-Macro: 0.5291 ± 0.0368
```

### 6.4 Comparative Analysis

**vs. Baseline**:
| Metric | Baseline | Sample Weights | Difference |
|--------|----------|----------------|------------|
| F1-Macro Mean | 0.5286 | 0.5291 | **+0.0005** ❌ |
| F1-Macro Std | 0.1078 | 0.0368 | **-0.0710** ✅ |

**vs. Challenge Baseline (0.5830)**:
- **-0.0539** (still below)

### 6.5 Key Findings

**What Happened**:
1. **Minimal Improvement**: +0.0005 is negligible
2. **Reduced Variance**: Std decreased from 0.1078 → 0.0368
   - Interpretation: Weights stabilized model across folds
   - Less dependent on specific fold composition
3. **Still Below Baseline**: Did not reach challenge baseline

**Why No Major Improvement**:
1. **XGBoost Robustness**: Already handles class imbalance reasonably
2. **F1-Macro Stability**: With 5 folds and balanced sampling, natural balance is maintained
3. **Model Limitation**: May not be capability issue but feature/architecture limitation

**Conclusion**:
✅ **Variance Reduction Achieved** (0.1078 → 0.0368)
❌ **No Meaningful Performance Gain** (only +0.0005)
⚠️ **Outcome**: Sample weights did NOT solve the performance gap

---

## 7. EXPERIMENT 3: Ensemble Learning

### 7.1 Rationale
**Hypothesis**: Combine XGBoost with Gradient Boosting to leverage complementary strengths

**Ensemble Strategy**:
1. Train two diverse models (XGBoost + Gradient Boosting)
2. Generate probability predictions for each fold
3. Test different weight combinations
4. Select weights that maximize F1-Macro

### 7.2 Ensemble Configuration

#### Model 1: XGBoost (Primary)
```python
XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    tree_method='hist',
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=20
)
```

#### Model 2: Gradient Boosting (Secondary)
```python
GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

### 7.3 Initial Ensemble Results (Equal Weights [1:1])

**Per-Fold Scores**:
- Fold 1: **F1 = 0.5587**
- Fold 2: **F1 = 0.3821**
- Fold 3: **F1 = 0.4735**
- Fold 4: **F1 = 0.5729**
- Fold 5: **F1 = 0.4528**

**Aggregate Results (Equal Weights)**:
```
F1-Macro: 0.4880 ± 0.0706
```

### 7.4 Weight Optimization

**Approach**: Test different weight ratios on OOF predictions
```python
weight_candidates = [
    (1, 1),   # Equal weights
    (2, 1),   # XGBoost favored
    (3, 1),   # XGBoost strongly favored
    (1, 2),   # GB favored
    (1, 3)    # GB strongly favored
]
```

**Optimization Results**:

| Weights [XGB:GB] | F1-Macro | vs Baseline |
|------------------|----------|------------|
| [1:1] Equal | 0.4937 | -0.0349 |
| [2:1] | 0.5311 | +0.0025 |
| **[3:1]** | **0.5346** | **+0.0060** |
| [1:2] | 0.5068 | -0.0218 |
| [1:3] | 0.4871 | -0.0415 |

**Optimal Weights Found**: **[3:1]** (XGBoost : Gradient Boosting)
- **Best F1-Macro**: 0.5346

### 7.5 Analysis of Ensemble Results

**Observation 1: Equal Weights Underperformed**
- Ensemble [1:1]: 0.4880 vs Baseline: 0.5286 → **-0.0406 ❌**
- GB model brought down XGBoost performance
- Indicates GB is weaker than XGBoost for this task

**Observation 2: Weight Optimization Helped**
- Best weights [3:1]: 0.5346 vs Baseline: 0.5286 → **+0.0060 ✅**
- First real improvement found!
- Giving 3x weight to XGBoost reduced GB's negative impact

**Observation 3: Still Below Challenge Baseline**
- Best ensemble 0.5346 vs Challenge Baseline 0.5830 → **-0.0484 ❌**
- Modest improvement but insufficient

### 7.6 Key Insights

1. **Ensemble with Equal Weights Backfires**: Weak model dilutes strong model
2. **Weighting Matters**: Properly weighted ensemble outperforms unweighted
3. **Model Diversity Limits**: GB doesn't add sufficient new information
4. **XGBoost Dominance**: Results favor XGBoost heavily, questioning ensemble value

---

## 8. Cross-Experiment Comparison

### 8.1 Complete Results Summary

| Experiment | F1-Macro | Std | vs Challenge Baseline | Status |
|-----------|----------|-----|----------------------|--------|
| Challenge Baseline (PART) | 0.5830 | - | - | Reference |
| Challenge Leader (ISDS) | 0.5794 | - | -0.0036 | Competitive |
| **Baseline XGBoost** | **0.5286** | **0.1078** | **-0.0544** | ❌ Below |
| **Sample Weights** | **0.5291** | **0.0368** | **-0.0539** | ❌ No gain |
| **Ensemble [1:1]** | **0.4880** | **0.0706** | **-0.0950** | ❌ Worse |
| **Ensemble [3:1]** (OOF) | **0.5346** | - | **-0.0484** | ⚠️ Marginal |

### 8.2 Findings by Experiment

**Experiment 1 - Baseline XGBoost**:
- ✅ Identified XGBoost as best individual model
- ❌ Underperforms competition baseline
- ⚠️ High variance (0.1078) indicates instability

**Experiment 2 - Sample Weights**:
- ✅ Reduced variance significantly (0.1078 → 0.0368)
- ❌ No meaningful performance improvement
- 💡 Suggests class imbalance not primary issue

**Experiment 3 - Ensemble**:
- ✅ Weight optimization improved unweighted ensemble
- ❌ Equal weights made performance worse
- ❌ Still below challenge baseline
- 💡 Shows importance of model weighting

### 8.3 Best Performer

**Best Overall**: Baseline XGBoost with F1 = 0.5286
- Simple, stable, no complex optimization
- Better than ensemble variants
- Better than sample weights approach

**Runner-Up**: Ensemble [3:1] with F1 = 0.5346
- Marginally better than baseline by +0.0060
- Requires careful weight selection
- Limited practical benefit

---

## 9. Discussion & Interpretations

### 9.1 Why Did We Not Beat the Baseline?

**Possible Reasons**:

1. **Feature Quality**:
   - 81,924 features might be suboptimal
   - Feature selection might have helped
   - Different pre-trained models might perform better

2. **Model Architecture**:
   - XGBoost + GB may not be ideal for this task
   - Deep learning end-to-end approach might be better
   - Neural networks for audio might capture patterns better

3. **Data Limitations**:
   - Only 132 samples is very small
   - Cross-validation with 5 folds = ~26 test samples per fold
   - High variance from small folds (Baseline Std: 0.1078)

4. **Class Imbalance**:
   - Rare class (13.6%) difficult to learn
   - F1-Macro penalizes rare class mistakes equally
   - Even balanced models struggle with such imbalance

5. **Feature Engineering**:
   - Multi-layer concatenation might not be optimal
   - Different layer combinations might work better
   - Could have performed PCA or feature selection

### 9.2 Variance Analysis

**High Variance in Baseline** (Std = 0.1078):
- Fold 2: 0.4139 (Low)
- Fold 3: 0.4118 (Low)
- Fold 4: 0.7004 (Very High) ← Outlier
- Range: 0.5886 (difference between high and low)

**Interpretation**:
- Model performance heavily dependent on test fold composition
- Possible: Fold 4 had favorable class distribution
- Sample Weights helped by reducing this variance (→ 0.0368)

### 9.3 Dimensionality Analysis

**Current Setup: 81,924 features**

**Thoughts**:
- Large feature space relative to 132 samples (616:1 ratio)
- Potential for overfitting or curse of dimensionality
- Feature selection was NOT tested in this run
- Could have tried PCA to reduce dimensions

### 9.4 What Worked

✅ **Consistent Implementation**:
- Proper CV methodology (no data leakage)
- Fold-specific transformations
- Stratified splits maintaining class balance
- Reproducible with seed=42

✅ **Comprehensive Experimentation**:
- Tested 3 different approaches (baseline, weights, ensemble)
- Systematic hyperparameter search in ensemble
- Documented all results

✅ **Variance Reduction**:
- Sample weights reduced variance from 0.1078 → 0.0368
- Shows stability improvements even without performance gains

### 9.5 What Didn't Work

❌ **Sample Weights**:
- Only +0.0005 improvement (negligible)
- Reduced variance but not performance
- Suggests not addressing core limitation

❌ **Ensemble with Gradient Boosting**:
- [1:1] weights: -0.0406 degradation
- Even [3:1] optimal weights: only +0.0060 improvement
- GB doesn't complement XGBoost effectively

❌ **Achieving Competitive Performance**:
- Best result: 0.5346 (ensemble [3:1])
- Target: 0.5830 (challenge baseline)
- Gap: 0.0484 remains

---

## 10. Conclusions & Recommendations

### 10.1 Final Performance Summary

```
┌─────────────────────────────────────────────────┐
│ FINAL RESULTS (5-Fold CV)                      │
├─────────────────────────────────────────────────┤
│ Best Performer: XGBoost Baseline                │
│ F1-Macro: 0.5286 ± 0.1078                      │
│ vs Challenge Baseline: -0.0544 (7.3% gap)      │
│                                                 │
│ Alternative: Ensemble [3:1]                    │
│ F1-Macro: 0.5346 (OOF optimized)               │
│ Gain: +0.0060 (+1.1% relative improvement)     │
└─────────────────────────────────────────────────┘
```

### 10.2 What We Learned

**Key Insights**:
1. XGBoost is robust baseline but insufficient for this task
2. Sample weights reduce variance without improving core performance
3. Ensemble requires careful weighting to avoid degradation
4. 81,924 features with only 132 samples creates challenges

**Validation Importance**:
- 5-fold CV provides robust evaluation
- High variance in baseline (0.1078) important to track
- OOF optimization can overfit (use nested CV for true estimate)

**Dataset Challenges**:
- Severe class imbalance (13.6% rare class)
- Very small sample size (132 total)
- High dimensionality relative to samples
- Results in high fold-to-fold variance

### 10.3 Recommendations for Future Work

**Short-term (within current framework)**:
1. Perform feature selection (e.g., SelectKBest) to reduce dimensionality
2. Test different layer combinations from Wav2Vec2/HuBERT
3. Try other pre-trained models (Whisper, WavLM, etc.)
4. Implement nested CV for true unbiased weight optimization

**Medium-term (different approaches)**:
1. Try end-to-end deep learning models
2. Implement data augmentation for rare class
3. Use class-specific losses (focal loss, weighted cross-entropy)
4. Explore other ensemble methods (stacking, blending)

**Long-term (research directions)**:
1. Collect more data to increase sample size
2. Domain-specific model pretraining on ALS speech data
3. Incorporate clinical features more explicitly
4. Multi-task learning (joint dysarthria + progression prediction)

### 10.4 Technical Validity Assessment

**Methodology Strengths ✅**:
- Proper 5-fold stratified cross-validation
- No data leakage (features extracted per fold)
- Fold-specific transformations (scaling, weights)
- Global seed for reproducibility
- Early stopping to prevent overfitting
- Separate train/validation split for early stopping

**Methodology Considerations ⚠️**:
- OOF weight optimization can overfit (could use nested CV)
- Small fold sizes (26-27 test samples per fold) increase variance
- No cross-validation of ensemble weights independently

**Recommendations for Future Reproducibility**:
- Use nested CV for hyperparameter/weight selection
- Report confidence intervals alongside means
- Document all data preprocessing steps
- Save train/test indices for full reproducibility

---

## 11. Technical Appendix

### 11.1 Computational Resources Used
- **Processing**: GPU acceleration (CUDA/Metal/MPS)
- **Feature Extraction**: ~45 minutes for first run (1,056 audio files)
- **Model Training**: ~15 minutes per experiment (with 5-fold CV)
- **Total Runtime**: ~2 hours for all three experiments

### 11.2 Software Stack
```
PyTorch 2.x
Transformers 4.x (Hugging Face)
XGBoost 2.x
Scikit-learn 1.3+
SoundFile (audio I/O)
NumPy, Pandas (data processing)
Matplotlib, Seaborn (visualization)
```

### 11.3 Reproducibility Details
- **Random Seed**: 42 (applied to NumPy, PyTorch, random module, CUDNN)
- **CV Strategy**: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- **Train/Val Split**: train_test_split(..., test_size=0.2, random_state=42)
- **All results**: Saved with configuration JSON for full reproducibility

### 11.4 Output Files Generated
```
experiments/task2/
├── experiments/
│   ├── 20260120_170706_XGBoost_Baseline.json
│   ├── 20260120_173136_XGBoost_SampleWeights.json
│   └── 20260120_191050_Ensemble_XGB_GB.json
├── final_results/
│   ├── baseline_oof_predictions.npy
│   ├── sample_weights_oof_predictions.npy
│   ├── ensemble_oof_predictions.npy
│   └── config.json
└── experiment_comparison.png
```

### 11.5 Model Configuration Details

**XGBoost Hyperparameters**:
```python
n_estimators = 200      # Number of boosting rounds
max_depth = 5           # Max tree depth (shallow trees)
learning_rate = 0.1     # Boosting learning rate
subsample = 0.8         # Fraction of samples per iteration
colsample_bytree = 0.8  # Fraction of features per tree
tree_method = 'hist'    # Histogram-based (CPU efficient)
early_stopping = 20     # Stop if no improvement for 20 rounds
eval_metric = 'mlogloss'# Multi-class log loss
```

**Gradient Boosting Hyperparameters**:
```python
n_estimators = 200
max_depth = 5
learning_rate = 0.1
random_state = 42
```

**Feature Extraction Parameters**:
```python
Wav2Vec2 Layers: [6, 9, 12, 15, 18]
HuBERT Layers: [6, 9, 12, 15, 18]
Audio Sample Rate: 16,000 Hz
Pooling: Mean temporal pooling
Layer Fusion: Concatenation
```

---

## 12. Conclusion

This comprehensive 5-fold cross-validation study evaluated three distinct approaches to ALS progression prediction:

1. **XGBoost Baseline**: F1 = 0.5286 ± 0.1078
   - Simple, interpretable approach
   - Highest individual performance
   - High variance across folds

2. **Sample Weights**: F1 = 0.5291 ± 0.0368
   - Reduced variance significantly
   - Minimal performance improvement (+0.0005)
   - Showed instability was not primary issue

3. **Ensemble Learning**: F1 = 0.4880 [1:1] → 0.5346 [3:1 optimized]
   - Equal weights degraded performance
   - Optimal weighting [3:1] provided +0.0060 improvement
   - Complex without sufficient benefit

**Key Takeaway**: Despite comprehensive optimization attempts, the 81,924-dimensional feature space with 132 samples proved challenging. The XGBoost baseline achieved F1 = 0.5286, falling short of the challenge baseline (0.5830) by 0.0544. Future work should explore alternative feature representations, deep learning approaches, or ensemble methods with more complementary models.

**Methodological Success**: While absolute performance fell short of competition, the experiment demonstrated rigorous validation methodology with proper cross-validation, no data leakage, and reproducible results.

---

**Experiment Date**: 20 January 2026  
**Executed in**: SAND_TASK_2_QuickRun_FINAL.ipynb  
**All results saved with configuration for full reproducibility**
