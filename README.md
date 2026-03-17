# BiLSTM + Attention — Alzheimer's Disease Binary Detection
## Dataset: Kaggle — Rabie El Kharoua (2024)

Binary classification: **Alzheimer's Disease (AD)** vs **Healthy (No AD)**

---

## Dataset
**Download** (free, no registration):
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

**File**: `alzheimers_disease_data.csv`
**Size**: 2,149 patients × 35 features + 1 label

### Features used (32 columns)
| Group               | Features |
|---------------------|----------|
| Demographic         | Age, Gender, Ethnicity, EducationLevel |
| Lifestyle           | BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality |
| Medical History     | FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension |
| Clinical Measures   | SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides |
| Cognitive Tests     | MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL |
| Symptoms            | Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness |

**Target**: `Diagnosis` — 0 = Healthy, 1 = Alzheimer's Disease

---

## Architecture
```
Input  (T=10, 32 features)
  ↓
BiLSTM Layer 1   [128 units × 2 dirs = 256]  + Dropout(0.3)
  ↓
BiLSTM Layer 2   [64  units × 2 dirs = 128]  + Dropout(0.3)
  ↓
Bahdanau Attention  (learns which time steps matter most)
  ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.4)
  ↓
Dense(64, ReLU)
  ↓
Dense(2, Softmax)
  ↓
[P(Healthy), P(AD)]
```

---

## Files
```
model.py               BiLSTM + BahdanauAttention architecture
data_preprocessing.py  Load CSV, SMOTE, normalise, build sequences
train.py               Full training loop + evaluation + plots
inference.py           Load saved model, predict on new patients
requirements.txt       Python dependencies
```

---

## Quick Start

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Download dataset from Kaggle
```bash
# Option A: Kaggle CLI
kaggle datasets download -d rabieelkharoua/alzheimers-disease-dataset
unzip alzheimers-disease-dataset.zip

# Option B: Manual download from
# https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
```

### Step 3 — Train
```bash
python train.py --data alzheimers_disease_data.csv
```

### Step 4 — Predict on new patients
```bash
python inference.py \
  --model    outputs/checkpoints/final_model.keras \
  --scaler   outputs/scaler.pkl \
  --features outputs/feature_names.pkl
```

---

## Output Files (after training)
```
outputs/
  checkpoints/
    best_model.keras      Best checkpoint (highest val AUC)
    final_model.keras     Final model after all epochs
  scaler.pkl              Fitted StandardScaler
  feature_names.pkl       Ordered feature list
  training_log.csv        Epoch-by-epoch metrics
  plots/
    training_curves.png   Accuracy / AUC / Loss / Precision curves
    confusion_matrix.png  TP / TN / FP / FN heatmap
    roc_pr_curves.png     ROC + Precision-Recall curves
    feature_importance.png Top 15 attention-weighted features
```

---

## Expected Performance (on this dataset)
| Metric      | Expected |
|-------------|----------|
| Accuracy    | ~92–96%  |
| AUC-ROC     | ~0.96–0.99 |
| Sensitivity | ~0.93+   |
| Specificity | ~0.91+   |
| F1 Score    | ~0.94+   |

> ⚠ For clinical use, validate on an independent cohort (ADNI / OASIS).
> This is a research model. Always involve a qualified neurologist.

---

## Key Design Decisions
- **SMOTE** applied to training set to handle class imbalance
- **EarlyStopping** monitors `val_auc` (not loss) — prioritises discrimination ability
- **Sequence simulation**: single-visit tabular data is converted to T=10 pseudo-sequences
  with small Gaussian noise per step, enabling BiLSTM to learn temporal feature dependencies
- **Bahdanau Attention** highlights which clinical time steps drove each prediction
- **Sensitivity prioritised** over specificity — missing an AD case (FN) is clinically worse
  than a false alarm (FP)
