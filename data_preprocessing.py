"""
Data Preprocessing — Kaggle Alzheimer's Disease Dataset
=========================================================
Dataset : Rabie El Kharoua (2024) — 2,149 patients, 35 features
Download: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
File    : alzheimers_disease_data.xlsx  (or .csv — both supported)

Goal    : Binary classification — AD (1) vs Healthy/No-Diagnosis (0)

Column groups in the dataset:
  Demographic       : Age, Gender, Ethnicity, EducationLevel
  Lifestyle         : BMI, Smoking, AlcoholConsumption, PhysicalActivity,
                      DietQuality, SleepQuality
  Medical History   : FamilyHistoryAlzheimers, CardiovascularDisease,
                      Diabetes, Depression, HeadInjury, Hypertension
  Clinical Measures : SystolicBP, DiastolicBP, CholesterolTotal,
                      CholesterolLDL, CholesterolHDL, CholesterolTriglycerides
  Cognitive Tests   : MMSE, FunctionalAssessment, MemoryComplaints,
                      BehavioralProblems, ADL
  Symptoms          : Confusion, Disorientation, PersonalityChanges,
                      DifficultyCompletingTasks, Forgetfulness
  Target            : Diagnosis  (0 = No AD, 1 = AD)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# Column Definitions
# ─────────────────────────────────────────────

# All 35 input features (drop PatientID, DoctorInCharge, Diagnosis)
FEATURE_COLS = [
    # Demographic
    "Age", "Gender", "Ethnicity", "EducationLevel",
    # Lifestyle
    "BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity",
    "DietQuality", "SleepQuality",
    # Medical history (binary flags)
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension",
    # Clinical measurements
    "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides",
    # Cognitive & functional assessments
    "MMSE", "FunctionalAssessment", "MemoryComplaints",
    "BehavioralProblems", "ADL",
    # Symptoms
    "Confusion", "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness",
]

TARGET_COL   = "Diagnosis"
PATIENT_ID   = "PatientID"
CLASS_NAMES  = ["Healthy (No AD)", "Alzheimer's Disease"]
N_FEATURES   = len(FEATURE_COLS)   # 32
N_CLASSES    = 2


# ─────────────────────────────────────────────
# Load & Clean
# ─────────────────────────────────────────────

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the Kaggle Alzheimer's dataset.
    Supports Excel (.xlsx / .xls) AND CSV (.csv) — detected automatically.

    Args:
        file_path: Path to your downloaded file:
                   alzheimers_disease_data.xlsx  (Excel from Kaggle)
                   alzheimers_disease_data.csv   (CSV version)

    Returns:
        Cleaned DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\n  Dataset not found: {file_path}\n"
            f"  Download: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset\n"
            f"  Accepted formats: .xlsx, .xls, .csv\n"
            f"  Place the file in the same folder as train.py"
        )

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".xlsx", ".xls"):
        print(f"Reading Excel file : {file_path}")
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        try:
            df = pd.read_excel(file_path, engine=engine)
        except ImportError:
            raise ImportError(
                "Missing Excel reader. Fix with:\n"
                "  pip install openpyxl\n"
                "Then re-run train.py"
            )
    else:
        print(f"Reading CSV file   : {file_path}")
        df = pd.read_csv(file_path)

    print(f"Loaded  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Target  : {df[TARGET_COL].value_counts().to_dict()}")

    # Drop identifier and free-text columns if present
    drop_cols = [c for c in ["PatientID", "DoctorInCharge"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Keep only the feature columns + target that exist in this file
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"Warning : {len(missing)} expected columns not found: {missing}")

    df = df[available + [TARGET_COL]].copy()

    # Handle missing values
    num_cols = df[available].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[available].select_dtypes(exclude=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    print(f"Features: {len(available)} columns used")
    print(f"Missing % after fill: {df.isnull().mean().max():.2%}")
    return df


# ─────────────────────────────────────────────
# EDA Summary
# ─────────────────────────────────────────────

def print_eda_summary(df: pd.DataFrame):
    """Print a quick exploratory summary of the dataset."""
    print("\n" + "═"*55)
    print("  DATASET SUMMARY (EDA)")
    print("═"*55)
    print(f"  Rows          : {df.shape[0]:,}")
    print(f"  Features      : {df.shape[1]-1}")
    print(f"  Class balance :")
    vc = df[TARGET_COL].value_counts()
    for label, count in vc.items():
        name = CLASS_NAMES[int(label)]
        print(f"    {name:<22} {count:,}  ({count/len(df)*100:.1f}%)")
    print(f"  Age range     : {df['Age'].min():.0f} – {df['Age'].max():.0f}  (mean {df['Age'].mean():.1f})")
    if "MMSE" in df.columns:
        print(f"  MMSE range    : {df['MMSE'].min():.1f} – {df['MMSE'].max():.1f}  (mean {df['MMSE'].mean():.1f})")
    print("═"*55 + "\n")


# ─────────────────────────────────────────────
# Build Sequences for BiLSTM
# ─────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    sequence_length: int = 10,
    stride: int = 1,
) -> np.ndarray:
    """
    Convert a 2D patient matrix (n_patients, n_features) into
    3D sequences (n_patients, sequence_length, n_features) for BiLSTM.

    Strategy: each patient's feature vector is augmented with small
    Gaussian noise across `sequence_length` steps, simulating
    repeated clinical measurements. This is appropriate when you
    have single-visit tabular data (as in this Kaggle dataset).

    For true longitudinal data (ADNI/NACC with multiple visits),
    replace this with actual visit sequences per patient.

    Args:
        X              : (n_patients, n_features) normalised array
        sequence_length: Number of pseudo time steps (T)
        stride         : Noise scaling per step

    Returns:
        X_seq: (n_patients, sequence_length, n_features)
    """
    n_patients, n_features = X.shape
    X_seq = np.zeros((n_patients, sequence_length, n_features), dtype=np.float32)

    for t in range(sequence_length):
        # Slight Gaussian noise at each step simulates measurement variability
        noise_scale = 0.01 * (t / sequence_length)
        X_seq[:, t, :] = X + np.random.normal(0, noise_scale, X.shape)

    return X_seq


# ─────────────────────────────────────────────
# Full Preprocessing Pipeline
# ─────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    sequence_length: int = 10,
    test_size: float = 0.15,
    val_size: float  = 0.15,
    apply_smote: bool = True,
    random_state: int = 42,
) -> dict:
    """
    Complete preprocessing pipeline:
      1. Extract features and binary target
      2. Train / val / test split (stratified)
      3. Z-score normalisation (fit on train only)
      4. SMOTE oversampling on train set (handles class imbalance)
      5. Build 3D sequences for BiLSTM
      6. One-hot encode labels
      7. Compute class weights

    Args:
        df             : Cleaned DataFrame from load_dataset()
        sequence_length: BiLSTM time steps (T)
        test_size      : Fraction for test set
        val_size       : Fraction for validation set
        apply_smote    : Whether to apply SMOTE on training data
        random_state   : Reproducibility seed

    Returns:
        dict with X_train/val/test, y_train/val/test, scaler, class_weights, etc.
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X_raw = df[available_features].values.astype(np.float32)
    y_raw = df[TARGET_COL].values.astype(np.int32)

    # ── Train / val+test split ────────────────────────────────
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X_raw, y_raw,
        test_size=(test_size + val_size),
        stratify=y_raw,
        random_state=random_state,
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - rel_val),
        stratify=y_temp,
        random_state=random_state,
    )

    # ── Z-score normalisation ─────────────────────────────────
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ── SMOTE oversampling (train only) ───────────────────────
    if apply_smote:
        before = dict(zip(*np.unique(y_tr, return_counts=True)))
        sm = SMOTE(random_state=random_state)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        after = dict(zip(*np.unique(y_tr, return_counts=True)))
        print(f"SMOTE   : {before}  →  {after}")

    # ── Class weights ─────────────────────────────────────────
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    # ── Build 3D sequences ────────────────────────────────────
    np.random.seed(random_state)
    X_tr_seq   = build_sequences(X_tr,   sequence_length)
    X_val_seq  = build_sequences(X_val,  sequence_length)
    X_test_seq = build_sequences(X_test, sequence_length)

    # ── One-hot encode ────────────────────────────────────────
    y_tr_oh   = to_categorical(y_tr,   N_CLASSES)
    y_val_oh  = to_categorical(y_val,  N_CLASSES)
    y_test_oh = to_categorical(y_test, N_CLASSES)

    print(f"\nSplit   : Train {X_tr_seq.shape[0]:,}  |  Val {X_val_seq.shape[0]:,}  |  Test {X_test_seq.shape[0]:,}")
    print(f"Shapes  : X={X_tr_seq.shape}  y={y_tr_oh.shape}")
    print(f"Weights : {class_weights}\n")

    return {
        "X_train": X_tr_seq,   "y_train": y_tr_oh,   "y_train_int": y_tr,
        "X_val":   X_val_seq,  "y_val":   y_val_oh,  "y_val_int":   y_val,
        "X_test":  X_test_seq, "y_test":  y_test_oh, "y_test_int":  y_test,
        "scaler":        scaler,
        "class_weights": class_weights,
        "class_names":   CLASS_NAMES,
        "feature_names": available_features,
        "n_features":    len(available_features),
        "n_classes":     N_CLASSES,
        "sequence_length": sequence_length,
    }


# ─────────────────────────────────────────────
# Single Patient Inference Helper
# ─────────────────────────────────────────────

def prepare_patient(
    raw_features: dict,
    scaler: StandardScaler,
    feature_names: list,
    sequence_length: int = 10,
) -> np.ndarray:
    """
    Prepare one patient's raw feature dict for inference.

    Args:
        raw_features  : {feature_name: value} — raw unscaled values
        scaler        : Fitted StandardScaler from training
        feature_names : Ordered list of feature names used in training
        sequence_length: Model's T

    Returns:
        X: shape (1, sequence_length, n_features)
    """
    vec = np.array([raw_features.get(f, 0.0) for f in feature_names], dtype=np.float32)
    vec_scaled = scaler.transform(vec.reshape(1, -1))
    seq = build_sequences(vec_scaled, sequence_length)   # (1, T, F)
    return seq


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "alzheimers_disease_data.csv"
    df = load_dataset(path)
    print_eda_summary(df)
    data = preprocess(df)
    print("Preprocessing complete.")
