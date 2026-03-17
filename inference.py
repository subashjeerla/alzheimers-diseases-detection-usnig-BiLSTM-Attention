"""
Inference — Alzheimer's BiLSTM + Attention
==========================================
Load trained model and predict on new patients.

Usage (Python):
  from inference import AlzheimerPredictor
  predictor = AlzheimerPredictor("outputs/checkpoints/final_model.keras",
                                  "outputs/scaler.pkl",
                                  "outputs/feature_names.pkl")
  result = predictor.predict(patient_dict)
  predictor.print_result(result)

Usage (CLI):
  python inference.py --model outputs/checkpoints/final_model.keras \
                      --scaler outputs/scaler.pkl \
                      --features outputs/feature_names.pkl
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from model import BahdanauAttention, build_interpretable_model
from data_preprocessing import prepare_patient, CLASS_NAMES, FEATURE_COLS


class AlzheimerPredictor:

    def __init__(
        self,
        model_path   : str,
        scaler_path  : str,
        features_path: str,
        sequence_length: int = 10,
    ):
        self.sequence_length = sequence_length

        # Load scaler and feature names
        with open(scaler_path,   "rb") as f: self.scaler   = pickle.load(f)
        with open(features_path, "rb") as f: self.feature_names = pickle.load(f)

        # Load model with custom layer
        self.model = load_model(
            model_path,
            custom_objects={"BahdanauAttention": BahdanauAttention},
        )

        # Interpretable model (also returns attention weights)
        try:
            self.interp_model = build_interpretable_model(self.model)
        except Exception:
            self.interp_model = None

        print(f"Model loaded: {model_path}")
        print(f"Features    : {len(self.feature_names)}")

    def predict(self, patient: dict) -> dict:
        """
        Predict AD vs Healthy for one patient.

        Args:
            patient: dict with feature names as keys.
                     Missing features are filled with 0.

        Returns:
            {
              predicted_class   : "Alzheimer's Disease" or "Healthy (No AD)"
              ad_probability    : float  (0–1)
              healthy_probability: float
              confidence        : "High" / "Moderate" / "Low"
              risk_level        : "High Risk" / "Moderate Risk" / "Low Risk"
              top_features      : [(name, importance), ...]
              attention_weights : np.ndarray (T,) or None
            }
        """
        X = prepare_patient(
            patient, self.scaler, self.feature_names, self.sequence_length
        )

        if self.interp_model:
            probs, attn_w = self.interp_model.predict(X, verbose=0)
            attn_w = attn_w[0]
        else:
            probs  = self.model.predict(X, verbose=0)
            attn_w = None

        probs      = probs[0]
        ad_prob    = float(probs[1])
        pred_idx   = int(np.argmax(probs))
        confidence = "High" if max(probs) > 0.80 else "Moderate" if max(probs) > 0.60 else "Low"
        risk       = "High Risk" if ad_prob > 0.70 else "Moderate Risk" if ad_prob > 0.40 else "Low Risk"

        # Feature importance: scale raw feature values by attention
        raw_vec = np.array([float(patient.get(f, 0)) for f in self.feature_names])
        if attn_w is not None:
            importance = np.abs(raw_vec) * float(np.mean(attn_w))
        else:
            importance = np.abs(raw_vec)
        top_idx   = np.argsort(importance)[::-1][:8]
        top_feats = [(self.feature_names[i], float(importance[i])) for i in top_idx]

        return {
            "predicted_class"    : CLASS_NAMES[pred_idx],
            "ad_probability"     : ad_prob,
            "healthy_probability": float(probs[0]),
            "confidence"         : confidence,
            "risk_level"         : risk,
            "top_features"       : top_feats,
            "attention_weights"  : attn_w,
        }

    def print_result(self, r: dict):
        line = "─" * 50
        print(f"\n{line}")
        print(f"  Prediction    : {r['predicted_class']}")
        print(f"  AD Probability: {r['ad_probability']*100:.1f}%")
        print(f"  Risk Level    : {r['risk_level']}")
        print(f"  Confidence    : {r['confidence']}")
        print(f"{line}")
        ad_bar  = "█" * int(r['ad_probability']  * 30)
        hlt_bar = "█" * int(r['healthy_probability'] * 30)
        print(f"  Alzheimer's  {ad_bar:<30} {r['ad_probability']*100:5.1f}%")
        print(f"  Healthy      {hlt_bar:<30} {r['healthy_probability']*100:5.1f}%")
        print(f"{line}")
        print("  Key driving features:")
        for name, score in r["top_features"][:5]:
            print(f"    {name:<35} {score:.4f}")
        print(f"{line}")
        print("  ⚠  Research tool only. Consult a neurologist for diagnosis.")
        print(f"{line}\n")


# ─────────────────────────────────────────────
# Sample patient for quick test
# ─────────────────────────────────────────────

SAMPLE_PATIENT_AD = {
    "Age": 75, "Gender": 1, "Ethnicity": 0, "EducationLevel": 1,
    "BMI": 26.5, "Smoking": 1, "AlcoholConsumption": 2.0,
    "PhysicalActivity": 1.5, "DietQuality": 4.0, "SleepQuality": 5.0,
    "FamilyHistoryAlzheimers": 1, "CardiovascularDisease": 1,
    "Diabetes": 0, "Depression": 1, "HeadInjury": 0, "Hypertension": 1,
    "SystolicBP": 145, "DiastolicBP": 88,
    "CholesterolTotal": 220, "CholesterolLDL": 140,
    "CholesterolHDL": 45, "CholesterolTriglycerides": 180,
    "MMSE": 18,             # Low MMSE — strong AD indicator
    "FunctionalAssessment": 2.5,
    "MemoryComplaints": 1, "BehavioralProblems": 1, "ADL": 3.0,
    "Confusion": 1, "Disorientation": 1, "PersonalityChanges": 1,
    "DifficultyCompletingTasks": 1, "Forgetfulness": 1,
}

SAMPLE_PATIENT_HEALTHY = {
    "Age": 65, "Gender": 0, "Ethnicity": 0, "EducationLevel": 3,
    "BMI": 23.0, "Smoking": 0, "AlcoholConsumption": 1.0,
    "PhysicalActivity": 7.0, "DietQuality": 8.0, "SleepQuality": 8.0,
    "FamilyHistoryAlzheimers": 0, "CardiovascularDisease": 0,
    "Diabetes": 0, "Depression": 0, "HeadInjury": 0, "Hypertension": 0,
    "SystolicBP": 118, "DiastolicBP": 72,
    "CholesterolTotal": 175, "CholesterolLDL": 95,
    "CholesterolHDL": 65, "CholesterolTriglycerides": 110,
    "MMSE": 29,             # High MMSE — healthy indicator
    "FunctionalAssessment": 9.0,
    "MemoryComplaints": 0, "BehavioralProblems": 0, "ADL": 9.5,
    "Confusion": 0, "Disorientation": 0, "PersonalityChanges": 0,
    "DifficultyCompletingTasks": 0, "Forgetfulness": 0,
}


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="outputs/checkpoints/final_model.keras")
    parser.add_argument("--scaler",   default="outputs/scaler.pkl")
    parser.add_argument("--features", default="outputs/feature_names.pkl")
    args = parser.parse_args()

    predictor = AlzheimerPredictor(args.model, args.scaler, args.features)

    print("\n--- Testing with high-risk AD patient ---")
    predictor.print_result(predictor.predict(SAMPLE_PATIENT_AD))

    print("--- Testing with healthy patient ---")
    predictor.print_result(predictor.predict(SAMPLE_PATIENT_HEALTHY))
