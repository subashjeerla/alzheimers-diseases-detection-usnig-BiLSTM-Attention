"""
Train — BiLSTM + Attention Alzheimer's Binary Classifier
=========================================================
Dataset : Kaggle — Rabie El Kharoua (2024)
          https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset

Usage:
  python train.py --data alzheimers_disease_data.csv
  python train.py --data alzheimers_disease_data.csv --epochs 80 --lr 0.0005
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)
import tensorflow as tf

from model import build_model, BahdanauAttention
from data_preprocessing import load_dataset, print_eda_summary, preprocess, CLASS_NAMES


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CFG = {
    "sequence_length" : 10,
    "lstm_units_1"    : 128,
    "lstm_units_2"    : 64,
    "attention_units" : 64,
    "dense_units"     : 128,
    "dropout_rate"    : 0.3,
    "rec_dropout"     : 0.2,
    "l2_reg"          : 1e-4,
    "learning_rate"   : 1e-3,
    "batch_size"      : 32,
    "max_epochs"      : 100,
    "es_patience"     : 12,
    "lr_patience"     : 6,
    "lr_factor"       : 0.5,
    "min_lr"          : 1e-6,
    "out_dir"         : "outputs",
}


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

def get_callbacks(out_dir: str, cfg: dict) -> list:
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=cfg["es_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg["lr_factor"],
            patience=cfg["lr_patience"],
            min_lr=cfg["min_lr"],
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{out_dir}/checkpoints/best_model.keras",
            monitor="val_auc", mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(f"{out_dir}/training_log.csv"),
    ]


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, data: dict) -> dict:
    X_test     = data["X_test"]
    y_test_oh  = data["y_test"]
    y_test_int = data["y_test_int"]

    y_pred_prob = model.predict(X_test, batch_size=32, verbose=0)
    y_pred_int  = np.argmax(y_pred_prob, axis=1)
    y_pred_ad   = y_pred_prob[:, 1]     # probability of AD class

    loss, acc, auc, prec, rec = model.evaluate(X_test, y_test_oh, verbose=0)
    f1  = 2 * prec * rec / (prec + rec + 1e-9)
    cm  = confusion_matrix(y_test_int, y_pred_int)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-9)

    report = classification_report(
        y_test_int, y_pred_int,
        target_names=CLASS_NAMES, output_dict=True,
    )

    print("\n" + "═"*55)
    print("  BINARY CLASSIFICATION RESULTS — TEST SET")
    print("═"*55)
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Precision   : {prec:.4f}  (AD class)")
    print(f"  Recall      : {rec:.4f}  (AD class = Sensitivity)")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Loss        : {loss:.4f}")
    print("─"*55)
    print(f"  Confusion Matrix:")
    print(f"    True Negatives  (TN) : {tn}   (Healthy predicted Healthy)")
    print(f"    False Positives (FP) : {fp}   (Healthy predicted AD)")
    print(f"    False Negatives (FN) : {fn}   (AD predicted Healthy)  ← minimise")
    print(f"    True Positives  (TP) : {tp}   (AD predicted AD)")
    print("═"*55 + "\n")

    return {
        "accuracy": acc, "auc": auc, "precision": prec,
        "recall": rec, "specificity": specificity, "f1": f1,
        "loss": loss, "confusion_matrix": cm,
        "y_pred_prob": y_pred_prob, "y_pred_int": y_pred_int,
        "y_pred_ad": y_pred_ad, "report": report,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_all(history, results: dict, data: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Training curves (4 metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BiLSTM + Attention — Training History", fontsize=14, fontweight="bold")

    metrics = [
        ("accuracy", "Accuracy",  "tab:blue"),
        ("auc",      "AUC-ROC",   "tab:green"),
        ("loss",     "Loss",      "tab:red"),
        ("precision","Precision", "tab:orange"),
    ]
    for ax, (m, title, color) in zip(axes.flat, metrics):
        ax.plot(history.history[m],         label=f"Train", color=color,         linewidth=2)
        ax.plot(history.history[f"val_{m}"],label=f"Val",   color=color, linestyle="--", linewidth=2)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_dir}/training_curves.png")

    # 2. Confusion matrix
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax, annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_dir}/confusion_matrix.png")

    # 3. ROC curve
    y_true = data["y_test_int"]
    y_score = results["y_pred_ad"]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = results["auc"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc_val:.4f}")
    axes[0].plot([0,1],[0,1], "k--", lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color="steelblue")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3)

    # 4. Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    axes[1].plot(recall_vals, precision_vals, color="darkorange", lw=2, label=f"AP = {ap:.4f}")
    axes[1].fill_between(recall_vals, precision_vals, alpha=0.1, color="darkorange")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_dir}/roc_pr_curves.png")

    # 5. Feature importance via attention weights (top 15)
    if "feature_importances" in results:
        fi = results["feature_importances"]
        names = data["feature_names"]
        idx = np.argsort(fi)[-15:]
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["tomato" if fi[i] > np.median(fi) else "steelblue" for i in idx]
        ax.barh([names[i] for i in idx], fi[idx], color=colors)
        ax.set_xlabel("Mean Attention Score")
        ax.set_title("Top 15 Most Influential Features (Attention)", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out_dir}/feature_importance.png")


# ─────────────────────────────────────────────
# Compute Feature Importance from Attention
# ─────────────────────────────────────────────

def get_feature_importances(model, data: dict) -> np.ndarray:
    """
    Approximate feature importance by multiplying mean attention weights
    (across time) with mean absolute feature values across the test set.
    """
    try:
        from model import build_interpretable_model
        interp = build_interpretable_model(model)
        _, attn_weights = interp.predict(data["X_test"], batch_size=32, verbose=0)
        # attn_weights: (n_samples, T)
        # X_test:       (n_samples, T, n_features)
        mean_attn = attn_weights.mean(axis=0)                  # (T,)
        weighted_x = (data["X_test"] * mean_attn[None, :, None])  # (n, T, F)
        importance = np.abs(weighted_x).mean(axis=(0, 1))      # (F,)
        return importance
    except Exception as e:
        print(f"Could not compute attention importances: {e}")
        return None


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    out_dir = CFG["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df = load_dataset(args.data)
    print_eda_summary(df)

    # ── 2. Preprocess ────────────────────────────────────────
    print("[2/5] Preprocessing...")
    data = preprocess(
        df,
        sequence_length=CFG["sequence_length"],
        apply_smote=True,
    )

    # ── 3. Build model ───────────────────────────────────────
    print("[3/5] Building model...")
    model = build_model(
        sequence_length=CFG["sequence_length"],
        n_features     =data["n_features"],
        n_classes      =data["n_classes"],
        lstm_units_1   =CFG["lstm_units_1"],
        lstm_units_2   =CFG["lstm_units_2"],
        attention_units=CFG["attention_units"],
        dense_units    =CFG["dense_units"],
        dropout_rate   =CFG["dropout_rate"],
        rec_dropout    =CFG["rec_dropout"],
        l2_reg         =CFG["l2_reg"],
        learning_rate  =args.lr,
    )
    model.summary()

    # ── 4. Train ─────────────────────────────────────────────
    print("\n[4/5] Training...")
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs          =args.epochs,
        batch_size      =CFG["batch_size"],
        class_weight    =data["class_weights"],
        callbacks       =get_callbacks(out_dir, CFG),
        verbose         =1,
    )

    # ── 5. Evaluate ──────────────────────────────────────────
    print("\n[5/5] Evaluating...")
    results = evaluate(model, data)

    # Attention-based feature importances
    fi = get_feature_importances(model, data)
    if fi is not None:
        results["feature_importances"] = fi
        top5_idx = np.argsort(fi)[::-1][:5]
        print("Top 5 most influential features (attention):")
        for i in top5_idx:
            print(f"  {data['feature_names'][i]:<35} {fi[i]:.4f}")

    # ── Save artifacts ───────────────────────────────────────
    model.save(f"{out_dir}/checkpoints/final_model.keras")
    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(data["scaler"], f)
    with open(f"{out_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(data["feature_names"], f)

    # ── Plots ────────────────────────────────────────────────
    plot_all(history, results, data, f"{out_dir}/plots")

    print("\n✓ All done.")
    print(f"  Model     → {out_dir}/checkpoints/final_model.keras")
    print(f"  Scaler    → {out_dir}/scaler.pkl")
    print(f"  Plots     → {out_dir}/plots/")
    print(f"\n  Final AUC-ROC  : {results['auc']:.4f}")
    print(f"  Final Accuracy : {results['accuracy']*100:.2f}%")
    print(f"  Sensitivity    : {results['recall']:.4f}  (fraction of AD correctly found)")
    print(f"  Specificity    : {results['specificity']:.4f}  (fraction of Healthy correctly found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to dataset file (.xlsx or .csv)")
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
