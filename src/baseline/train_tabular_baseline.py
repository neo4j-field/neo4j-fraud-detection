"""
Phase 3 — Tabular Baseline ML Experiment
=========================================
Trains a LightGBM classifier on the raw tabular features from the IEEE-CIS
fraud detection dataset (transaction + identity joined), with no graph features.

Usage:
    python src/baseline/train_tabular_baseline.py

Outputs:
    artifacts/baseline_confusion_matrix.png
    artifacts/baseline_feature_importance.png
    reports/baseline_metrics.md
"""

import os
import sys
import warnings
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# String-typed columns in the raw CSV (pandas reads them as str/object)
CAT_COLS_TRANSACTION = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

CAT_COLS_IDENTITY = [
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]

# Columns to drop (leakage risk or pure keys)
DROP_COLS = ["TransactionID", "isFraud"]

# Temporal split threshold: TransactionDT is in seconds.
# The training data covers ~182 days (0 to 15,811,131 seconds).
# We use the last ~20% (~day 145 onward) as validation.
# Day 145 = 145 * 86400 = 12,528,000 seconds
TEMPORAL_SPLIT_DT = 12_528_000

# LightGBM parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 1000,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_join(data_dir: Path) -> pd.DataFrame:
    """Load train_transaction and train_identity, join on TransactionID."""
    log.info("Loading train_transaction.csv ...")
    tt = pd.read_csv(data_dir / "train_transaction.csv")
    log.info(f"  train_transaction shape: {tt.shape}")

    log.info("Loading train_identity.csv ...")
    ti = pd.read_csv(data_dir / "train_identity.csv")
    log.info(f"  train_identity shape: {ti.shape}")

    log.info("Joining on TransactionID (left join) ...")
    df = tt.merge(ti, on="TransactionID", how="left")
    log.info(f"  Joined shape: {df.shape}")

    # Add indicator: whether identity data is available
    df["has_identity"] = df["DeviceType"].notna().astype(int)
    log.info(f"  Transactions with identity: {df['has_identity'].sum()} ({df['has_identity'].mean()*100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame, cat_cols: list, fit: bool = True,
                         encoders: dict = None) -> tuple:
    """
    Label-encode categorical columns. NaN becomes its own category (-1).
    Returns (transformed_df, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            # Fill NaN with a sentinel before encoding
            filled = df[col].fillna("__missing__").astype(str)
            le.fit(filled)
            encoders[col] = le
        else:
            le = encoders[col]

        filled = df[col].fillna("__missing__").astype(str)
        # Handle unseen values in validation
        known = set(le.classes_)
        filled = filled.apply(lambda x: x if x in known else "__missing__")
        df[col] = le.transform(filled)

    return df, encoders


def build_feature_matrix(df: pd.DataFrame, cat_cols: list, encoders: dict = None,
                          fit: bool = True) -> tuple:
    """
    Build X, y from joined dataframe.
    Returns (X, y, feature_names, encoders).
    """
    df = df.copy()

    # All categorical columns present in df
    present_cats = [c for c in cat_cols if c in df.columns]
    df, encoders = encode_categoricals(df, present_cats, fit=fit, encoders=encoders)

    y = df["isFraud"].values
    X = df.drop(columns=DROP_COLS, errors="ignore")

    # Drop remaining object columns that we didn't explicitly encode
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        log.warning(f"  Dropping residual object columns: {obj_cols}")
        X = X.drop(columns=obj_cols)

    feature_names = X.columns.tolist()
    return X.values, y, feature_names, encoders


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, split_dt: int) -> tuple:
    """Split df into train/val based on TransactionDT threshold."""
    train = df[df["TransactionDT"] <= split_dt].copy()
    val = df[df["TransactionDT"] > split_dt].copy()
    log.info(f"  Train rows: {len(train)} | Val rows: {len(val)}")
    log.info(f"  Train fraud rate: {train['isFraud'].mean():.4f}")
    log.info(f"  Val fraud rate:   {val['isFraud'].mean():.4f}")
    return train, val


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, feature_names: list) -> lgb.LGBMClassifier:
    """Train LightGBM with early stopping."""
    # Compute class weight for imbalanced data
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"  scale_pos_weight = {scale_pos_weight:.2f} (neg={neg}, pos={pos})")

    params = dict(LGBM_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    model = lgb.LGBMClassifier(**params)

    # Identify categorical feature indices for LGBM native handling
    all_cats = CAT_COLS_TRANSACTION + CAT_COLS_IDENTITY + ["has_identity"]
    cat_indices = [i for i, f in enumerate(feature_names) if f in all_cats]

    log.info("  Training LightGBM ...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
        categorical_feature=cat_indices if cat_indices else "auto",
    )

    log.info(f"  Best iteration: {model.best_iteration_}")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_val, y_val, threshold: float = 0.5) -> dict:
    """Compute all evaluation metrics."""
    proba = model.predict_proba(X_val)[:, 1]

    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)

    y_pred = (proba >= threshold).astype(int)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    support_fraud = y_val.sum()

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": threshold,
        "support_fraud": int(support_fraud),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm,
        "proba": proba,
        "y_val": y_val,
    }


def find_best_threshold(proba, y_val) -> float:
    """Find threshold maximizing F1 score for the fraud class."""
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (proba >= t).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = t
    return best_thresh


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(metrics: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=["Legitimate", "Fraud"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix — Tabular Baseline\n"
        f"Threshold={metrics['threshold']:.2f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info(f"  Saved: {output_path}")


def plot_feature_importance(model, feature_names: list, output_path: Path, top_n: int = 30):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_importance, align="center", color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Feature Importance (split gain)")
    ax.set_title(f"Top {top_n} Feature Importances — Tabular Baseline")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_metrics_report(metrics: dict, output_path: Path):
    lines = [
        "# Baseline Model Metrics — Tabular Only (LightGBM)",
        "",
        "## Experiment Setup",
        "- **Model**: LightGBM (GBDT)",
        "- **Features**: All tabular features from train_transaction + train_identity (joined)",
        "- **Split**: Temporal (TransactionDT ≤ day 145 = train, > day 145 = validation)",
        "- **Imbalance handling**: scale_pos_weight proportional to class ratio",
        "- **Threshold**: Optimized for F1 on validation set",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **ROC-AUC** | {metrics['roc_auc']:.4f} |",
        f"| **PR-AUC** | {metrics['pr_auc']:.4f} |",
        f"| Precision (fraud class) | {metrics['precision']:.4f} |",
        f"| Recall (fraud class) | {metrics['recall']:.4f} |",
        f"| F1 (fraud class) | {metrics['f1']:.4f} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Threshold | {metrics['threshold']:.2f} |",
        f"| Fraud support (val) | {metrics['support_fraud']} |",
        "",
        "## Confusion Matrix",
        "",
        "| | Predicted Legit | Predicted Fraud |",
        "|---|---|---|",
        f"| **Actual Legit** | {metrics['tn']} | {metrics['fp']} |",
        f"| **Actual Fraud** | {metrics['fn']} | {metrics['tp']} |",
        "",
        "## Notes",
        "- Accuracy is reported for completeness but is NOT the primary metric due to class imbalance (~3.5% fraud)",
        "- PR-AUC is the most informative single metric for imbalanced fraud detection",
        "- This result serves as the baseline for comparison with the graph-enhanced model",
        "",
        "## Artifacts",
        "- `artifacts/baseline_confusion_matrix.png`",
        "- `artifacts/baseline_feature_importance.png`",
    ]
    output_path.write_text("\n".join(lines))
    log.info(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train tabular baseline model")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    args.artifacts_dir.mkdir(exist_ok=True)
    args.reports_dir.mkdir(exist_ok=True)

    # 1. Load data
    log.info("=== Phase 1: Loading data ===")
    df = load_and_join(args.data_dir)

    # 2. Temporal split
    log.info("=== Phase 2: Temporal split ===")
    train_df, val_df = temporal_split(df, TEMPORAL_SPLIT_DT)

    # 3. Build feature matrices
    log.info("=== Phase 3: Building feature matrices ===")
    all_cat_cols = CAT_COLS_TRANSACTION + CAT_COLS_IDENTITY
    X_train, y_train, feature_names, encoders = build_feature_matrix(
        train_df, all_cat_cols, fit=True
    )
    log.info(f"  Train feature matrix: {X_train.shape}")

    X_val, y_val, _, _ = build_feature_matrix(
        val_df, all_cat_cols, encoders=encoders, fit=False
    )
    log.info(f"  Val feature matrix:   {X_val.shape}")

    # 4. Train
    log.info("=== Phase 4: Training ===")
    model = train_model(X_train, y_train, X_val, y_val, feature_names)

    # 5. Find best threshold
    log.info("=== Phase 5: Finding best threshold ===")
    proba_val = model.predict_proba(X_val)[:, 1]
    best_thresh = find_best_threshold(proba_val, y_val)
    log.info(f"  Best threshold (F1-optimal): {best_thresh:.2f}")

    # 6. Evaluate
    log.info("=== Phase 6: Evaluating ===")
    metrics = evaluate(model, X_val, y_val, threshold=best_thresh)

    log.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    log.info(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall:    {metrics['recall']:.4f}")
    log.info(f"  F1:        {metrics['f1']:.4f}")
    log.info(f"  Accuracy:  {metrics['accuracy']:.4f}")

    # 7. Save artifacts
    log.info("=== Phase 7: Saving artifacts ===")
    plot_confusion_matrix(metrics, args.artifacts_dir / "baseline_confusion_matrix.png")
    plot_feature_importance(model, feature_names, args.artifacts_dir / "baseline_feature_importance.png")
    write_metrics_report(metrics, args.reports_dir / "baseline_metrics.md")

    # 8. Save model predictions for comparison report
    log.info("=== Phase 8: Saving val predictions for comparison ===")
    val_preds = pd.DataFrame({
        "TransactionID": val_df["TransactionID"].values,
        "isFraud": y_val,
        "baseline_proba": proba_val,
    })
    val_preds.to_parquet(args.artifacts_dir / "baseline_val_predictions.parquet", index=False)
    log.info(f"  Saved: {args.artifacts_dir / 'baseline_val_predictions.parquet'}")

    log.info("=== Done ===")
    return metrics


if __name__ == "__main__":
    main()
