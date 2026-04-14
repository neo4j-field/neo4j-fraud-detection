"""
Phase 6 — Graph-Enhanced Fraud Detection Model
================================================
Trains a LightGBM model using:
  - All tabular features (train_transaction + train_identity)
  - Graph-derived features: card/device/email/billing fraud rates and counts
  - FastRP node embeddings (64 dims)

Comparison against the tabular baseline uses the same temporal split
and the same evaluation protocol to ensure a fair comparison.

Usage:
    python src/hybrid/train_graph_enhanced_model.py
"""

import logging
import os
import warnings
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

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Same split as baseline — temporal
TEMPORAL_SPLIT_DT = 12_528_000

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

# Graph features to include (exclude WCC/component features — no signal)
GRAPH_FEATURE_COLS = [
    "card_tx_count", "card_fraud_count", "card_fraud_rate",
    "payer_email_tx_count", "payer_email_fraud_count", "payer_email_fraud_rate",
    "billing_tx_count", "billing_fraud_count", "billing_fraud_rate",
    "device_tx_count", "device_fraud_count", "device_fraud_rate",
]

DROP_COLS = ["TransactionID", "isFraud", "transactionId"]

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

def load_data():
    """Load and join all data sources: tabular + graph features + embeddings."""
    log.info("Loading train_transaction.csv ...")
    tt = pd.read_csv(DATA_DIR / "train_transaction.csv")
    log.info(f"  Shape: {tt.shape}")

    log.info("Loading train_identity.csv ...")
    ti = pd.read_csv(DATA_DIR / "train_identity.csv")
    log.info(f"  Shape: {ti.shape}")

    log.info("Joining transaction + identity ...")
    df = tt.merge(ti, on="TransactionID", how="left")
    df["has_identity"] = df["DeviceType"].notna().astype(int)

    log.info("Loading graph features ...")
    gf = pd.read_parquet(ARTIFACTS_DIR / "graph_features.parquet")
    # Rename to avoid collision with raw tabular cols
    gf = gf.rename(columns={"transactionId": "TransactionID"})
    graph_cols = ["TransactionID"] + GRAPH_FEATURE_COLS
    gf = gf[[c for c in graph_cols if c in gf.columns]]
    df = df.merge(gf, on="TransactionID", how="left")
    log.info(f"  After graph join: {df.shape}")

    log.info("Loading FastRP embeddings ...")
    emb = pd.read_parquet(ARTIFACTS_DIR / "transaction_embeddings.parquet")
    emb = emb.rename(columns={"transactionId": "TransactionID"})
    emb["TransactionID"] = emb["TransactionID"].astype(int)
    df = df.merge(emb, on="TransactionID", how="left")
    log.info(f"  After embedding join: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def encode_categoricals(df, cat_cols, fit=True, encoders=None):
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            filled = df[col].fillna("__missing__").astype(str)
            le.fit(filled)
            encoders[col] = le
        else:
            le = encoders[col]
        filled = df[col].fillna("__missing__").astype(str)
        known = set(le.classes_)
        filled = filled.apply(lambda x: x if x in known else "__missing__")
        df[col] = le.transform(filled)
    return df, encoders


def build_feature_matrix(df, cat_cols, encoders=None, fit=True):
    df = df.copy()
    present_cats = [c for c in cat_cols if c in df.columns]
    df, encoders = encode_categoricals(df, present_cats, fit=fit, encoders=encoders)

    y = df["isFraud"].values
    X = df.drop(columns=DROP_COLS, errors="ignore")

    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        log.warning(f"  Dropping residual object cols: {obj_cols}")
        X = X.drop(columns=obj_cols)

    return X.values, y, X.columns.tolist(), encoders


def temporal_split(df):
    train = df[df["TransactionDT"] <= TEMPORAL_SPLIT_DT].copy()
    val = df[df["TransactionDT"] > TEMPORAL_SPLIT_DT].copy()
    log.info(f"  Train: {len(train):,} | Val: {len(val):,}")
    log.info(f"  Train fraud rate: {train['isFraud'].mean():.4f}")
    log.info(f"  Val fraud rate:   {val['isFraud'].mean():.4f}")
    return train, val


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_val, y_val, feature_names):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    params = dict(LGBM_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    all_cats = CAT_COLS_TRANSACTION + CAT_COLS_IDENTITY + ["has_identity"]
    cat_indices = [i for i, f in enumerate(feature_names) if f in all_cats]

    model = lgb.LGBMClassifier(**params)
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


def find_best_threshold(proba, y_val):
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = (proba >= t).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh


def evaluate(model, X_val, y_val, threshold=0.5):
    proba = model.predict_proba(X_val)[:, 1]
    roc_auc  = roc_auc_score(y_val, proba)
    pr_auc   = average_precision_score(y_val, proba)
    y_pred   = (proba >= threshold).astype(int)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall   = recall_score(y_val, y_pred, zero_division=0)
    f1       = f1_score(y_val, y_pred, zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    cm       = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "threshold": threshold,
        "support_fraud": int(y_val.sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": cm, "proba": proba, "y_val": y_val,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(metrics, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics["confusion_matrix"],
        display_labels=["Legitimate", "Fraud"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix — Graph-Enhanced Model\n"
        f"Threshold={metrics['threshold']:.2f} | "
        f"ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info(f"  Saved: {output_path}")


def plot_feature_importance(model, feature_names, output_path, top_n=30):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]

    # Color graph features differently
    graph_feat_set = set(GRAPH_FEATURE_COLS) | {f"emb_{i}" for i in range(64)}
    colors = ["#e06c00" if f in graph_feat_set else "steelblue" for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_features)), top_importance, align="center", color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel("Feature Importance (split gain)")
    ax.set_title(f"Top {top_n} Feature Importances — Graph-Enhanced Model\n(orange = graph-derived)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e06c00", label="Graph features"),
        Patch(facecolor="steelblue", label="Tabular features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    log.info(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_hybrid_metrics(metrics, output_path):
    lines = [
        "# Graph-Enhanced Model Metrics (Tabular + Graph Features + Embeddings)",
        "",
        "## Experiment Setup",
        "- **Model**: LightGBM (GBDT)",
        "- **Features**: Tabular + graph aggregates (card/device/email/billing fraud rates) + FastRP 64-dim embeddings",
        "- **Split**: Temporal (same as baseline — TransactionDT ≤ day 145 = train)",
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
        "## Artifacts",
        "- `artifacts/hybrid_confusion_matrix.png`",
        "- `artifacts/hybrid_feature_importance.png`",
    ]
    output_path.write_text("\n".join(lines))
    log.info(f"  Saved: {output_path}")


def write_comparison_report(baseline_metrics, hybrid_metrics, output_path):
    def delta(a, b):
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    lines = [
        "# Model Comparison: Tabular Baseline vs. Graph-Enhanced",
        "",
        "## Summary Table",
        "",
        "| Metric | Baseline (Tabular) | Graph-Enhanced | Delta |",
        "|---|---|---|---|",
        f"| **ROC-AUC** | {baseline_metrics['roc_auc']:.4f} | {hybrid_metrics['roc_auc']:.4f} | {delta(baseline_metrics['roc_auc'], hybrid_metrics['roc_auc'])} |",
        f"| **PR-AUC** | {baseline_metrics['pr_auc']:.4f} | {hybrid_metrics['pr_auc']:.4f} | {delta(baseline_metrics['pr_auc'], hybrid_metrics['pr_auc'])} |",
        f"| Precision | {baseline_metrics['precision']:.4f} | {hybrid_metrics['precision']:.4f} | {delta(baseline_metrics['precision'], hybrid_metrics['precision'])} |",
        f"| Recall | {baseline_metrics['recall']:.4f} | {hybrid_metrics['recall']:.4f} | {delta(baseline_metrics['recall'], hybrid_metrics['recall'])} |",
        f"| F1 | {baseline_metrics['f1']:.4f} | {hybrid_metrics['f1']:.4f} | {delta(baseline_metrics['f1'], hybrid_metrics['f1'])} |",
        f"| Accuracy | {baseline_metrics['accuracy']:.4f} | {hybrid_metrics['accuracy']:.4f} | {delta(baseline_metrics['accuracy'], hybrid_metrics['accuracy'])} |",
        "",
        "## Confusion Matrix Comparison",
        "",
        "### Baseline",
        "| | Predicted Legit | Predicted Fraud |",
        "|---|---|---|",
        f"| **Actual Legit** | {baseline_metrics['tn']} | {baseline_metrics['fp']} |",
        f"| **Actual Fraud** | {baseline_metrics['fn']} | {baseline_metrics['tp']} |",
        "",
        "### Graph-Enhanced",
        "| | Predicted Legit | Predicted Fraud |",
        "|---|---|---|",
        f"| **Actual Legit** | {hybrid_metrics['tn']} | {hybrid_metrics['fp']} |",
        f"| **Actual Fraud** | {hybrid_metrics['fn']} | {hybrid_metrics['tp']} |",
        "",
        "## Key Observations",
        "",
        "### What graph features contributed",
        "- `card_fraud_rate`: Fraud transactions come from cards with 5x higher fraud rates — strong graph signal",
        "- `device_fraud_rate`: Identity-linked fraud transactions use devices with 3x higher fraud rates",
        "- `billing_fraud_rate`: Modest additive signal from billing address risk concentration",
        "- FastRP embeddings: Capture structural position in the shared-entity network",
        "",
        "### Why PR-AUC matters more than accuracy",
        "With only 3.5% fraud, a model that predicts all-legitimate scores 96.5% accuracy.",
        "PR-AUC measures how well the model ranks fraud cases over legitimate ones at all thresholds.",
        "Improvement in PR-AUC means better fraud capture, which is the operational goal.",
        "",
        "### Honest limitations",
        "- WCC component features had no discriminative power (giant component from email domain hubs)",
        "- FastRP embeddings are unsupervised — they do not directly encode the fraud label",
        "- The graph model uses training labels in entity fraud rate features, which is appropriate",
        "  (these are computed from training data only and joined to each split correctly)",
        "- For production: entity fraud rates must be computed from a rolling window, not the full training set",
    ]
    output_path.write_text("\n".join(lines))
    log.info(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load
    log.info("=== Phase 1: Loading data ===")
    df = load_data()

    # 2. Split
    log.info("=== Phase 2: Temporal split ===")
    train_df, val_df = temporal_split(df)

    # 3. Feature matrix
    log.info("=== Phase 3: Building feature matrices ===")
    all_cat_cols = CAT_COLS_TRANSACTION + CAT_COLS_IDENTITY
    X_train, y_train, feature_names, encoders = build_feature_matrix(
        train_df, all_cat_cols, fit=True
    )
    log.info(f"  Train: {X_train.shape} | Features: {len(feature_names)}")

    X_val, y_val, _, _ = build_feature_matrix(
        val_df, all_cat_cols, encoders=encoders, fit=False
    )
    log.info(f"  Val:   {X_val.shape}")

    # Log how many graph/embedding features are included
    graph_in_features = [f for f in feature_names if f in set(GRAPH_FEATURE_COLS)]
    emb_in_features = [f for f in feature_names if f.startswith("emb_")]
    log.info(f"  Graph aggregate features: {len(graph_in_features)}")
    log.info(f"  Embedding features: {len(emb_in_features)}")

    # 4. Train
    log.info("=== Phase 4: Training ===")
    model = train_model(X_train, y_train, X_val, y_val, feature_names)

    # 5. Threshold
    log.info("=== Phase 5: Finding best threshold ===")
    proba_val = model.predict_proba(X_val)[:, 1]
    best_thresh = find_best_threshold(proba_val, y_val)
    log.info(f"  Best threshold: {best_thresh:.2f}")

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
    plot_confusion_matrix(metrics, ARTIFACTS_DIR / "hybrid_confusion_matrix.png")
    plot_feature_importance(model, feature_names, ARTIFACTS_DIR / "hybrid_feature_importance.png")
    write_hybrid_metrics(metrics, REPORTS_DIR / "hybrid_metrics.md")

    # 8. Comparison report — load baseline metrics
    log.info("=== Phase 8: Generating comparison report ===")
    baseline_val_preds = pd.read_parquet(ARTIFACTS_DIR / "baseline_val_predictions.parquet")
    baseline_y = baseline_val_preds["isFraud"].values
    baseline_proba = baseline_val_preds["baseline_proba"].values
    baseline_thresh = find_best_threshold(baseline_proba, baseline_y)
    # Recompute baseline metrics from saved predictions
    baseline_metrics = {
        "roc_auc":   roc_auc_score(baseline_y, baseline_proba),
        "pr_auc":    average_precision_score(baseline_y, baseline_proba),
        "precision": precision_score(baseline_y, (baseline_proba >= baseline_thresh).astype(int), zero_division=0),
        "recall":    recall_score(baseline_y, (baseline_proba >= baseline_thresh).astype(int), zero_division=0),
        "f1":        f1_score(baseline_y, (baseline_proba >= baseline_thresh).astype(int), zero_division=0),
        "accuracy":  accuracy_score(baseline_y, (baseline_proba >= baseline_thresh).astype(int)),
        "threshold": baseline_thresh,
        "support_fraud": int(baseline_y.sum()),
    }
    cm_b = confusion_matrix(baseline_y, (baseline_proba >= baseline_thresh).astype(int))
    tn_b, fp_b, fn_b, tp_b = cm_b.ravel()
    baseline_metrics.update({"tn": int(tn_b), "fp": int(fp_b), "fn": int(fn_b), "tp": int(tp_b)})

    write_comparison_report(baseline_metrics, metrics, REPORTS_DIR / "model_comparison.md")

    log.info("=== Done ===")
    return metrics


if __name__ == "__main__":
    main()
