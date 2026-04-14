# Baseline Model Metrics - Tabular Only (LightGBM)

## Experiment Setup
- **Model**: LightGBM (GBDT)
- **Features**: All tabular features from train_transaction + train_identity (joined)
- **Split**: Temporal (TransactionDT ≤ day 145 = train, > day 145 = validation)
- **Imbalance handling**: scale_pos_weight proportional to class ratio
- **Threshold**: Optimized for F1 on validation set

## Results

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.9208 |
| **PR-AUC** | 0.5966 |
| Precision (fraud class) | 0.6689 |
| Recall (fraud class) | 0.5103 |
| F1 (fraud class) | 0.5790 |
| Accuracy | 0.9748 |
| Threshold | 0.59 |
| Fraud support (val) | 3678 |

## Confusion Matrix

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 103754 | 929 |
| **Actual Fraud** | 1801 | 1877 |

## Notes
- Accuracy is reported for completeness but is NOT the primary metric due to class imbalance (~3.5% fraud)
- PR-AUC is the most informative single metric for imbalanced fraud detection
- This result serves as the baseline for comparison with the graph-enhanced model

## Artifacts
- `artifacts/baseline_confusion_matrix.png`
- `artifacts/baseline_feature_importance.png`