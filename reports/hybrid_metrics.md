# Graph-Enhanced Model Metrics (Tabular + Graph Features + Embeddings)

## Experiment Setup
- **Model**: LightGBM (GBDT)
- **Features**: Tabular + graph aggregates (card/device/email/billing fraud rates) + FastRP 64-dim embeddings
- **Split**: Temporal (same as baseline — TransactionDT ≤ day 145 = train)
- **Imbalance handling**: scale_pos_weight proportional to class ratio
- **Threshold**: Optimized for F1 on validation set

## Results

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.9441 |
| **PR-AUC** | 0.6488 |
| Precision (fraud class) | 0.7249 |
| Recall (fraud class) | 0.5351 |
| F1 (fraud class) | 0.6157 |
| Accuracy | 0.9773 |
| Threshold | 0.76 |
| Fraud support (val) | 3678 |

## Confusion Matrix

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 103936 | 747 |
| **Actual Fraud** | 1710 | 1968 |

## Artifacts
- `artifacts/hybrid_confusion_matrix.png`
- `artifacts/hybrid_feature_importance.png`