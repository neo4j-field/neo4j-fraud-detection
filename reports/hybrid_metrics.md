# Graph-Enhanced Model Metrics

## Experiment Setup
- **Model**: LightGBM (GBDT)
- **Features**: Tabular + 22 graph aggregate features + 64 FastRP embedding dims
- **Graph V2 additions**: recipient email fraud rate, OS+browser fingerprint fraud rate, proxy type signal, prev_card_is_fraud (temporal card chain)
- **Split**: Temporal (TransactionDT ≤ day 145 = train, > day 145 = validation)
- **Imbalance handling**: scale_pos_weight = 27.4
- **Threshold**: 0.79 (F1-optimized)

## Results

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.9536 |
| **PR-AUC** | 0.7247 |
| Precision (fraud class) | 0.7914 |
| Recall (fraud class) | 0.6126 |
| F1 (fraud class) | 0.6906 |
| Accuracy | 0.9814 |
| Threshold | 0.79 |
| Fraud support (val) | 3678 |

## Confusion Matrix

| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 104089 | 594 |
| **Actual Fraud** | 1425 | 2253 |
