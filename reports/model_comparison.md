# Model Comparison: Tabular Baseline vs. Graph-Enhanced

## Summary Table

| Metric | Baseline (Tabular) | Graph-Enhanced | Delta |
|---|---|---|---|
| **ROC-AUC** | 0.9208 | 0.9536 | +0.0328 |
| **PR-AUC** | 0.5966 | 0.7247 | **+0.1281 (+21.5%)** |
| Precision | 0.6689 | 0.7914 | +0.1225 |
| Recall | 0.5103 | 0.6126 | +0.1023 |
| F1 | 0.5790 | 0.6906 | +0.1116 |
| Accuracy | 0.9748 | 0.9814 | +0.0066 |

## Confusion Matrix Comparison

### Baseline (Tabular Only)
| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 103,754 | 929 |
| **Actual Fraud** | 1,801 | 1,877 |

### Graph-Enhanced
| | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 104,089 | 594 |
| **Actual Fraud** | 1,425 | 2,253 |

**376 more fraud cases caught. 335 fewer false alarms. Same validation set.**

## What Drove the Improvement

| Graph feature | Fraud mean | Legit mean | Ratio |
|---|---|---|---|
| `prev_card_is_fraud` | 0.344 | 0.018 | **18.7x** |
| `card_fraud_rate` | 0.153 | 0.031 | 5.0x |
| `device_fraud_rate` | 0.206 | 0.062 | 3.3x |
| `os_browser_fraud_rate` | 0.143 | 0.074 | 1.9x |
| `is_proxy` | 0.021 | 0.008 | 2.5x |
| `recip_email_fraud_rate` | 0.106 | 0.080 | 1.3x |

## Key Observations

- **`prev_card_is_fraud` is the breakthrough feature**: when the previous transaction on the same card was fraud, 34% of the time the current one is too. This temporal chain — created via `PREV_ON_CARD` edges — is completely invisible to tabular models that treat each row independently.
- **`os_browser_fraud_rate`** adds a soft device fingerprint layer even for transactions without a specific `DeviceInfo` entry.
- **`recip_email_fraud_rate`** was already in the graph (RECIPIENT_EMAIL relationships) but missing from the V1 feature set — a free gain.
- **FastRP embeddings** benefit from the richer V2 graph: `PREV_ON_CARD` edges create direct transaction-to-transaction propagation, giving fraud clusters a tighter spatial signature in embedding space.

## Honest Limitations

- Entity fraud rates use training data labels. In production: compute from a rolling lookback window to prevent lookahead bias.
- `PREV_ON_CARD` edges encode training-set fraud labels into the graph structure — the model must be periodically retrained as new fraud patterns emerge.
- Cold start: new cards, devices, or email domains not seen in training have no entity fraud rate (defaults to -1). The model handles this gracefully but has weaker signal for genuinely new entities.
