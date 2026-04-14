# Graph Features Report - Phase 5

## Features Generated

### Entity-Level Aggregate Features (12 features)

Computed using entity-first aggregation pattern in Cypher: aggregate stats per entity, then join back to transactions.

| Feature | Description | Coverage |
|---|---|---|
| `card_tx_count` | Total transactions using the same card (card1) | 590,540 (100%) |
| `card_fraud_count` | Fraud transactions on the same card | 590,540 |
| `card_fraud_rate` | Fraud rate on the card | 590,540 |
| `payer_email_tx_count` | Transactions with the same payer email domain | 496,084 (~84%) |
| `payer_email_fraud_count` | Fraud transactions on payer domain | 496,084 |
| `payer_email_fraud_rate` | Fraud rate on payer email domain | 496,084 |
| `billing_tx_count` | Transactions at the same billing address | 524,834 (~89%) |
| `billing_fraud_count` | Fraud transactions at billing address | 524,834 |
| `billing_fraud_rate` | Fraud rate at billing address | 524,834 |
| `device_tx_count` | Transactions using the same device | 118,666 (~20%) |
| `device_fraud_count` | Fraud transactions on same device | 118,666 |
| `device_fraud_rate` | Fraud rate on device | 118,666 |

Missing values (transactions without entity link) are filled: counts → 0, rates → -1.0.

### Connected Component Features (4 features)

Via WCC (Weakly Connected Components) on the full bipartite graph.

| Feature | Description |
|---|---|
| `componentId` | Component ID |
| `component_size` | Number of transactions in the same component |
| `component_fraud_count` | Fraud transactions in component |
| `component_fraud_rate` | Fraud rate in component |

**Important finding**: WCC components are not useful as ML features in this dataset. Because EmailDomain nodes (only 60 values) connect nearly all transactions into one giant component, most transactions share the same component stats. `component_fraud_rate` shows fraud≈0.0351 vs legit≈0.0350 - essentially no discrimination. These 4 features will be excluded from the ML feature set.

### Node Embeddings (64 features)

FastRP embeddings (64 dimensions) for all 590,540 Transaction nodes.

- **Graph projection**: All 5 node labels + 5 relationship types, undirected
- **iterationWeights**: [0.0, 1.0, 1.0] - captures 2-hop neighborhood
- **Embedding quality**: Centroid cosine similarity (fraud vs legit) = 0.9195; L2 distance = 0.1012
- The modest separation is expected - FastRP is unsupervised (doesn't use isFraud). The discriminative power will be unlocked by the downstream LightGBM model, which can find non-linear patterns in embedding space.

## Signal Strength

| Feature | Fraud mean | Legit mean | Ratio |
|---|---|---|---|
| `card_fraud_rate` | 0.1531 | 0.0307 | **5.0x** |
| `device_fraud_rate` | 0.2062 | 0.0621 | **3.3x** |
| `payer_email_fraud_rate` | ~0.04 | ~0.035 | ~1.1x |
| `billing_fraud_rate` | 0.0347 | 0.0244 | 1.4x |
| `component_fraud_rate` | 0.0351 | 0.0350 | ~1.0x (useless) |

**card_fraud_rate** and **device_fraud_rate** are the strongest graph-derived signals. They capture the shared-identity fraud pattern directly: fraudulent transactions disproportionately appear on compromised cards and specific devices.

## Artifacts

- `artifacts/graph_features.parquet` - 590,540 rows × 17 columns
- `artifacts/transaction_embeddings.parquet` - 590,540 rows × 65 columns (transactionId + 64 emb dims)
