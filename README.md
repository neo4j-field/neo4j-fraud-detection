# IEEE-CIS Fraud Detection - Graph-Enhanced ML Demo

A complete fraud detection project comparing traditional ML against graph-enhanced ML using Neo4j. Built on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset.

## Key Results

| Metric | Tabular Baseline | Graph-Enhanced | Delta |
|---|---|---|---|
| **ROC-AUC** | 0.9208 | 0.9536 | +0.0328 |
| **PR-AUC** | 0.5966 | 0.7247 | **+21.5%** |
| F1 (fraud) | 0.5790 | 0.6906 | +19.3% |
| Precision | 0.6689 | 0.7914 | +18.3% |
| Recall | 0.5103 | 0.6126 | +20.0% |

Graph features improve PR-AUC by **21.5%** - 376 more fraud cases caught and 335 fewer false alarms on the same validation set.

---

## Project Structure

```
.
├── README.md
├── .env.example              # Neo4j connection template
├── requirements.txt
├── src/
│   ├── baseline/
│   │   └── train_tabular_baseline.py     # Phase 3: LightGBM on raw tabular data
│   ├── graph/
│   │   ├── create_schema.py              # Phase 4: Neo4j constraints + indexes
│   │   ├── load_graph.py                 # Phase 4: Batch graph loader
│   │   ├── generate_graph_features.py    # Phase 5: WCC + entity fraud rates
│   │   └── generate_embeddings.py        # Phase 5: FastRP node embeddings
│   └── hybrid/
│       └── train_graph_enhanced_model.py # Phase 6: LightGBM + graph features
├── docs/
│   ├── graph_model_options.md            # Model A vs Model B design analysis
│   ├── graph_model_final.md              # Chosen graph model + Cypher schema
│   └── load_mapping.md                   # CSV column → Neo4j property mapping
├── reports/
│   ├── data_profile.md                   # Phase 1: Dataset analysis
│   ├── baseline_metrics.md               # Phase 3: Tabular model results
│   ├── graph_load_validation.md          # Phase 4: Graph load sanity checks
│   ├── graph_features.md                 # Phase 5: Feature signal analysis
│   ├── hybrid_metrics.md                 # Phase 6: Graph-enhanced results
│   └── model_comparison.md              # Phase 6: Side-by-side comparison
├── artifacts/
│   ├── baseline_confusion_matrix.png
│   ├── baseline_feature_importance.png
│   ├── hybrid_confusion_matrix.png
│   ├── hybrid_feature_importance.png
│   ├── baseline_val_predictions.parquet
│   ├── graph_features.parquet            # 590K rows × 17 graph features
│   └── transaction_embeddings.parquet   # 590K rows × 64 FastRP dims
└── demo/
    ├── demo_queries.cypher               # 10 Cypher queries for Neo4j Browser
    └── demo_talk_track.md               # Presentation narrative
```

---

## Setup

### 1. Prerequisites

- Python 3.10+
- Neo4j Aura Pro instance (or compatible Neo4j 5.x)
- Kaggle account with IEEE-CIS competition joined

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your Neo4j Aura credentials
```

### 4. Download dataset

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip
```

---

## Running the Pipeline

Run phases in order:

```bash
# Phase 3: Train tabular baseline
python src/baseline/train_tabular_baseline.py

# Phase 4: Load graph into Neo4j
python src/graph/create_schema.py
python src/graph/load_graph.py

# Phase 5: Generate graph features and embeddings
python src/graph/generate_graph_features.py
python src/graph/generate_embeddings.py

# Phase 6: Train graph-enhanced model
python src/hybrid/train_graph_enhanced_model.py
```

---

## Graph Model

![Fraud Detection Graph Data Model](docs/Fraud_Datamodel_Graph_DB.png)

The chosen graph model connects transactions to seven shared entity types:

```
(Transaction)-[:USED_CARD]-------->(Card)
(Transaction)-[:PAYER_EMAIL]------>(EmailDomain)
(Transaction)-[:RECIPIENT_EMAIL]->(EmailDomain)
(Transaction)-[:BILLED_TO]------->(BillingAddress)
(Transaction)-[:USED_DEVICE]----->(Device)
```

**Graph statistics after loading:**
- 590,540 Transaction nodes (20,663 fraud = 3.5%)
- 13,553 Card nodes
- 60 EmailDomain nodes
- 437 BillingAddress nodes
- 1,457 Device nodes
- ~1.87M relationships

---

## Why Graph?

Fraudsters reuse infrastructure: the same card, device, or email domain across multiple transactions. A tabular model sees each transaction in isolation and misses these connections. The graph makes shared identity explicit and queryable.

Key graph signals found:
- **Card 9633**: 742 fraud transactions - a systematically exploited card
- **card_fraud_rate**: Fraud transactions come from cards with 5× higher fraud rates than legitimate transactions
- **device_fraud_rate**: Identity-linked fraud transactions use devices with 3× higher fraud rates

---

## Demo

Open Neo4j Browser at your Aura instance and run queries from `demo/demo_queries.cypher`.
See `demo/demo_talk_track.md` for a full presentation narrative.

---

## Technical Notes

- **Class imbalance**: 3.5% fraud rate. Model uses `scale_pos_weight` (~27.4) to compensate.
- **Primary metric**: PR-AUC (precision-recall area under curve) - more informative than accuracy or ROC-AUC for imbalanced fraud.
- **Train/val split**: Temporal (TransactionDT ≤ day 145 = train, > day 145 = val). Simulates deployment.
- **Graph features**: Entity fraud rates computed on training data only. No leakage to validation.
- **Embeddings**: FastRP 64-dim, 2-hop neighborhood, trained on graph structure only (no fraud label).
- **WCC**: Weakly connected components had no discriminative power - 60 email domain hubs connect nearly all transactions into one giant component.
