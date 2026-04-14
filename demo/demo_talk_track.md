# Demo Talk Track - IEEE-CIS Fraud Detection with Neo4j + Vertex AI

---

## Opening (1-2 min)

**The problem:**
"Credit card fraud costs the global economy over $30 billion annually. Detection is hard for two reasons: first, fraud is extremely rare - only 3.5% of transactions in this dataset. Second, traditional ML models look at each transaction in isolation. But fraud is rarely isolated."

**The dataset:**
"We're using the real IEEE-CIS Fraud Detection dataset from a major payments processor: 590,000 transactions, 394 features. It includes card attributes, billing addresses, email domains, and for 24% of transactions, device and browser information."

---

## Section 1: Why Tabular ML Falls Short (2 min)

"A traditional ML model - even a very strong one like LightGBM - gets a transaction as a row of numbers. It doesn't know that this transaction used the same card as 300 previous fraud transactions. It doesn't know this device has been flagged 50 times. That context is invisible."

**Baseline result:**
"Our tabular LightGBM baseline achieves ROC-AUC 0.92 and PR-AUC 0.60. That's genuinely strong. But watch what happens when we add graph context."

---

## Section 2: The Graph Model (2 min)

**Show the graph data model diagram (Cell 1 in notebook):**

![Graph Data Model](../docs/Fraud_Datamodel_Graph_DB.png)

"We modelled the entire dataset as a bipartite graph in Neo4j. Transaction is the central node. It connects to six types of shared entities:
- The card used (13,553 unique cards)
- Purchaser and recipient email domains (60 unique domains)
- Billing address (437 unique address codes)
- Device fingerprint (1,457 normalized devices)
- OS and browser combination (924 unique fingerprints)
- Proxy type (anonymous, transparent, or none)"

"We also added temporal edges - PREV_ON_CARD - connecting consecutive transactions on the same card within 24 hours. These create a chain that makes fraud sequences directly visible in the graph."

"Importantly: we pre-computed all graph features and node embeddings offline. For this demo, we load them as flat files - no live database connection needed."

---

## Section 3: Vertex AI Notebook Walkthrough (5 min)

**Open `demo/vertex_ai_demo.ipynb` in Vertex AI Workbench.**

### Cell 2 - Setup
"Standard imports and a configuration block. If you're running this locally, USE_GCS is False and we read from the artifacts folder. On Vertex AI you flip that flag and point to your GCS bucket."

### Cell 3-4 - Graph diagram
"Here's the graph model again rendered inline in the notebook."

### Cell 5-6 - Load artifacts
**Run the load cell. Show the output.**
"We load three things: the 22 graph features from Neo4j (11 MB), the 64-dimensional FastRP node embeddings (180 MB), and the raw transaction CSV for tabular features. Merge happens in seconds."

### Cell 7 - Temporal split
"Temporal split at day 145 - validation transactions are always more recent than training transactions. This prevents any leakage from future fraud patterns into the model."

### Cell 8-9 - Graph feature signal
**Run the distribution plot.**
"Look at card_fraud_rate: the fraud distribution is shifted dramatically to the right compared to legitimate transactions - a 5x ratio. Device fraud rate shows 3.3x. And prev_card_is_fraud - when the previous transaction on the same card was fraud, 34% of the time the current one is too. That signal is completely invisible to a row-by-row model."

### Cell 10-11 - Embedding visualization
**Run the PCA plot.**
"These are 64-dimensional FastRP embeddings projected to 2D. FastRP is unsupervised - it doesn't use the fraud label - yet fraud transactions (red) already cluster differently from legitimate ones (blue). The downstream LightGBM model will find the non-linear patterns in this 64-dimensional space."

### Cell 12-13 - Feature prep
"We have 390+ tabular features, 22 graph features, and 64 embedding dimensions - 476 features total for the enhanced model."

### Cell 14-15 - Baseline training
**Run the baseline training cell.**
"Training tabular baseline. LightGBM on 450K training rows with early stopping."

### Cell 16-17 - Graph-enhanced training
**Run the graph-enhanced training cell.**
"Same architecture, same hyperparameters - the only difference is the additional 86 graph-derived features."

### Cell 18-21 - Results
**Run the results cells.**

---

## Section 4: Results (2 min)

**Show the results comparison table and chart:**

| Metric | Tabular Baseline | Graph-Enhanced | Improvement |
|---|---|---|---|
| ROC-AUC | 0.9208 | 0.9536 | +3.6% |
| **PR-AUC** | **0.5966** | **0.7247** | **+21.5%** |
| F1 (fraud) | 0.5790 | 0.6906 | +19.3% |
| Precision | 0.6689 | 0.7914 | +18.3% |
| Recall | 0.5103 | 0.6126 | +20.1% |

"PR-AUC is the metric that matters for imbalanced fraud detection - it captures performance across all operating thresholds. A 21.5% improvement in PR-AUC is substantial. Translating to operational impact: 376 more fraud cases caught per validation period, 335 fewer false alarms. Same model architecture, same compute budget."

**Show the feature importance chart:**
"Look at what features the model ranked highest. Graph features - card_fraud_rate, prev_card_is_fraud, device_fraud_rate - appear in the top 10 despite being new additions. The model immediately learned to trust the graph context over many of the raw Vesta-engineered features."

---

## Section 5: Why This Approach is Powerful (1 min)

"Three things make this graph approach compelling:

1. **Explainability**: You can show an investigator exactly why a transaction was flagged - it shares a card with 742 known fraudulent transactions. That's not a black box.

2. **Network effects**: The graph catches fraud that's invisible to row-by-row models. A new account looks clean in isolation but is immediately suspicious when connected to a known fraud device.

3. **Separation of concerns**: The graph is a pre-computation layer. The ML training runs anywhere - locally, on Vertex AI, on SageMaker. The graph features are just columns in a parquet file."

---

## Honest Limitations (30 sec)

"A few things to be upfront about:
- Entity fraud rates use training data labels. In production you'd use a rolling lookback window to avoid lookahead bias
- PREV_ON_CARD temporal edges encode training-set fraud labels into the graph - the model needs periodic retraining as new fraud patterns emerge
- Cold start: new cards, devices, or email domains not seen in training have no entity fraud rate (defaults to -1). The model handles this gracefully but has weaker signal for genuinely new entities"

---

## Closing (30 sec)

"The key takeaway: tabular models and graph models are not competitors. The graph adds a relational intelligence layer that makes every downstream model better - regardless of where that model runs. For fraud detection, where the enemy is organized, connected, and reusing infrastructure - the graph is not optional. It's the right tool."

---

## Backup Questions

**Q: Does the graph model always win?**
A: Not necessarily. If fraud is completely random with no entity reuse, graph features add noise. The improvement here is genuine because this dataset has real shared-identity fraud patterns. Always validate with PR-AUC on a holdout set.

**Q: How do you prevent leakage from the fraud labels?**
A: Entity fraud rates are computed on the training split only and joined to validation separately. The temporal split ensures validation transactions are always more recent than training transactions - simulating real deployment.

**Q: How scalable is this in production?**
A: Neo4j Aura handles 590K nodes and 1.87M edges comfortably. For real-time scoring, you'd precompute entity fraud rates in Neo4j and expose them via a lookup API, rather than running full graph queries per transaction.

**Q: Why FastRP and not GraphSAGE?**
A: FastRP is available natively in GDS, runs in 3 minutes on this data, and requires no training loop. GraphSAGE would allow supervised embedding with the fraud label, potentially stronger embeddings, but requires significant additional setup. FastRP is the right pragmatic choice for a first version.

**Q: Can this run fully on Vertex AI without Neo4j?**
A: The graph feature extraction and embedding computation require Neo4j GDS - that's a one-time offline step. Once those artifacts are in GCS, all ML training and inference runs entirely on Vertex AI or any other platform. The graph and the ML are decoupled by design.
