# IEEE-CIS Fraud Detection — Data Profile Report

Generated during Phase 1 data understanding.

---

## 1. Dataset Overview

| File | Rows | Columns | Size |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | 652 MB |
| `train_identity.csv` | 144,233 | 41 | 25 MB |
| `test_transaction.csv` | 506,691 | 393 | 585 MB |
| `test_identity.csv` | 141,907 | 41 | 25 MB |
| `sample_submission.csv` | 506,691 | 2 | 5.8 MB |

---

## 2. Join Strategy

**Join key**: `TransactionID` (integer, unique per row in both files).

```
train_transaction LEFT JOIN train_identity ON TransactionID
```

- 590,540 transactions total
- 144,233 have identity data (**24.4% coverage**)
- The join is a strict left join: every transaction has a row in `train_transaction`; most do not have a corresponding `train_identity` row
- `test_transaction` and `test_identity` use the same join key and same structure

**Implication**: Identity features must be treated as optional. A NULL identity record is itself informative (anonymous/non-digital transaction). The missing-identity indicator should be kept as an explicit feature.

---

## 3. Target Label

- **Column**: `isFraud` in `train_transaction` only
- **Distribution**:
  | Class | Count | Rate |
  |---|---|---|
  | 0 (legitimate) | 569,877 | 96.50% |
  | 1 (fraud) | 20,663 | **3.50%** |
- **Highly imbalanced**: ~1:27.6 fraud-to-legitimate ratio
- Test set has no labels (Kaggle holdout)
- **Key metric implication**: Accuracy is misleading. PR-AUC, recall, precision, and F1 for the fraud class are the meaningful metrics

---

## 4. Column Taxonomy

### 4.1 Direct Entity / Identity Columns

These are the most valuable for graph modeling — they represent real-world entities that can be shared across transactions:

| Column | Type | Cardinality | Graph value |
|---|---|---|---|
| `card1` | int | 13,553 unique | High — acts as payment instrument fingerprint |
| `card2` | float | 500 unique | Medium — partial card identifier |
| `card4` | str | 4 unique (visa/mc/amex/discover) | Low cardinality — card network |
| `card6` | str | 4 unique (debit/credit) | Low cardinality — card type |
| `addr1` | float | 332 unique | Medium — billing zip/postal area |
| `addr2` | float | 74 unique | Low — country/region code |
| `P_emaildomain` | str | 59 unique | Medium — purchaser email domain |
| `R_emaildomain` | str | 60 unique | Medium — recipient email domain |
| `ProductCD` | str | 5 unique (W/C/H/S/R) | Low — product category |

From `train_identity` (only 24.4% transactions):

| Column | Type | Notes |
|---|---|---|
| `DeviceType` | str | mobile / desktop |
| `DeviceInfo` | str | ~1,000+ unique device strings (Samsung, Windows, iOS, etc.) |
| `id_30` | str | OS version (Windows 10, iOS 11.x, Android 7.x, etc.) |
| `id_31` | str | Browser + version (chrome 63.0, mobile safari, etc.) |
| `id_33` | str | Screen resolution (1920x1080, 1366x768, etc.) |
| `id_23` | str | IP proxy type: IP_PROXY:TRANSPARENT / ANONYMOUS |

### 4.2 Weak Identity / Behavioral Signals

| Column Group | Count | Description |
|---|---|---|
| `C1`–`C14` | 14 cols | Count-like features (e.g., how many addresses on card). Not directly entity-identifiable |
| `D1`–`D15` | 15 cols | Time delta features (e.g., days since last transaction, card activation age) |
| `M1`–`M9` | 9 cols | Match flags: T/F/NaN — whether card/billing/name fields match on file |
| `id_12`, `id_15`, `id_16`, `id_28`, `id_29` | 5 cols | Found/NotFound flags — account/proxy lookup results |
| `id_34`–`id_38` | 5 cols | match_status codes and T/F flags |
| `id_01`–`id_11`, `id_13`–`id_14`, `id_17`–`id_27` | ~20 cols | Mostly numeric — risk scores, counts, offsets (unclear semantics) |

### 4.3 Vesta-Engineered Features (Opaque)

| Group | Count | Notes |
|---|---|---|
| `V1`–`V339` | 339 cols | Engineered by Vesta Corp. Semantics not disclosed. Block-structured missingness suggests they come from different data pipelines/sub-systems |

**Missingness pattern in V-cols**: Three distinct tiers of availability (~0%, ~86%, ~100% missing), indicating these features come from 2–3 distinct sub-systems that don't always activate.

### 4.4 Transactional Features

| Column | Description |
|---|---|
| `TransactionID` | Unique transaction identifier |
| `TransactionDT` | Seconds from reference epoch (covers ~182 days of data) |
| `TransactionAmt` | Amount in USD (range: $0.25 – $31,937; median: $68.77) |

---

## 5. Missing Value Summary

### train_transaction (selected)

| Column | Missing % | Notes |
|---|---|---|
| `dist2` | 93.6% | Likely only populated for specific product types |
| `D7`, `D12`–`D14` | 87–89% | Sparse time deltas |
| `D6`, `D8`, `D9` | ~87% | Sparse |
| `V138`–`V166` | ~86% | Entire V-col sub-block missing |
| `V322`–`V339` | ~86% | Another sparse V-col sub-block |
| `R_emaildomain` | ~33% | Many transactions have no recipient email |
| `P_emaildomain` | ~20% | Some purchasers have no email on file |
| `card2`, `card3`, `card5` | 2–8% | Minor card field gaps |
| `M7`–`M9` | ~46–61% | Selective match flags |

### train_identity (selected)

| Column | Missing % | Notes |
|---|---|---|
| `id_07`, `id_08`, `id_24`–`id_27` | >96% | Nearly unusable |
| `id_18`, `id_03`, `id_04` | 54–69% | High missingness |
| `id_09`, `id_10`, `id_30`, `id_32`–`id_34` | 46–48% | Moderate |
| `DeviceType`, `DeviceInfo` | 2–18% | Generally well populated |
| `id_30` (OS), `id_31` (browser) | ~3–46% | Useful but not always present |

---

## 6. Fraud Signal by Key Dimensions

| Dimension | Notable Findings |
|---|---|
| **Overall** | 3.50% fraud rate |
| **ProductCD=C** | 11.7% fraud rate — highest risk category |
| **ProductCD=S** | 5.9% fraud |
| **card6=credit** | 6.7% fraud (vs 2.4% debit) |
| **card4=discover** | 7.7% fraud (highest among card networks) |
| **P_emaildomain=outlook.com** | 9.5% fraud rate |
| **card1 extremes** | card1=9917 has 33.3% fraud rate on 919 transactions |
| **High-fraud card1 groups** | Multiple card1 buckets >10% fraud, suggesting shared compromised cards |

---

## 7. Candidate Entity Columns for Graph Modeling

These columns can plausibly represent real-world entities that multiple transactions share:

| Entity | Source Column(s) | Rationale |
|---|---|---|
| **Card** | `card1` | High-cardinality instrument ID; shared card = shared account |
| **EmailDomain** | `P_emaildomain`, `R_emaildomain` | Domain-level grouping; certain domains have elevated fraud |
| **BillingAddress** | `addr1` + `addr2` composite | 332×74 combinations; spatial risk clustering |
| **Device** | `DeviceInfo` (normalized) | Exact device fingerprint; fraud rings share devices |
| **OS_Browser** | `id_30` + `id_31` composite | Soft device fingerprint for transactions with identity |
| **ProductType** | `ProductCD` | 5-way product taxonomy; strong risk signal |
| **ProxyType** | `id_23` | IP proxy classification; direct fraud signal |

Columns **not** suitable as graph nodes (too granular, too noisy, or semantically opaque):
- `card2`, `card3`, `card5` — partial card fields, poor semantics
- `V1`–`V339` — opaque engineered features; useful as transaction properties, not nodes
- `C`, `D`, `M` columns — behavioral signals; should be properties on Transaction node
- `id_01`–`id_11`, etc. — numeric signals; should be properties, not nodes

---

## 8. Class Imbalance Treatment

With 3.5% fraud:
- **Do not use accuracy as primary metric**
- Preferred metrics: **PR-AUC**, **Recall@Precision**, **F1 (fraud class)**
- For LightGBM/XGBoost: use `scale_pos_weight` or `is_unbalance=True`
- For threshold optimization: tune threshold on validation set to maximize F1 or recall at acceptable precision
- Do not oversample training data unless PR-AUC stalls — tree models handle imbalance natively with weight parameters

---

## 9. Recommendations

### 9.1 Baseline ML Approach

**Recommended: LightGBM with default imbalance handling**

Reasons:
- Handles 394 mixed-type columns natively with LGBM categorical support
- Built-in missing value handling (no imputation needed for tree models)
- Fast training on 590K rows
- Strong feature importance output for interpretability
- Use `scale_pos_weight = 569877 / 20663 ≈ 27.6` to correct for imbalance
- Encode string columns (card4, card6, ProductCD, email domains) as categoricals or label-encode

**Validation**: Stratified time-aware split. `TransactionDT` covers 182 days — split at day 150 to simulate temporal validation (avoids leakage from future to past).

### 9.2 Graph Modeling Strategy

**Best approach for this dataset**: Bipartite shared-entity graph

The key insight for fraud detection in this dataset is **shared entity reuse**: fraudsters reuse the same cards, email domains, devices, and addresses across multiple transactions. A graph model should expose these connections directly.

The recommended graph makes `Transaction` nodes the central entity, connected to shared entity nodes via typed relationships. Fraud rings become visible as dense sub-graphs where many fraud-labeled transactions share the same card, device, or email domain.

### 9.3 Columns for First Graph Version

**Transaction node properties**: TransactionID, TransactionDT, TransactionAmt, ProductCD, isFraud, card4, card6, addr1, addr2, C1–C14 (selected), M1–M9 (selected)

**Entity nodes to create**:
1. `Card` (card1) → `(:Transaction)-[:USED_CARD]->(:Card)`
2. `EmailDomain` (P_emaildomain) → `(:Transaction)-[:PAYER_EMAIL]->(:EmailDomain)`
3. `EmailDomain` (R_emaildomain) → `(:Transaction)-[:RECIPIENT_EMAIL]->(:EmailDomain)`
4. `BillingAddress` (addr1+addr2 composite) → `(:Transaction)-[:BILLED_TO]->(:BillingAddress)`
5. `Device` (DeviceInfo normalized) → `(:Transaction)-[:USED_DEVICE]->(:Device)`

These 5 entity types with 5 relationship types give a clean, defensible, demo-friendly first graph.
