# Graph Model Options — IEEE-CIS Fraud Detection

This document presents two candidate graph models for the fraud detection demo.

---

## Why a Graph Model?

The core insight is **shared entity reuse**. Fraudsters rarely act in total isolation:
- A stolen card (`card1`) is often used in multiple transactions
- Fraud rings share devices (`DeviceInfo`), IP proxies, or email domains
- Billing addresses get reused across related fraudulent accounts
- Certain email domains (e.g., `anonymous.com`, `outlook.com`) concentrate fraud

A tabular model sees each transaction independently and cannot detect these patterns. A graph makes the connections between transactions explicit — a dense neighborhood of fraud-labeled transactions around a single shared card or device is a strong signal.

---

## Model A — Simple Demo-Friendly Model

### Node Labels

| Label | Key | Source | Description |
|---|---|---|---|
| `Transaction` | `transactionId` | `TransactionID` | One node per transaction. Carries all tabular features as properties. |
| `Card` | `cardId` | `card1` | Payment instrument fingerprint. High-cardinality (13K unique). |
| `EmailDomain` | `domain` | `P_emaildomain`, `R_emaildomain` | Email domain (not full address). Low-to-medium cardinality (60 unique). |
| `BillingAddress` | `addrKey` | `addr1` + `addr2` composite | Billing region. Moderate cardinality (332×74 = ~24K possible, ~5K populated). |
| `Device` | `deviceKey` | `DeviceInfo` (normalized) | Device fingerprint. Only for transactions with identity data. |

### Relationships

| Type | From | To | Source | Meaning |
|---|---|---|---|---|
| `USED_CARD` | Transaction | Card | card1 | Transaction was made with this card |
| `PAYER_EMAIL` | Transaction | EmailDomain | P_emaildomain | Purchaser's email domain |
| `RECIPIENT_EMAIL` | Transaction | EmailDomain | R_emaildomain | Recipient's email domain |
| `BILLED_TO` | Transaction | BillingAddress | addr1+addr2 | Billing location of transaction |
| `USED_DEVICE` | Transaction | Device | DeviceInfo | Device used (identity-linked transactions only) |

### Strengths
- Clean and minimal: 5 node types, 5 relationship types
- All entity nodes have stable, meaningful keys
- Easy to explain to non-technical audiences
- Device nodes naturally represent only 24.4% of transactions (those with identity) — this is not a problem, it's realistic
- Supports connected-component analysis at the card and device level
- Supports FastRP embeddings on the Transaction node projection

### Weaknesses
- `BillingAddress` is a zip-code-level proxy; not a precise address
- `DeviceInfo` strings need normalization (many near-duplicates like "Windows" vs. "Windows 10")
- Does not model temporal patterns explicitly
- `card1` alone doesn't distinguish between different cardholders who happen to share the same `card1` value (though in practice card1 is close to a card fingerprint)

---

## Model B — Richer, More Expressive Model

### Additional Node Labels (on top of Model A)

| Label | Key | Source | Notes |
|---|---|---|---|
| `OSBrowser` | `osBrowserKey` | `id_30` + `id_31` | Soft device fingerprint from OS + browser combination |
| `ProductType` | `productCode` | `ProductCD` | Transaction product category (W/C/H/S/R) |
| `ProxyType` | `proxyLabel` | `id_23` | IP proxy classification (TRANSPARENT/ANONYMOUS/HIDDEN) |
| `CardNetwork` | `network` | `card4` | visa / mastercard / discover / amex |
| `TimeBucket` | `bucket` | `TransactionDT` | Discretized time window (e.g., day of data) |

### Additional Relationships

| Type | From | To | Meaning |
|---|---|---|---|
| `HAS_OS_BROWSER` | Transaction | OSBrowser | Browser/OS fingerprint at transaction time |
| `IN_PRODUCT` | Transaction | ProductType | Product category |
| `VIA_PROXY` | Transaction | ProxyType | Transaction came through this proxy type |
| `ON_NETWORK` | Transaction | CardNetwork | Card network used |
| `IN_TIME_BUCKET` | Transaction | TimeBucket | Temporal grouping |

### Strengths
- Richer connectivity: fraudulent transactions become identifiable through multiple simultaneous entity matches
- `ProxyType` is a direct fraud signal — any connection to TRANSPARENT or ANONYMOUS proxy is meaningful
- `OSBrowser` provides an additional soft-fingerprint layer for device identification even when `DeviceInfo` is missing
- `TimeBucket` enables temporal subgraph queries ("all fraud in this time window")
- More relationship types → richer embeddings

### Weaknesses
- More nodes and relationships increase loading complexity and graph size
- `ProductType` and `CardNetwork` have very low cardinality (4–5 values); they become hub nodes that connect vast numbers of transactions — these mega-hubs dilute graph features rather than sharpening them
- `TimeBucket` adds temporal context but requires careful design to avoid trivially connecting every transaction in the same day
- `ProxyType` and `OSBrowser` are only available for the 24.4% of transactions with identity data, so large parts of the graph won't have these connections
- More complex to explain and debug

---

## Recommendation: Model A

**Choose Model A for this project.**

### Reasoning

1. **Cardinality is appropriate**: Card (13K), EmailDomain (60), BillingAddress (~5K), Device (~500 after normalization) — none of these are mega-hubs that would destroy graph signal
2. **All 5 entities have clear fraud intuition**: A card shared across many fraudulent transactions, a device used in a fraud cluster, an email domain with high fraud concentration — all of these are immediately explainable
3. **Device covers the identity-linked sub-population cleanly** without forcing artificial null nodes for the 75% of transactions without identity data
4. **Demo-friendly**: A non-technical audience can understand "these 6 transactions all used the same card" without needing to know what `id_30` or `OSBrowser` means
5. **Model B's extra nodes** (ProductType, CardNetwork, TimeBucket) have too-low cardinality and would create hub nodes connecting millions of transactions — this destroys graph discriminability
6. **Model A is sufficient for FastRP embeddings**: The bipartite Transaction-Entity structure gives FastRP enough heterogeneous connectivity to produce meaningful embeddings

Model B elements worth incorporating later: `ProxyType` (strong signal) and `OSBrowser` (useful for the identity sub-population) can be added as Phase 2 improvements if Model A embeddings underperform.
