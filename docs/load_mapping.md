# Graph Load Mapping - IEEE-CIS Fraud Detection

Documents exactly which CSV columns map to which Neo4j node labels and relationship types.

---

## Source Files

| File | Rows | Role |
|---|---|---|
| `train_transaction.csv` | 590,540 | Primary source: Transaction nodes + most entity relationships |
| `train_identity.csv` | 144,233 | Secondary source: Device nodes (joined to transactions via TransactionID) |

Join: `LEFT JOIN train_identity ON TransactionID`

---

## Node Mappings

### Transaction

| Neo4j Property | Source Column | Transformation |
|---|---|---|
| `transactionId` | `TransactionID` | Cast to int. Primary key. |
| `transactionDT` | `TransactionDT` | Float (seconds from reference epoch) |
| `transactionAmt` | `TransactionAmt` | Float (USD) |
| `productCD` | `ProductCD` | String (W/C/H/S/R) |
| `isFraud` | `isFraud` | Int (0 or 1). NULL for test data. |
| `card4` | `card4` | String (visa/mastercard/amex/discover) |
| `card6` | `card6` | String (debit/credit) |
| `addr1` | `addr1` | Float (billing region code) |
| `addr2` | `addr2` | Float (country/region code) |
| `dist1` | `dist1` | Float (distance signal) |
| `hasIdentity` | derived | Boolean: whether identity row exists for this transaction |

Columns intentionally excluded from graph node (kept for ML only): all C, D, M, V columns.

### Card

| Neo4j Property | Source Column | Transformation |
|---|---|---|
| `cardId` | `card1` | String (cast from int). Unique per card instrument. |

NULL policy: if `card1` is null, skip `USED_CARD` relationship (rare, <0.1%).

### EmailDomain

| Neo4j Property | Source Column | Transformation |
|---|---|---|
| `domain` | `P_emaildomain` or `R_emaildomain` | Lowercased string. Shared node label for both directions. |

One `EmailDomain` node per distinct domain value. Two relationship types distinguish direction.

NULL policy: if email domain is null, skip the relationship.

### BillingAddress

| Neo4j Property | Source Column | Transformation |
|---|---|---|
| `addrKey` | `addr1` + `addr2` | Composite key: `"{int(addr1)}|{int(addr2)}"` |

NULL policy: if either `addr1` or `addr2` is null, skip `BILLED_TO` relationship.

### Device

| Neo4j Property | Source Column | Transformation |
|---|---|---|
| `deviceKey` | `DeviceInfo` | Normalized string (strip build suffixes, cap at 200 chars) |

Source: `train_identity.csv` only (24.4% of transactions).
Normalization: `" Build/XXXXXXX"` suffixes stripped. `"Trident/7.0"` → `"Windows IE11"`.

NULL policy: if `DeviceInfo` is null or no identity row, skip `USED_DEVICE` relationship.

---

## Relationship Mappings

| Relationship | From | To | Source Column | NULL Policy |
|---|---|---|---|---|
| `USED_CARD` | Transaction | Card | `card1` | Skip if null |
| `PAYER_EMAIL` | Transaction | EmailDomain | `P_emaildomain` | Skip if null |
| `RECIPIENT_EMAIL` | Transaction | EmailDomain | `R_emaildomain` | Skip if null |
| `BILLED_TO` | Transaction | BillingAddress | `addr1` + `addr2` | Skip if either is null |
| `USED_DEVICE` | Transaction | Device | `DeviceInfo` | Skip if null or no identity row |

---

## Idempotency

All writes use `MERGE` not `CREATE`. Running the loader twice will not create duplicate nodes or relationships.
