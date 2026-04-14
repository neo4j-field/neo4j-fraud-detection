# Graph Load Validation Report

## Load Summary

| Metric | Value |
|---|---|
| Load time | ~445 seconds (~7.4 minutes) |
| Throughput | ~1,330 rows/second |
| Total CSV rows processed | 590,540 |

## Node Counts

| Label | Count | Notes |
|---|---|---|
| **Transaction** | 590,540 | All training transactions loaded |
| **Card** | 13,553 | Unique `card1` values |
| **EmailDomain** | 60 | Unique email domains (purchaser + recipient combined) |
| **BillingAddress** | 437 | Unique `addr1|addr2` combinations |
| **Device** | 1,457 | Normalized DeviceInfo values |

## Relationship Counts

| Type | Count | Notes |
|---|---|---|
| `USED_CARD` | 590,540 | Near 1:1 with transactions (minimal null card1) |
| `PAYER_EMAIL` | 496,084 | ~84% of transactions have P_emaildomain |
| `RECIPIENT_EMAIL` | 137,291 | ~23% of transactions have R_emaildomain |
| `BILLED_TO` | 524,834 | ~89% have both addr1 and addr2 |
| `USED_DEVICE` | 118,666 | ~20% (identity-linked transactions with non-null DeviceInfo) |
| **Total edges** | **~1,866,915** | |

## Fraud Distribution

| Metric | Value |
|---|---|
| Fraud transactions in graph | 20,663 |
| Fraud rate | 3.50% |

## Key Observations

### Top Cards by Fraud Volume
| Card ID | Fraud Transactions |
|---|---|
| 9633 | 742 |
| 9500 | 528 |
| 15885 | 444 |
| 9026 | 397 |
| 15063 | 319 |

These cards show strong fraud concentration. Card 9633 has 742 fraud transactions out of ~4,158 total (17.8% fraud rate on that card).

### Top Devices by Fraud Volume
| Device | Fraud Transactions |
|---|---|
| Windows | 3,121 |
| iOS Device | 1,240 |
| MacOS | 278 |
| hi6210sft | 180 |
| SM-A300H | 169 |

Note: "Windows" and "iOS Device" are generic device categories (high volume). The Samsung model-specific entries (hi6210sft, SM-A300H) at positions 4–5 are more specific and represent stronger fraud signals per device fingerprint.

## Validation Checks Passed

- [x] Transaction count matches CSV row count (590,540)
- [x] Fraud transaction count matches known label distribution (20,663 = 3.50%)
- [x] Card count matches `card1` cardinality from profiling (13,553)
- [x] EmailDomain count matches expected domain count (60)
- [x] All relationships created without fatal errors
- [x] Idempotent MERGE writes (safe to re-run)
