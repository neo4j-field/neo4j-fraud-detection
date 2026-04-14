// ============================================================
// IEEE-CIS Fraud Detection — Demo Cypher Queries
// Neo4j Aura: neo4j+s://27ad415a.databases.neo4j.io
// ============================================================

// ------------------------------------------------------------
// 1. OVERVIEW: Graph size and fraud distribution
// ------------------------------------------------------------

MATCH (t:Transaction)
RETURN
  count(t) AS total_transactions,
  sum(t.isFraud) AS fraud_transactions,
  round(100.0 * sum(t.isFraud) / count(t), 2) AS fraud_rate_pct;

// ------------------------------------------------------------
// 2. TOP COMPROMISED CARDS
//    Cards with the highest fraud transaction count.
//    A card with 300+ fraud transactions is almost certainly
//    a stolen or cloned card being systematically exploited.
// ------------------------------------------------------------

MATCH (c:Card)<-[:USED_CARD]-(t:Transaction)
WITH c,
     count(t) AS total_txns,
     sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS fraud_txns
WHERE total_txns >= 20
RETURN c.cardId AS card,
       total_txns,
       fraud_txns,
       round(100.0 * fraud_txns / total_txns, 1) AS fraud_rate_pct
ORDER BY fraud_txns DESC
LIMIT 10;

// ------------------------------------------------------------
// 3. SHARED DEVICE FRAUD RINGS
//    Devices used in 10+ transactions where >30% are fraudulent.
//    These are strong indicators of fraud ring devices.
// ------------------------------------------------------------

MATCH (d:Device)<-[:USED_DEVICE]-(t:Transaction)
WITH d,
     count(t) AS total_txns,
     sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS fraud_txns
WHERE total_txns >= 10 AND toFloat(fraud_txns) / total_txns > 0.30
RETURN d.deviceKey AS device,
       total_txns,
       fraud_txns,
       round(100.0 * fraud_txns / total_txns, 1) AS fraud_rate_pct
ORDER BY fraud_txns DESC
LIMIT 15;

// ------------------------------------------------------------
// 4. EMAIL DOMAIN FRAUD CONCENTRATION
//    Email domains ranked by fraud rate and volume.
//    Useful for identifying domains that anonymize or attract fraudsters.
// ------------------------------------------------------------

MATCH (e:EmailDomain)<-[:PAYER_EMAIL]-(t:Transaction)
WITH e,
     count(t) AS total_txns,
     sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS fraud_txns
WHERE total_txns >= 100
RETURN e.domain AS email_domain,
       total_txns,
       fraud_txns,
       round(100.0 * fraud_txns / total_txns, 2) AS fraud_rate_pct
ORDER BY fraud_rate_pct DESC
LIMIT 15;

// ------------------------------------------------------------
// 5. NEIGHBORHOOD OF A KNOWN FRAUDULENT TRANSACTION
//    Shows all entities connected to a specific fraud transaction
//    and how many other transactions (legitimate or fraudulent)
//    share those same entities.
//    Replace 2987004 with any known fraud TransactionID.
// ------------------------------------------------------------

MATCH (t:Transaction {transactionId: 2987004})
OPTIONAL MATCH (t)-[:USED_CARD]->(c:Card)<-[:USED_CARD]-(t2:Transaction)
OPTIONAL MATCH (t)-[:PAYER_EMAIL]->(e:EmailDomain)<-[:PAYER_EMAIL]-(t3:Transaction)
OPTIONAL MATCH (t)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(t4:Transaction)
RETURN
  t.transactionId AS source_transaction,
  t.isFraud AS is_fraud,
  t.transactionAmt AS amount,
  count(DISTINCT t2) AS transactions_on_same_card,
  count(DISTINCT t3) AS transactions_on_same_email_domain,
  count(DISTINCT t4) AS transactions_on_same_device;

// ------------------------------------------------------------
// 6. DENSE FRAUD CLUSTER AROUND A COMPROMISED CARD
//    Finds all fraud transactions sharing a card, and shows
//    the connected entity neighborhood.
//    Good for visualizing a fraud ring in Neo4j Browser.
// ------------------------------------------------------------

MATCH (c:Card {cardId: '9633'})
MATCH (c)<-[:USED_CARD]-(t:Transaction {isFraud: 1})
WITH c, t LIMIT 20
OPTIONAL MATCH (t)-[:PAYER_EMAIL]->(e:EmailDomain)
OPTIONAL MATCH (t)-[:BILLED_TO]->(b:BillingAddress)
RETURN c, t, e, b;

// ------------------------------------------------------------
// 7. SUSPICIOUS BILLING ADDRESSES
//    Billing addresses that concentrate fraud.
//    An address with many fraud transactions could indicate
//    a drop shipping or reshipping fraud scheme.
// ------------------------------------------------------------

MATCH (b:BillingAddress)<-[:BILLED_TO]-(t:Transaction)
WITH b,
     count(t) AS total_txns,
     sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS fraud_txns
WHERE total_txns >= 50
RETURN b.addrKey AS address,
       total_txns,
       fraud_txns,
       round(100.0 * fraud_txns / total_txns, 2) AS fraud_rate_pct
ORDER BY fraud_rate_pct DESC
LIMIT 10;

// ------------------------------------------------------------
// 8. CROSS-ENTITY FRAUD AMPLIFICATION
//    Transactions that share BOTH a compromised card AND
//    a high-fraud email domain. Double-linked fraud signals
//    are much stronger than single-entity signals.
// ------------------------------------------------------------

MATCH (c:Card)<-[:USED_CARD]-(t:Transaction)-[:PAYER_EMAIL]->(e:EmailDomain)
WITH c, e,
     count(t) AS total_txns,
     sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS fraud_txns
WHERE total_txns >= 5 AND toFloat(fraud_txns) / total_txns > 0.15
RETURN c.cardId AS card,
       e.domain AS email_domain,
       total_txns,
       fraud_txns,
       round(100.0 * fraud_txns / total_txns, 1) AS fraud_rate_pct
ORDER BY fraud_txns DESC
LIMIT 10;

// ------------------------------------------------------------
// 9. HIGH-VALUE FRAUD TRANSACTIONS ON SUSPICIOUS DEVICES
//    Large transactions (>$500) on devices with elevated fraud rates.
//    High-value fraud on shared devices = highest-priority alerts.
// ------------------------------------------------------------

MATCH (d:Device)<-[:USED_DEVICE]-(t:Transaction)
WHERE t.transactionAmt > 500
WITH d,
     count(t) AS total_high_value,
     sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS fraud_high_value,
     max(t.transactionAmt) AS max_fraud_amt
WHERE total_high_value >= 5 AND fraud_high_value >= 2
RETURN d.deviceKey AS device,
       total_high_value,
       fraud_high_value,
       round(100.0 * fraud_high_value / total_high_value, 1) AS fraud_rate_pct,
       round(max_fraud_amt, 2) AS max_txn_amount
ORDER BY fraud_high_value DESC
LIMIT 10;

// ------------------------------------------------------------
// 10. GRAPH STATS SUMMARY (for demo slide)
// ------------------------------------------------------------

MATCH (t:Transaction) WITH count(t) AS txCount, sum(t.isFraud) AS fraudCount
MATCH (c:Card)        WITH txCount, fraudCount, count(c) AS cardCount
MATCH (e:EmailDomain) WITH txCount, fraudCount, cardCount, count(e) AS emailCount
MATCH (d:Device)      WITH txCount, fraudCount, cardCount, emailCount, count(d) AS deviceCount
RETURN
  txCount       AS transactions,
  fraudCount    AS fraud_transactions,
  cardCount     AS unique_cards,
  emailCount    AS unique_email_domains,
  deviceCount   AS unique_devices;
