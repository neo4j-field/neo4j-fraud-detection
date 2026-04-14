"""
Phase 5 — Generate Graph-Derived Features from Neo4j
======================================================
Generates tabular features for every Transaction node by querying the graph.

Feature categories:
  1. Entity-level aggregates: degree, fraud count, fraud rate per shared entity
  2. Connected components (WCC): component size, fraud concentration
  3. Neighborhood fraud signal: weighted neighbor fraud exposure

All features are exported as a Parquet keyed by TransactionID so they can be
joined to the ML training set in Phase 6.

Usage:
    python src/graph/generate_graph_features.py
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

GDS_GRAPH_NAME = "fraud_graph"


# ---------------------------------------------------------------------------
# GDS graph projection
# ---------------------------------------------------------------------------

def drop_graph_if_exists(session, graph_name: str):
    exists = session.run(
        "CALL gds.graph.exists($name) YIELD exists RETURN exists",
        name=graph_name,
    ).single()["exists"]
    if exists:
        session.run("CALL gds.graph.drop($name)", name=graph_name)
        log.info(f"  Dropped existing projected graph: {graph_name}")


def project_graph(session, graph_name: str) -> dict:
    """
    Project the bipartite fraud graph for GDS.
    Includes all node labels and relationship types as undirected.
    """
    log.info(f"Projecting GDS graph '{graph_name}' ...")
    result = session.run(
        """
        CALL gds.graph.project(
            $graphName,
            ['Transaction', 'Card', 'EmailDomain', 'BillingAddress', 'Device'],
            {
                USED_CARD:        {orientation: 'UNDIRECTED'},
                PAYER_EMAIL:      {orientation: 'UNDIRECTED'},
                RECIPIENT_EMAIL:  {orientation: 'UNDIRECTED'},
                BILLED_TO:        {orientation: 'UNDIRECTED'},
                USED_DEVICE:      {orientation: 'UNDIRECTED'}
            }
        )
        YIELD graphName, nodeCount, relationshipCount, projectMillis
        RETURN graphName, nodeCount, relationshipCount, projectMillis
        """,
        graphName=graph_name,
    ).single()
    log.info(
        f"  Projected: {result['nodeCount']:,} nodes, "
        f"{result['relationshipCount']:,} relationships "
        f"({result['projectMillis']}ms)"
    )
    return dict(result)


# ---------------------------------------------------------------------------
# WCC (Weakly Connected Components)
# ---------------------------------------------------------------------------

def run_wcc(session, graph_name: str) -> pd.DataFrame:
    """
    Run WCC and return component assignments for Transaction nodes only.
    Returns DataFrame: TransactionID, componentId
    """
    log.info("Running WCC ...")
    t0 = time.time()
    result = session.run(
        """
        CALL gds.wcc.stream($graphName)
        YIELD nodeId, componentId
        WITH gds.util.asNode(nodeId) AS node, componentId
        WHERE 'Transaction' IN labels(node)
        RETURN node.transactionId AS transactionId, componentId
        """,
        graphName=graph_name,
    )
    rows = [{"transactionId": r["transactionId"], "componentId": r["componentId"]}
            for r in result]
    df = pd.DataFrame(rows)
    log.info(f"  WCC done in {time.time()-t0:.1f}s: {len(df):,} Transaction nodes assigned")
    return df


def enrich_wcc(df_wcc: pd.DataFrame, df_fraud: pd.DataFrame) -> pd.DataFrame:
    """
    Add component-level stats: size, fraud count, fraud rate.
    df_fraud: DataFrame with transactionId and isFraud columns.
    """
    # Join fraud label
    df = df_wcc.merge(df_fraud[["transactionId", "isFraud"]], on="transactionId", how="left")

    # Component stats
    comp_stats = df.groupby("componentId").agg(
        component_size=("transactionId", "count"),
        component_fraud_count=("isFraud", "sum"),
    ).reset_index()
    comp_stats["component_fraud_rate"] = (
        comp_stats["component_fraud_count"] / comp_stats["component_size"]
    )

    df = df.merge(comp_stats, on="componentId", how="left")
    return df[["transactionId", "componentId", "component_size",
               "component_fraud_count", "component_fraud_rate"]]


# ---------------------------------------------------------------------------
# Entity-level graph features (Cypher aggregates)
# ---------------------------------------------------------------------------

def fetch_card_features(session) -> pd.DataFrame:
    """
    Per-transaction card aggregates using a two-step approach:
    1. Aggregate stats per Card entity.
    2. Join card stats back to each Transaction.
    Avoids the memory-heavy self-join on 590K transactions.
    """
    log.info("Fetching card features (entity-first aggregation) ...")
    t0 = time.time()
    result = session.run(
        """
        MATCH (c:Card)<-[:USED_CARD]-(t:Transaction)
        WITH c,
             count(t) AS card_tx_count,
             sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS card_fraud_count
        MATCH (c)<-[:USED_CARD]-(t2:Transaction)
        RETURN t2.transactionId AS transactionId,
               card_tx_count,
               card_fraud_count,
               toFloat(card_fraud_count) / card_tx_count AS card_fraud_rate
        """
    )
    rows = [dict(r) for r in result]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["transactionId", "card_tx_count", "card_fraud_count", "card_fraud_rate"])
    log.info(f"  Card features: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def fetch_payer_email_features(session) -> pd.DataFrame:
    """Per-transaction payer email domain aggregates (entity-first)."""
    log.info("Fetching payer email domain features (entity-first aggregation) ...")
    t0 = time.time()
    result = session.run(
        """
        MATCH (e:EmailDomain)<-[:PAYER_EMAIL]-(t:Transaction)
        WITH e,
             count(t) AS payer_email_tx_count,
             sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS payer_email_fraud_count
        MATCH (e)<-[:PAYER_EMAIL]-(t2:Transaction)
        RETURN t2.transactionId AS transactionId,
               payer_email_tx_count,
               payer_email_fraud_count,
               toFloat(payer_email_fraud_count) / payer_email_tx_count AS payer_email_fraud_rate
        """
    )
    rows = [dict(r) for r in result]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["transactionId", "payer_email_tx_count", "payer_email_fraud_count", "payer_email_fraud_rate"])
    log.info(f"  Payer email features: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def fetch_billing_features(session) -> pd.DataFrame:
    """Per-transaction billing address aggregates (entity-first)."""
    log.info("Fetching billing address features (entity-first aggregation) ...")
    t0 = time.time()
    result = session.run(
        """
        MATCH (b:BillingAddress)<-[:BILLED_TO]-(t:Transaction)
        WITH b,
             count(t) AS billing_tx_count,
             sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS billing_fraud_count
        MATCH (b)<-[:BILLED_TO]-(t2:Transaction)
        RETURN t2.transactionId AS transactionId,
               billing_tx_count,
               billing_fraud_count,
               toFloat(billing_fraud_count) / billing_tx_count AS billing_fraud_rate
        """
    )
    rows = [dict(r) for r in result]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["transactionId", "billing_tx_count", "billing_fraud_count", "billing_fraud_rate"])
    log.info(f"  Billing features: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def fetch_device_features(session) -> pd.DataFrame:
    """Per-transaction device aggregates (entity-first, identity-linked only)."""
    log.info("Fetching device features (entity-first aggregation) ...")
    t0 = time.time()
    result = session.run(
        """
        MATCH (d:Device)<-[:USED_DEVICE]-(t:Transaction)
        WITH d,
             count(t) AS device_tx_count,
             sum(CASE WHEN t.isFraud = 1 THEN 1 ELSE 0 END) AS device_fraud_count
        MATCH (d)<-[:USED_DEVICE]-(t2:Transaction)
        RETURN t2.transactionId AS transactionId,
               device_tx_count,
               device_fraud_count,
               toFloat(device_fraud_count) / device_tx_count AS device_fraud_rate
        """
    )
    rows = [dict(r) for r in result]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["transactionId", "device_tx_count", "device_fraud_count", "device_fraud_rate"])
    log.info(f"  Device features: {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Combine all graph features
# ---------------------------------------------------------------------------

def combine_features(
    df_wcc: pd.DataFrame,
    df_card: pd.DataFrame,
    df_email: pd.DataFrame,
    df_billing: pd.DataFrame,
    df_device: pd.DataFrame,
    all_transaction_ids: pd.Series,
) -> pd.DataFrame:
    """
    Left-join all graph features onto the full set of transaction IDs.
    Missing values (e.g. transactions without device) are filled with 0
    for counts and -1 for rates (indicating "no data", not zero fraud).
    """
    df = pd.DataFrame({"transactionId": all_transaction_ids})

    # WCC
    df = df.merge(df_wcc, on="transactionId", how="left")

    # Entity features
    for feat_df in [df_card, df_email, df_billing, df_device]:
        if len(feat_df) > 0:
            df = df.merge(feat_df, on="transactionId", how="left")

    # Fill missing count features with 0
    count_cols = [c for c in df.columns if c.endswith("_count") or c.endswith("_size")]
    for c in count_cols:
        df[c] = df[c].fillna(0).astype(float)

    # Fill missing rate features with -1 (explicit "no entity data")
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    for c in rate_cols:
        df[c] = df[c].fillna(-1.0)

    # componentId: fill isolated nodes with -1
    if "componentId" in df.columns:
        df["componentId"] = df["componentId"].fillna(-1).astype(int)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    log.info(f"Connecting to: {uri}")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    log.info("Connected.")

    # Fetch fraud labels for WCC enrichment
    log.info("Fetching transaction IDs and fraud labels from graph ...")
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (t:Transaction) RETURN t.transactionId AS transactionId, t.isFraud AS isFraud"
        )
        df_all = pd.DataFrame([dict(r) for r in result])
    log.info(f"  Fetched {len(df_all):,} transactions")

    # --- Project GDS graph ---
    with driver.session(database=database) as session:
        drop_graph_if_exists(session, GDS_GRAPH_NAME)
        project_graph(session, GDS_GRAPH_NAME)

    # --- WCC ---
    with driver.session(database=database) as session:
        df_wcc_raw = run_wcc(session, GDS_GRAPH_NAME)
    df_wcc = enrich_wcc(df_wcc_raw, df_all)

    # --- Entity features ---
    with driver.session(database=database) as session:
        df_card    = fetch_card_features(session)
        df_email   = fetch_payer_email_features(session)
        df_billing = fetch_billing_features(session)
        df_device  = fetch_device_features(session)

    # --- Drop projected graph (free memory) ---
    with driver.session(database=database) as session:
        session.run("CALL gds.graph.drop($name)", name=GDS_GRAPH_NAME)
        log.info(f"  Dropped GDS graph '{GDS_GRAPH_NAME}'")

    # --- Combine ---
    log.info("Combining all graph features ...")
    df_features = combine_features(
        df_wcc, df_card, df_email, df_billing, df_device,
        df_all["transactionId"],
    )
    log.info(f"  Final feature frame: {df_features.shape}")
    log.info(f"  Columns: {list(df_features.columns)}")

    # --- Save ---
    out_path = ARTIFACTS_DIR / "graph_features.parquet"
    df_features.to_parquet(out_path, index=False)
    log.info(f"  Saved: {out_path}")

    # Quick sanity check
    log.info("\n=== Feature Summary ===")
    log.info(f"  Rows: {len(df_features):,}")
    log.info(f"  Columns: {len(df_features.columns)}")
    log.info(f"  Null counts:\n{df_features.isnull().sum()[df_features.isnull().sum()>0]}")
    log.info(f"\n  Sample stats (fraud transactions):")
    fraud_mask = df_all.set_index("transactionId")["isFraud"] == 1
    fraud_ids = df_all[df_all["isFraud"] == 1]["transactionId"].values
    df_fraud_feats = df_features[df_features["transactionId"].isin(fraud_ids)]
    df_legit_feats = df_features[~df_features["transactionId"].isin(fraud_ids)]
    for col in ["card_fraud_rate", "billing_fraud_rate", "device_fraud_rate",
                "component_fraud_rate", "component_size"]:
        if col in df_features.columns:
            fraud_mean = df_fraud_feats[col].replace(-1, np.nan).mean()
            legit_mean = df_legit_feats[col].replace(-1, np.nan).mean()
            log.info(f"    {col}: fraud={fraud_mean:.4f} | legit={legit_mean:.4f}")

    driver.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
