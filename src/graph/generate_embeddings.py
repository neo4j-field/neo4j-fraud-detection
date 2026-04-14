"""
Phase 5 — Generate Node Embeddings with FastRP (Neo4j GDS)
============================================================
Uses FastRP (Fast Random Projection) to generate node embeddings for
Transaction nodes in the fraud detection graph.

Why FastRP?
- Designed for large graphs; linear time complexity
- Available natively in GDS — no external ML framework needed
- Works on heterogeneous graphs with multiple node/relationship types
- Produces good-quality embeddings that capture neighborhood structure
- Transactions connected to the same fraudulent card/device will embed near each other

Embedding strategy:
- Project the full bipartite graph (Transaction + all 4 entity types)
- Run FastRP with iterationWeights to propagate signal 2 hops
- Stream embeddings for Transaction nodes only
- Export as Parquet keyed by TransactionID

Usage:
    python src/graph/generate_embeddings.py
    python src/graph/generate_embeddings.py --embedding-dim 64
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

GDS_GRAPH_NAME = "fraud_embedding_graph"
DEFAULT_EMBEDDING_DIM = 64


# ---------------------------------------------------------------------------
# GDS projection
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
    Project bipartite graph including all entity node types.
    All relationship types are treated as undirected so embeddings can
    propagate information from entity nodes to transaction nodes and vice versa.
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
# FastRP embedding
# ---------------------------------------------------------------------------

def run_fastrp(session, graph_name: str, embedding_dim: int) -> pd.DataFrame:
    """
    Run FastRP on the projected graph and return Transaction node embeddings.

    iterationWeights: [0.0, 1.0, 1.0] means:
      - hop 0 (self): identity (not used)
      - hop 1 (direct neighbors): weight 1.0
      - hop 2 (2-hop neighbors): weight 1.0
    This ensures embeddings capture both direct entity sharing (1 hop)
    and indirect co-occurrence through shared entities (2 hops).
    """
    log.info(f"Running FastRP (dim={embedding_dim}) ...")
    t0 = time.time()

    result = session.run(
        """
        CALL gds.fastRP.stream(
            $graphName,
            {
                embeddingDimension: $dim,
                iterationWeights:   [0.0, 1.0, 1.0],
                normalizationStrength: -1.0,
                randomSeed: 42
            }
        )
        YIELD nodeId, embedding
        WITH gds.util.asNode(nodeId) AS node, embedding
        WHERE 'Transaction' IN labels(node)
        RETURN node.transactionId AS transactionId, embedding
        """,
        graphName=graph_name,
        dim=embedding_dim,
    )

    rows = []
    for r in result:
        rows.append({
            "transactionId": r["transactionId"],
            "embedding": list(r["embedding"]),
        })

    log.info(f"  FastRP done in {time.time()-t0:.1f}s: {len(rows):,} embeddings")

    if not rows:
        return pd.DataFrame()

    # Expand embedding list into columns: emb_0, emb_1, ..., emb_N
    df = pd.DataFrame(rows)
    emb_array = np.array(df["embedding"].tolist(), dtype=np.float32)
    emb_cols = {f"emb_{i}": emb_array[:, i] for i in range(emb_array.shape[1])}
    df = pd.concat([df[["transactionId"]], pd.DataFrame(emb_cols)], axis=1)

    return df


# ---------------------------------------------------------------------------
# Embedding quality check
# ---------------------------------------------------------------------------

def check_embedding_quality(df_emb: pd.DataFrame, df_fraud: pd.DataFrame):
    """
    Simple quality check: do fraud transactions cluster differently from legit?
    Compute mean embedding per class and cosine similarity between centroids.
    Lower cosine similarity = better separation.
    """
    if "isFraud" not in df_fraud.columns:
        return

    df = df_emb.merge(df_fraud[["transactionId", "isFraud"]], on="transactionId", how="left")
    df = df.dropna(subset=["isFraud"])

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    fraud_centroid = df[df["isFraud"] == 1][emb_cols].mean().values
    legit_centroid = df[df["isFraud"] == 0][emb_cols].mean().values

    cosine = np.dot(fraud_centroid, legit_centroid) / (
        np.linalg.norm(fraud_centroid) * np.linalg.norm(legit_centroid) + 1e-10
    )
    log.info(f"  Centroid cosine similarity (fraud vs legit): {cosine:.4f}")
    log.info(f"  (Closer to 0 = better class separation in embedding space)")

    # L2 distance between centroids
    l2 = np.linalg.norm(fraud_centroid - legit_centroid)
    log.info(f"  Centroid L2 distance: {l2:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate FastRP embeddings")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    args = parser.parse_args()

    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    log.info(f"Connecting to: {uri}")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    log.info("Connected.")

    # Project graph
    with driver.session(database=database) as session:
        drop_graph_if_exists(session, GDS_GRAPH_NAME)
        project_graph(session, GDS_GRAPH_NAME)

    # Run FastRP
    with driver.session(database=database) as session:
        df_emb = run_fastrp(session, GDS_GRAPH_NAME, args.embedding_dim)

    # Drop projected graph
    with driver.session(database=database) as session:
        session.run("CALL gds.graph.drop($name)", name=GDS_GRAPH_NAME)
        log.info(f"  Dropped GDS graph '{GDS_GRAPH_NAME}'")

    if df_emb.empty:
        log.error("No embeddings returned. Check graph projection.")
        driver.close()
        return

    log.info(f"Embedding frame shape: {df_emb.shape}")
    log.info(f"Sample (first row):\n  {df_emb.iloc[0].to_dict()}")

    # Fetch fraud labels for quality check
    log.info("Fetching fraud labels for quality check ...")
    with driver.session(database=database) as session:
        result = session.run(
            "MATCH (t:Transaction) RETURN t.transactionId AS transactionId, t.isFraud AS isFraud"
        )
        df_fraud = pd.DataFrame([dict(r) for r in result])

    log.info("=== Embedding Quality Check ===")
    check_embedding_quality(df_emb, df_fraud)

    # Save
    out_path = ARTIFACTS_DIR / "transaction_embeddings.parquet"
    df_emb.to_parquet(out_path, index=False)
    log.info(f"Saved: {out_path}")
    log.info(f"  Shape: {df_emb.shape}")
    log.info(f"  Columns: transactionId + {args.embedding_dim} embedding dims")

    driver.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
