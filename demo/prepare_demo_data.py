"""
prepare_demo_data.py
--------------------
Run this script ONCE on your local machine before the Vertex AI demo.

What it does:
  1. Merges train_transaction.csv + graph_features_v2.parquet
     + transaction_embeddings_v2.parquet into a single
     artifacts/demo_dataset.parquet (~300 MB).
  2. Uploads it (and the graph model image) to GCS so the
     Vertex AI notebook can load everything from one path.

Usage:
  # Step 1 only (create local parquet, no GCS):
  python demo/prepare_demo_data.py

  # Step 1 + Step 2 (create and upload to GCS):
  python demo/prepare_demo_data.py --bucket your-bucket-name

Requirements:
  pip install pandas pyarrow
  gcloud auth application-default login   (for GCS upload)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "demo_dataset.parquet"


def build_merged_parquet() -> Path:
    print("Building demo_dataset.parquet ...")

    print("  [1/3] Loading train_transaction.csv ...")
    tx = pd.read_csv(ROOT / "train_transaction.csv")
    print(f"        {tx.shape[0]:,} rows x {tx.shape[1]} cols")

    print("  [2/3] Loading graph_features_v2.parquet ...")
    gf = pd.read_parquet(ARTIFACTS / "graph_features_v2.parquet")
    gf = gf.rename(columns={"transactionId": "TransactionID"})
    print(f"        {gf.shape[0]:,} rows x {gf.shape[1]} cols")

    print("  [3/3] Loading transaction_embeddings_v2.parquet ...")
    em = pd.read_parquet(ARTIFACTS / "transaction_embeddings_v2.parquet")
    em = em.rename(columns={"transactionId": "TransactionID"})
    print(f"        {em.shape[0]:,} rows x {em.shape[1]} cols")

    print("  Merging ...")
    df = tx.merge(gf, on="TransactionID", how="left")
    df = df.merge(em, on="TransactionID", how="left")
    print(f"  Merged: {df.shape[0]:,} rows x {df.shape[1]} cols")

    print(f"  Saving to {OUTPUT} ...")
    df.to_parquet(OUTPUT, index=False, compression="snappy")
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"  Saved: {size_mb:.0f} MB")
    return OUTPUT


def upload_to_gcs(bucket: str, prefix: str = "neo4j-fraud-detection") -> None:
    base = f"gs://{bucket}/{prefix}"
    uploads = [
        (OUTPUT,                                           f"{base}/artifacts/"),
        (ROOT / "docs" / "Fraud_Datamodel_Graph_DB.png",  f"{base}/docs/"),
    ]
    print(f"\nUploading to gs://{bucket}/{prefix}/ ...")
    for local, remote in uploads:
        if not local.exists():
            print(f"  SKIP (not found): {local.name}")
            continue
        size_mb = local.stat().st_size / 1e6
        print(f"  {local.name} ({size_mb:.0f} MB) -> {remote}")
        result = subprocess.run(
            ["gsutil", "-m", "cp", str(local), remote],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
            sys.exit(1)
        print(f"  Done.")

    print(f"\nAll uploads complete.")
    print(f"\nIn vertex_ai_experiments.ipynb, set:")
    print(f"  DATASET_PATH = 'gs://{bucket}/{prefix}/artifacts/demo_dataset.parquet'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare demo data for Vertex AI.")
    parser.add_argument(
        "--bucket",
        default=None,
        help="GCS bucket name (no gs://). Omit to skip GCS upload.",
    )
    parser.add_argument(
        "--prefix",
        default="neo4j-fraud-detection",
        help="GCS path prefix (default: neo4j-fraud-detection)",
    )
    args = parser.parse_args()

    build_merged_parquet()

    if args.bucket:
        upload_to_gcs(args.bucket, args.prefix)
    else:
        print(f"\nLocal parquet ready: {OUTPUT}")
        print("To upload to GCS, re-run with: --bucket your-bucket-name")


if __name__ == "__main__":
    main()
