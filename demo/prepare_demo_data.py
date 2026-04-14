"""
prepare_demo_data.py
--------------------
Run once before the Vertex AI demo to upload all required artifacts to GCS.

Usage:
    python demo/prepare_demo_data.py \
        --bucket your-gcs-bucket \
        --prefix neo4j-fraud-detection

What it uploads:
    artifacts/graph_features_v2.parquet        -> gs://BUCKET/PREFIX/artifacts/
    artifacts/transaction_embeddings_v2.parquet -> gs://BUCKET/PREFIX/artifacts/
    train_transaction.csv                       -> gs://BUCKET/PREFIX/
    docs/Fraud_Datamodel_Graph_DB.png           -> gs://BUCKET/PREFIX/docs/
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def gcs_copy(local: Path, gcs_uri: str) -> None:
    print(f"  Uploading {local.name} ({local.stat().st_size / 1e6:.0f} MB) ...")
    result = subprocess.run(
        ["gsutil", "-m", "cp", str(local), gcs_uri],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        sys.exit(1)
    print(f"  Done -> {gcs_uri}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    parser.add_argument("--prefix", default="neo4j-fraud-detection", help="GCS path prefix")
    args = parser.parse_args()

    base = f"gs://{args.bucket}/{args.prefix}"

    files = [
        (ROOT / "artifacts" / "graph_features_v2.parquet",        f"{base}/artifacts/"),
        (ROOT / "artifacts" / "transaction_embeddings_v2.parquet", f"{base}/artifacts/"),
        (ROOT / "train_transaction.csv",                           f"{base}/"),
        (ROOT / "docs" / "Fraud_Datamodel_Graph_DB.png",          f"{base}/docs/"),
    ]

    print(f"Uploading {len(files)} files to gs://{args.bucket}/{args.prefix}/\n")
    for local, remote in files:
        if not local.exists():
            print(f"  SKIP (not found): {local}")
            continue
        gcs_copy(local, remote)

    print("\nAll uploads complete.")
    print(f"\nIn vertex_ai_demo.ipynb, set:")
    print(f"  USE_GCS    = True")
    print(f"  GCS_BUCKET = '{args.bucket}'")
    print(f"  GCS_PREFIX = '{args.prefix}'")


if __name__ == "__main__":
    main()
