"""
Computes dense vector embeddings for the Spotify Knowledge Base 
and saves them to disk for fast cosine similarity search.
"""

import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    start_time = time.time()
    csv_path = "data/clean_spotify_kb.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find `{csv_path}`. Run normalize_data.py first!"
        )

    print(f"1. Loading Knowledge Base from `{csv_path}`...")
    df = pd.read_csv(csv_path)
    print(f"Total entries to index: {len(df):,}")

    # 2. Load Embedding Model
    model_name = "all-MiniLM-L6-v2"
    print(f"2. Loading SentenceTransformer model: `{model_name}`...")
    model = SentenceTransformer(model_name)

    # 3. Compute Embeddings
    print("3. Encoding customer queries in batches...")
    queries = df["customer_text"].tolist()

    # normalize_embeddings=True makes vectors unit length (cosine similarity = dot product)
    embeddings = model.encode(
        queries,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(f"Generated embeddings matrix shape: {embeddings.shape}")

    # 4. Save to disk
    os.makedirs("data", exist_ok=True)
    embeddings_path = "data/kb_embeddings.npy"
    metadata_path = "data/kb_metadata.parquet"

    print("4. Saving embeddings and metadata to disk...")
    np.save(embeddings_path, embeddings.astype(np.float32))
    df[["tweet_id_customer", "customer_text", "reply_text"]].to_parquet(
        metadata_path, index=False
    )

    elapsed = time.time() - start_time
    print(f"\nIndexing complete in {elapsed:.1f} seconds!")
    print(f"Embeddings saved to: `{embeddings_path}`")
    print(f"Metadata saved to: `{metadata_path}`")


if __name__ == "__main__":
    main()
