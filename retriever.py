"""
Two-stage semantic retriever for the Spotify Support Knowledge Base:
1. Stage 1: Fast Bi-Encoder (all-MiniLM-L6-v2) retrieves Top-10 candidates.
2. Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) re-ranks to return Top-k.
"""

import os
import time
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer


class SpotifyRetriever:

    def __init__(
        self,
        embeddings_path="data/kb_embeddings.npy",
        metadata_path="data/kb_metadata.parquet",
        bi_encoder_name="all-MiniLM-L6-v2",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        if not os.path.exists(embeddings_path) or not os.path.exists(
            metadata_path
        ):
            raise FileNotFoundError(
                "Knowledge base files not found. Please run `build_kb_index.py` first!"
            )

        print("Loading Bi-Encoder & Cross-Encoder models...")
        # Stage 1: Bi-Encoder for initial search
        self.bi_encoder = SentenceTransformer(bi_encoder_name)

        # Stage 2: Cross-Encoder for deep re-ranking
        self.cross_encoder = CrossEncoder(cross_encoder_name)

        # Pre-computed vectors & metadata
        self.embeddings = np.load(embeddings_path)
        self.metadata_df = pd.read_parquet(metadata_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        initial_candidates: int = 15,
        rerank: bool = True,
    ) -> list[dict]:
        """Retrieves top-k resolutions using vector search + cross-encoder re-ranking."""
        # --- Stage 1: Fast Vector Search (Top 15) ---
        query_vec = self.bi_encoder.encode([query], normalize_embeddings=True)[
            0
        ]
        bi_scores = np.dot(self.embeddings, query_vec)
        candidate_indices = np.argsort(bi_scores)[::-1][:initial_candidates]

        candidates = []
        cross_inp = []
        for idx in candidate_indices:
            row = self.metadata_df.iloc[idx]
            # Construct candidate context string
            candidate_text = (
                f"{row['customer_text']} - Reply: {row['reply_text']}"
            )
            candidates.append(
                {
                    "tweet_id": int(row["tweet_id_customer"]),
                    "historical_customer": row["customer_text"],
                    "historical_reply": row["reply_text"],
                    "bi_score": float(bi_scores[idx]),
                }
            )
            cross_inp.append([query, candidate_text])

        # If re-ranking is disabled, return bi-encoder top-k
        if not rerank:
            return candidates[:top_k]

        # --- Stage 2: Cross-Encoder Re-ranking ---
        cross_scores = self.cross_encoder.predict(cross_inp)
        for i, score in enumerate(cross_scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort by cross-encoder score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


# Quick test
if __name__ == "__main__":
    retriever = SpotifyRetriever()

    test_query = "@SpotifyCares my app keeps crashing whenever I try to play downloaded songs on offline mode"
    print(f"\nQuery: '{test_query}'")

    start = time.time()
    results = retriever.retrieve(
        test_query, top_k=3, initial_candidates=15, rerank=True
    )
    elapsed_ms = (time.time() - start) * 1000

    print(
        f"\nFound {len(results)} re-ranked matches in {elapsed_ms:.2f} ms:\n"
    )
    for i, match in enumerate(results, 1):
        print(
            f"[{i}] Re-rank Score: {match['rerank_score']:.4f} (Bi-score: {match['bi_score']:.4f})"
        )
        print(f"    Past Customer: {match['historical_customer']}")
        print(f"    Spotify Reply: {match['historical_reply']}\n")
