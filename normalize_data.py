"""
Extracts clean, leak-free customer-support interaction pairs 
for the RAG Knowledge Base.
"""

import json
import os
import re
import pandas as pd


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip @mentions
    text = re.sub(r"@\w+", "", text)
    # Normalize extra whitespaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("1. Loading twcs.csv...")
    df = pd.read_csv("twcs.csv")

    # Match Spotify outbound replies with customer parent tweets
    spotify = df[df["author_id"] == "SpotifyCares"].dropna(
        subset=["in_response_to_tweet_id"]
    ).copy()
    spotify["in_response_to_tweet_id"] = spotify[
        "in_response_to_tweet_id"
    ].astype(int)

    customer_ids = spotify["in_response_to_tweet_id"].unique()
    customer_msgs = df[df["tweet_id"].isin(customer_ids)]

    pairs = spotify.merge(
        customer_msgs,
        left_on="in_response_to_tweet_id",
        right_on="tweet_id",
        suffixes=("_reply", "_customer"),
    )

    print(f"Total raw pairs: {len(pairs):,}")

    # 2. Exclude Golden Eval Set IDs (Zero Test Leakage)
    print("2. Removing Golden Evaluation Set examples...")
    with open("data/golden_eval_set.json", "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    golden_ids = {
        int(item["id"].replace("real_", ""))
        for item in golden_set
        if item["id"].replace("real_", "").isdigit()
    }

    pairs = pairs[~pairs["tweet_id_customer"].isin(golden_ids)].copy()
    print(f"Remaining pairs after zero-leakage filter: {len(pairs):,}")

    # 3. Clean and Normalize Text
    print("3. Normalizing text and removing noise...")
    pairs["customer_text"] = pairs["text_customer"].apply(clean_text)
    pairs["reply_text"] = pairs["text_reply"].apply(clean_text)

    # Filter out empty or ultra-short noise (under 15 chars)
    pairs = pairs[
        (pairs["customer_text"].str.len() >= 15)
        & (pairs["reply_text"].str.len() >= 15)
    ]

    # Deduplicate identical customer questions
    pairs = pairs.drop_duplicates(subset=["customer_text"])

    # 4. Save Knowledge Base
    os.makedirs("data", exist_ok=True)
    output_path = "data/clean_spotify_kb.csv"
    pairs[["tweet_id_customer", "customer_text", "reply_text"]].to_csv(
        output_path, index=False, encoding="utf-8"
    )

    print(
        f"Clean RAG Knowledge Base ready: `{output_path}` ({len(pairs):,} records)"
    )


if __name__ == "__main__":
    main()
