import pandas as pd

df = pd.read_csv('twcs.csv')

print(f"Total rows: {len(df):,}")
print(df.dtypes)

spotify = df[df['author_id'] == 'SpotifyCares']
print(f"\nSpotifyCares tweets (outbound replies): {len(spotify):,}")

# SpotifyCares tweets always have in_response_to_tweet_id pointing to the customer's tweet
spotify_replies = spotify.dropna(subset=['in_response_to_tweet_id']).copy()
spotify_replies['in_response_to_tweet_id'] = spotify_replies['in_response_to_tweet_id'].astype(int)

print(f"SpotifyCares replies with a parent tweet: {len(spotify_replies):,}")

# Pull the customer tweets being replied to
customer_tweet_ids = spotify_replies['in_response_to_tweet_id'].unique()
customer_msgs = df[df['tweet_id'].isin(customer_tweet_ids)]

print(f"Matched customer tweets: {len(customer_msgs):,}")
print(f"Inbound check (should be ~all True): {customer_msgs['inbound'].mean():.2%}")

pairs = spotify_replies.merge(
    customer_msgs,
    left_on='in_response_to_tweet_id',
    right_on='tweet_id',
    suffixes=('_reply', '_customer')
)

pairs = pairs[[
    'tweet_id_customer', 'text_customer', 'created_at_customer',
    'tweet_id_reply', 'text_reply', 'created_at_reply'
]]

print(f"Total clean pairs: {len(pairs):,}")
pairs.head(10)

import random

sample_replies = pairs['text_reply'].sample(20, random_state=42).tolist()
for i, r in enumerate(sample_replies, 1):
    print(f"{i}. {r}\n")

deflection_keywords = ['dm', 'direct message', 'follow you', 'follow us so we can dm']

def is_deflection(text):
    text_lower = str(text).lower()
    return any(kw in text_lower for kw in deflection_keywords)

pairs['is_deflection'] = pairs['text_reply'].apply(is_deflection)

deflection_rate = pairs['is_deflection'].mean()
print(f"DM-deflection rate: {deflection_rate:.2%}")
print(f"Deflection examples:")
print(pairs[pairs['is_deflection']]['text_reply'].sample(5, random_state=1).tolist())

pairs['customer_len'] = pairs['text_customer'].str.len()
pairs['reply_len'] = pairs['text_reply'].str.len()

print(pairs[['customer_len', 'reply_len']].describe())

# Count how many replies exist per original customer tweet (proxy for thread complexity)
thread_depth = pairs.groupby('tweet_id_customer').size()
print(thread_depth.value_counts().sort_index())
