"""Register CryptoPanic News Sentiment as a data source.

Fetches crypto news posts with sentiment scores from CryptoPanic.
Needs a free API key from https://cryptopanic.com

Metrics (aggregated to 15m):
  - news_sentiment      → avg sentiment score (-1 to 1)
  - news_count          → number of news items
  - news_positive_ratio → fraction of positive news
  - news_negative_ratio → fraction of negative news
  - news_important      → count of important/breaking news

Usage:
    # Fetch last 7 days (default)
    CRYPTOPANIC_API_KEY="your_key" \\
      PYTHONPATH="$PWD" .venv/bin/python3 scripts/register_cryptopanic.py

    # Fetch specific page range
    CRYPTOPANIC_API_KEY="your_key" \\
      PYTHONPATH="$PWD" .venv/bin/python3 scripts/register_cryptopanic.py --pages 5
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TS_ASSET_ID = "news:cryptopanic:sentiment"

SENTIMENT_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "important": 0.0,
    "neutral": 0.0,
    "bullish": 0.75,
    "bearish": -0.75,
}


async def fetch_news(api_key: str, pages: int = 3) -> list[dict]:
    """Fetch news posts from CryptoPanic API."""
    import httpx

    all_posts = []
    async with httpx.AsyncClient(timeout=30) as client:
        for page in range(1, pages + 1):
            try:
                resp = await client.get(
                    "https://cryptopanic.com/api/v1/posts/",
                    params={
                        "auth_token": api_key,
                        "limit": 100,
                        "page": page,
                        "public": "true",
                    },
                )
                if resp.status_code == 403:
                    print("  ❌ Invalid API key. Get one at https://cryptopanic.com")
                    return []
                if resp.status_code == 429:
                    print("  ⏳ Rate limited, waiting...")
                    await asyncio.sleep(60)
                    continue
                resp.raise_for_status()
                data = resp.json()
                posts = data.get("results", [])
                all_posts.extend(posts)
                print(f"  Page {page}: {len(posts)} posts")
                if len(posts) < 100:
                    break
            except Exception as e:
                print(f"  ⚠️  Page {page} error: {e}")
                break

    return all_posts


def process_posts(posts: list[dict]) -> list[dict]:
    """Convert posts to time-series rows."""
    import re
    from html import unescape

    rows = []
    for post in posts:
        ts_str = post.get("published_at", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            continue

        raw_sentiment = (post.get("votes", {}) or {}).get("sentiment", None)
        if raw_sentiment is None:
            sentiment = 0.0
        else:
            sentiment = SENTIMENT_MAP.get(raw_sentiment.lower(), 0.0)

        title = unescape(post.get("title", ""))
        domain = post.get("domain", "")
        source = post.get("source", {})

        rows.append({
            "ts": ts,
            "sentiment": sentiment,
            "title": title[:200],
            "domain": domain[:50],
            "kind": post.get("kind", "news"),
        })

    return rows


def aggregate_to_15m(rows: list[dict]) -> list[dict]:
    """Aggregate news to 15m buckets."""
    import pandas as pd
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    buckets = df.resample("15min").agg({
        "sentiment": "mean",
        "title": "count",
    }).rename(columns={"title": "news_count"}).reset_index()

    positive = df[df["sentiment"] > 0].resample("15min")["sentiment"].count().reset_index()
    negative = df[df["sentiment"] < 0].resample("15min")["sentiment"].count().reset_index()
    important = df[df["sentiment"].abs() >= 0.75].resample("15min")["sentiment"].count().reset_index()

    buckets = buckets.merge(positive.rename(columns={"sentiment": "news_positive_count"}),
                            on="ts", how="left")
    buckets = buckets.merge(negative.rename(columns={"sentiment": "news_negative_count"}),
                            on="ts", how="left")
    buckets = buckets.merge(important.rename(columns={"sentiment": "news_important_count"}),
                            on="ts", how="left")

    buckets = buckets.fillna(0)
    buckets["news_sentiment"] = buckets["sentiment"].round(3)
    buckets["news_positive_ratio"] = (
        buckets["news_positive_count"] / buckets["news_count"].replace(0, 1)
    ).round(2)
    buckets["news_negative_ratio"] = (
        buckets["news_negative_count"] / buckets["news_count"].replace(0, 1)
    ).round(2)

    out_cols = ["ts", "news_sentiment", "news_count", "news_positive_count",
                "news_negative_count", "news_important_count",
                "news_positive_ratio", "news_negative_ratio"]
    return buckets[[c for c in out_cols if c in buckets.columns]].to_dict("records")


async def main():
    parser = argparse.ArgumentParser(description="Register CryptoPanic News Sentiment")
    parser.add_argument("--pages", type=int, default=3,
                        help="API pages to fetch (100 posts each, default: 3)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="CryptoPanic API key (or set CRYPTOPANIC_API_KEY env)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("CRYPTOPANIC_API_KEY", "")
    if not api_key:
        print("❌ No API key. Set CRYPTOPANIC_API_KEY env var or pass --api-key")
        print("   Get one free at: https://cryptopanic.com")
        sys.exit(1)

    import asyncio

    print(f"📰 Fetching {args.pages} pages of news from CryptoPanic...")
    posts = await fetch_news(api_key, pages=args.pages)
    if not posts:
        print("  No posts received.")
        return

    print(f"  Total posts: {len(posts)}")

    rows = process_posts(posts)
    print(f"  Processed: {len(rows)} time-series rows")

    buckets = aggregate_to_15m(rows)

    import pandas as pd
    from src.ts_store import write as ts_write
    df = pd.DataFrame(buckets)
    n = ts_write(TS_ASSET_ID, df, frequency="15m")
    print(f"  Saved {n} 15m-aggregated rows to {TS_ASSET_ID}")

    from src.ts_catalog import refresh_catalog
    refresh_catalog()

    if not df.empty:
        print(f"\n✅ News sentiment registered.")
        print(f"  Range: {df['ts'].min()} → {df['ts'].max()}")
        print(f"  Avg sentiment: {df['news_sentiment'].mean():.3f}")
        print(f"  Total news items: {df['news_count'].sum():.0f}")


if __name__ == "__main__":
    asyncio.run(main())
