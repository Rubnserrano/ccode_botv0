"""Register Binance Whale Trades — large transactions from aggTrades.

Tracks trades >$100k from Binance's public aggTrades endpoint.
No API key needed. Uses existing Binance infrastructure.

Metrics (aggregated to 15m):
  - whale_trade_count    → number of large trades
  - whale_volume_usd     → total USD volume of large trades
  - whale_buy_volume     → USD volume of buy-aggressive trades
  - whale_sell_volume    → USD volume of sell-aggressive trades
  - whale_buy_sell_ratio → buy_volume / (sell_volume + 1)
  - whale_avg_size_usd   → average large trade size

Usage:
    PYTHONPATH="$PWD" .venv/bin/python3 scripts/register_whale_trades.py --minutes 360
    PYTHONPATH="$PWD" .venv/bin/python3 scripts/register_whale_trades.py --minutes 60 --whale-threshold 50000
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TS_ASSET_ID = "onchain:binance:whale_trades"


async def fetch_agg_trades(symbol: str, target_trades: int = 50000) -> list[dict]:
    """Fetch aggTrades going backwards from most recent using fromId pagination."""
    import httpx

    all_trades = []
    seen_ids = set()

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: get the most recent 1000 to find current max tradeId
        resp = await client.get(
            "https://api.binance.com/api/v3/aggTrades",
            params={"symbol": symbol, "limit": 1000},
        )
        resp.raise_for_status()
        recent = resp.json()
        if not recent:
            return []

        max_id = recent[-1]["a"]
        all_trades.extend(recent)
        seen_ids.add(recent[-1]["a"])
        from_id = max_id - 1001  # go backwards

        print(f"  Latest trade ID: {max_id:,}")

        # Step 2: paginate backwards
        bar = None
        try:
            from tqdm import tqdm
            bar = tqdm(total=target_trades, desc="  Fetching", unit="trades")
            bar.update(len(all_trades))
        except ImportError:
            pass

        while len(all_trades) < target_trades and from_id > 0:
            try:
                resp = await client.get(
                    "https://api.binance.com/api/v3/aggTrades",
                    params={"symbol": symbol, "fromId": max(0, from_id), "limit": 1000},
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break

                # Filter duplicates
                new_trades = [t for t in batch if t["a"] not in seen_ids]
                if not new_trades:
                    break

                seen_ids.update(t["a"] for t in new_trades)
                all_trades.extend(new_trades)
                if bar:
                    bar.update(len(new_trades))
                else:
                    print(f"  Fetched {len(all_trades):,}/{target_trades:,}", end="\r")

                from_id = new_trades[0]["a"] - len(new_trades) - 1

                await asyncio.sleep(0.05)  # rate limit courtesy

            except Exception as e:
                print(f"\n  Fetch error: {e}")
                await asyncio.sleep(2)
                continue

        if bar:
            bar.close()

    return all_trades


def process_trades(all_trades: list[dict], threshold: float = 100000) -> list[dict]:
    """Filter whale trades and aggregate."""
    whale_rows = []
    for t in all_trades:
        price = float(t["p"])
        qty = float(t["q"])
        usd_value = price * qty
        if usd_value < threshold:
            continue
        ts = datetime.fromtimestamp(t["T"] / 1000, tz=timezone.utc)
        is_buy = not t["m"]
        whale_rows.append({
            "ts": ts,
            "usd_value": usd_value,
            "is_buy": is_buy,
            "qty_btc": qty,
        })

    return whale_rows


def aggregate_to_15m(whale_rows: list[dict]) -> list[dict]:
    """Aggregate whale trades to 15m buckets."""
    import pandas as pd
    if not whale_rows:
        return []
    df = pd.DataFrame(whale_rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    buckets = df.resample("15min").agg({
        "usd_value": "sum",
        "qty_btc": "sum",
        "is_buy": "count",
    }).rename(columns={"is_buy": "whale_trade_count"}).reset_index()

    buy_vol = df[df["is_buy"]].resample("15min")["usd_value"].sum().reset_index()
    sell_vol = df[~df["is_buy"]].resample("15min")["usd_value"].sum().reset_index()

    buckets = buckets.merge(buy_vol.rename(columns={"usd_value": "whale_buy_volume"}),
                            on="ts", how="left")
    buckets = buckets.merge(sell_vol.rename(columns={"usd_value": "whale_sell_volume"}),
                            on="ts", how="left")

    buckets = buckets.fillna(0)
    buckets["whale_volume_usd"] = buckets["whale_buy_volume"] + buckets["whale_sell_volume"]
    buckets["whale_buy_sell_ratio"] = (
        buckets["whale_buy_volume"] / (buckets["whale_sell_volume"] + 1)
    ).round(2)
    buckets["whale_avg_size_usd"] = (
        buckets["whale_volume_usd"] / buckets["whale_trade_count"].replace(0, 1)
    ).round(0)

    return buckets.to_dict("records")


async def main():
    parser = argparse.ArgumentParser(description="Register Binance Whale Trades")
    parser.add_argument("--trades", type=int, default=50000,
                        help="Number of aggTrades to fetch (default: 50000, ~few hours)")
    parser.add_argument("--whale-threshold", type=int, default=50000,
                        help="Min trade size in USD (default: $50000)")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    threshold = args.whale_threshold
    target = min(args.trades, 200000)

    print(f"🐳 Fetching {target:,} aggTrades for {args.symbol}...")
    all_trades = await fetch_agg_trades(args.symbol, target_trades=target)
    print(f"\n  Total trades: {len(all_trades):,}")

    whale_rows = process_trades(all_trades, threshold=threshold)
    print(f"  Whale trades (>=${threshold:,}): {len(whale_rows):,}")

    if not whale_rows:
        print("  No whale trades found in this period.")
        return

    buckets = aggregate_to_15m(whale_rows)

    import pandas as pd
    from src.ts_store import write as ts_write
    df = pd.DataFrame(buckets)
    n = ts_write(TS_ASSET_ID, df, frequency="15m")
    print(f"  Saved {n} 15m-aggregated rows to {TS_ASSET_ID}")

    from src.ts_catalog import refresh_catalog
    refresh_catalog()

    print(f"\n✅ Whale trades registered.")
    print(f"  Range: {df['ts'].min()} → {df['ts'].max()}")
    print(f"  Avg whale/15m: ${df['whale_volume_usd'].mean():.0f}")
    print(f"  Buy/Sell ratio avg: {df['whale_buy_sell_ratio'].mean():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
