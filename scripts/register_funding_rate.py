"""Register Binance Funding Rate as a new data source.

This adds a derivatives:narrative that MCP agents can query alongside
market and sentiment data.

Usage:
    PYTHONPATH="$PWD" .venv/bin/python3 scripts/register_funding_rate.py
    PYTHONPATH="$PWD" .venv/bin/python3 scripts/register_funding_rate.py --fetch

The funding rate measures the cost of holding long/short positions.
Extreme positive → market is over-leveraged long (bearish signal).
Extreme negative → market is over-leveraged short (bullish signal).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SOURCE_NAME = "binance_funding_rate"
SOURCE_DISPLAY = "derivatives:binance:funding_rate"


async def fetch_funding_rate():
    """Fetch historical funding rate data from Binance Futures."""
    import httpx
    import pandas as pd
    from src.ts_store import write as ts_write

    async with httpx.AsyncClient() as client:
        all_data = []
        start_time = int((datetime.now(timezone.utc).timestamp() - 365 * 86400) * 1000)
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

        while start_time < end_time:
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={
                    "symbol": "BTCUSDT",
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1000,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break

            for entry in data:
                all_data.append({
                    "ts": datetime.fromtimestamp(entry["fundingTime"] / 1000, tz=timezone.utc),
                    "funding_rate": float(entry["fundingRate"]),
                })

            last_ts = int(data[-1]["fundingTime"])
            start_time = last_ts + 1

            if len(data) < 1000:
                break

            await asyncio.sleep(0.2)
            print(f"  Fetched {len(all_data)} rows so far...", end="\r")

    if not all_data:
        print("  No funding rate data received.")
        return 0

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    n = ts_write(SOURCE_NAME, df, frequency="raw")
    print(f"\n  Saved {n} funding rate rows to ts_store")
    print(f"  Range: {df['ts'].min()} → {df['ts'].max()}")
    return n


def main():
    parser = argparse.ArgumentParser(description="Register Binance Funding Rate source")
    parser.add_argument("--fetch", action="store_true", help="Fetch historical data")
    args = parser.parse_args()

    ts_asset_id = "derivatives:binance:funding_rate"
    print(f"✅ Asset ID: {ts_asset_id}")

    if args.fetch:
        print(f"\nFetching 1 year of funding rate history...")
        # Write to ts_store with proper asset_id format
        async def _fetch_and_store():
            import httpx
            import pandas as pd
            from src.ts_store import write as ts_write

            async with httpx.AsyncClient() as client:
                all_data = []
                start_time = int((datetime.now(timezone.utc).timestamp() - 365 * 86400) * 1000)
                end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

                while start_time < end_time:
                    resp = await client.get(
                        "https://fapi.binance.com/fapi/v1/fundingRate",
                        params={"symbol": "BTCUSDT", "startTime": start_time, "endTime": end_time, "limit": 1000},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if not data:
                        break
                    for entry in data:
                        all_data.append({
                            "ts": datetime.fromtimestamp(entry["fundingTime"] / 1000, tz=timezone.utc),
                            "funding_rate": float(entry["fundingRate"]),
                        })
                    last_ts = int(data[-1]["fundingTime"])
                    start_time = last_ts + 1
                    if len(data) < 1000:
                        break
                    await asyncio.sleep(0.2)

            if all_data:
                df = pd.DataFrame(all_data).drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
                n = ts_write(ts_asset_id, df, frequency="raw")
                print(f"\n  Saved {n} funding rate rows to {ts_asset_id}/raw")
                print(f"  Range: {df['ts'].min()} → {df['ts'].max()}")

        asyncio.run(_fetch_and_store())

        # Align to 15m
        from src.ts_aligner import align
        n = align(ts_asset_id, to_freq="15m")
        print(f"  Aligned to 15m: {n} rows")

        from src.ts_catalog import refresh_catalog
        refresh_catalog()

    print(f"\nDone! MCP agent can now query:")
    print(f"  get({{'btc': 'market:binance:btcusdt', 'funding': '{ts_asset_id}'}}, freq='15m')")


if __name__ == "__main__":
    main()
