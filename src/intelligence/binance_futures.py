"""Binance Futures data fetcher — funding rate, open interest, taker ratio, long/short ratio.

These are the highest-priority semantic features for anomaly detection.
Unlike the generic intelligence fetcher, this handles Binance-specific
API formats (arrays of objects with numeric timestamps, pagination).

Usage:
    from src.intelligence.binance_futures import fetch_funding_rate_history, fetch_all_derivatives
    df = fetch_funding_rate_history(symbol='BTCUSDT', days=365)
    results = fetch_all_derivatives(symbol='BTCUSDT')
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path("data/ts/derivatives")
FUTURES_BASE = "https://fapi.binance.com"

MAX_PAGES = 50
PAGE_LIMIT = 1000
REQUEST_TIMEOUT = 30
RATE_LIMIT_PAUSE = 0.3
_HIST_MAX_DAYS = 5


def _paginate_funding_rate(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    all_records = []
    current_start = start_ms
    for _ in range(MAX_PAGES):
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": PAGE_LIMIT,
        }
        try:
            resp = httpx.get(f"{FUTURES_BASE}/fapi/v1/fundingRate", params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("binance_futures: funding rate fetch error: %s", e)
            break

        if not data:
            break

        all_records.extend(data)

        last_ts = data[-1].get("fundingTime", 0)
        if last_ts == 0 or len(data) < PAGE_LIMIT:
            break

        current_start = last_ts + 1
        time.sleep(RATE_LIMIT_PAUSE)

    return all_records


def _paginate_hist_endpoint(url: str, symbol: str, period: str, start_ms: int, end_ms: int) -> list[dict]:
    all_records = []
    chunk_ms = _HIST_MAX_DAYS * 24 * 3600 * 1000
    current_start = start_ms

    while current_start < end_ms:
        current_end = min(current_start + chunk_ms, end_ms)
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": current_start,
            "endTime": current_end,
            "limit": 500,
        }
        try:
            resp = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("binance_futures: pagination error for %s: %s", url.split("/")[-1], e)
            break

        if not data:
            break

        all_records.extend(data)

        if len(data) < 500:
            break

        last_ts = data[-1].get("timestamp", 0)
        if last_ts == 0:
            break
        current_start = last_ts + 1
        time.sleep(RATE_LIMIT_PAUSE)

    return all_records


def fetch_funding_rate_history(symbol: str = "BTCUSDT", days: int = 365) -> pd.DataFrame:
    """Fetch historical funding rates from Binance Futures.

    Funding rate is posted every 8 hours. A year has ~1095 records.

    Returns DataFrame with columns: ts, funding_rate, mark_price
    """
    from datetime import timedelta
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info("binance_futures: fetching funding rate for %s (%d days)", symbol, days)
    records = _paginate_funding_rate(symbol, start_ms, end_ms)

    if not records:
        logger.warning("binance_futures: no funding rate data returned")
        return pd.DataFrame()

    rows = []
    for r in records:
        try:
            ts = pd.to_datetime(int(r["fundingTime"]), unit="ms", utc=True)
            fr = float(r["fundingRate"])
            mp = float(r.get("markPrice", 0)) if r.get("markPrice") else 0.0
            rows.append({"ts": ts, "funding_rate": fr, "mark_price": mp})
        except (ValueError, KeyError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    logger.info("binance_futures: got %d funding rate records for %s", len(df), symbol)

    _save_external("funding_rate", symbol, df)
    return df


def fetch_open_interest_history(symbol: str = "BTCUSDT", period: str = "1h", days: int = 5) -> pd.DataFrame:
    """Fetch historical open interest from Binance Futures.

    Period options: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
    Note: Binance limits /futures/data/ endpoints to ~5 days per chunk.

    Returns DataFrame with columns: ts, open_interest, open_interest_value
    """
    from datetime import timedelta
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info("binance_futures: fetching OI history for %s period=%s days=%d", symbol, period, days)

    records = _paginate_hist_endpoint(
        f"{FUTURES_BASE}/futures/data/openInterestHist", symbol, period, start_ms, end_ms
    )

    if not records:
        logger.warning("binance_futures: no OI history data returned")
        return pd.DataFrame()

    rows = []
    for r in records:
        try:
            ts = pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True)
            oi = float(r["sumOpenInterest"])
            oi_val = float(r.get("sumOpenInterestValue", 0))
            rows.append({"ts": ts, "open_interest": oi, "open_interest_value": oi_val})
        except (ValueError, KeyError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    logger.info("binance_futures: got %d OI records for %s", len(df), symbol)

    _save_external("open_interest", symbol, df)
    return df


def fetch_taker_ratio_history(symbol: str = "BTCUSDT", period: str = "1h", days: int = 5) -> pd.DataFrame:
    """Fetch taker buy/sell volume ratio from Binance Futures.

    Returns DataFrame with columns: ts, taker_buy_vol, taker_sell_vol, taker_ratio
    """
    from datetime import timedelta
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info("binance_futures: fetching taker ratio for %s period=%s days=%d", symbol, period, days)

    records = _paginate_hist_endpoint(
        f"{FUTURES_BASE}/futures/data/takerlongshortRatio", symbol, period, start_ms, end_ms
    )

    if not records:
        logger.warning("binance_futures: no taker ratio data returned")
        return pd.DataFrame()

    rows = []
    for r in records:
        try:
            ts = pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True)
            buy_vol = float(r.get("buyVol", 0))
            sell_vol = float(r.get("sellVol", 0))
            ratio = float(r.get("buySellRatio", 0))
            rows.append({"ts": ts, "taker_buy_vol": buy_vol, "taker_sell_vol": sell_vol, "taker_ratio": ratio})
        except (ValueError, KeyError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    logger.info("binance_futures: got %d taker ratio records for %s", len(df), symbol)

    _save_external("taker_ratio", symbol, df)
    return df


def fetch_long_short_ratio(symbol: str = "BTCUSDT", period: str = "1h", days: int = 5) -> pd.DataFrame:
    """Fetch top trader long/short account ratio from Binance Futures.

    Returns DataFrame with columns: ts, long_account, short_account, long_short_ratio
    """
    from datetime import timedelta
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    logger.info("binance_futures: fetching long/short ratio for %s period=%s days=%d", symbol, period, days)

    records = _paginate_hist_endpoint(
        f"{FUTURES_BASE}/futures/data/topLongShortAccountRatio", symbol, period, start_ms, end_ms
    )

    if not records:
        logger.warning("binance_futures: no long/short ratio data returned")
        return pd.DataFrame()

    rows = []
    for r in records:
        try:
            ts = pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True)
            long_acc = float(r.get("longAccount", 0))
            short_acc = float(r.get("shortAccount", 0))
            ratio = float(r.get("longShortRatio", 0))
            rows.append({"ts": ts, "long_account": long_acc, "short_account": short_acc, "long_short_ratio": ratio})
        except (ValueError, KeyError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    logger.info("binance_futures: got %d long/short ratio records for %s", len(df), symbol)

    _save_external("long_short_ratio", symbol, df)
    return df


def fetch_all_derivatives(symbol: str = "BTCUSDT", days: int = 365, oi_period: str = "1h") -> dict[str, pd.DataFrame]:
    """Fetch all derivatives data and return dict of DataFrames.

    Returns:
        {"funding_rate": df, "open_interest": df, "taker_ratio": df, "long_short_ratio": df}
    """
    results = {}

    logger.info("binance_futures: === Fetching all derivatives data for %s ===", symbol)

    results["funding_rate"] = fetch_funding_rate_history(symbol, days=days)

    oi_days = min(days, 5)
    results["open_interest"] = fetch_open_interest_history(symbol, period=oi_period, days=oi_days)
    results["taker_ratio"] = fetch_taker_ratio_history(symbol, period=oi_period, days=oi_days)
    results["long_short_ratio"] = fetch_long_short_ratio(symbol, period=oi_period, days=oi_days)

    for name, df in results.items():
        if df.empty:
            logger.warning("binance_futures: %s returned empty DataFrame", name)
        else:
            logger.info("binance_futures: %s → %d rows (%s → %s)",
                       name, len(df), df["ts"].iloc[0], df["ts"].iloc[-1])

    return results


def _save_external(source_name: str, symbol: str, df: pd.DataFrame) -> None:
    """Save fetched data to data/ts/derivatives/{source_name}/{symbol}/{YYYY-MM}.parquet."""
    if df.empty or "ts" not in df.columns:
        return

    out_dir = EXTERNAL_DIR / source_name / symbol.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    for month, group in df.groupby(df["ts"].dt.to_period("M")):
        path = out_dir / f"{month}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            existing["ts"] = pd.to_datetime(existing["ts"], utc=True)
            merged = pd.concat([existing, group], ignore_index=True)
            merged.drop_duplicates(subset="ts", keep="last", inplace=True)
            merged.sort_values("ts", inplace=True)
        else:
            merged = group.sort_values("ts")
        merged.to_parquet(path, compression="snappy")

    logger.info("binance_futures: saved %s/%s → %d rows", source_name, symbol, len(df))


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(description="Fetch Binance Futures derivatives data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=365, help="Days of funding rate history")
    parser.add_argument("--oi-days", type=int, default=5, help="Days of OI/ratio history (max ~5)")
    parser.add_argument("--period", default="1h", help="OI/taker period: 5m, 15m, 1h, 4h, 1d")
    parser.add_argument("--align", action="store_true", help="Align all fetched data to 15m")
    args = parser.parse_args()

    results = fetch_all_derivatives(args.symbol, days=args.days, oi_period=args.period)

    for name, df in results.items():
        if df.empty:
            print(f"  {name}: NO DATA")
        else:
            print(f"  {name}: {len(df)} rows ({df['ts'].iloc[0]} → {df['ts'].iloc[-1]})")

    if args.align:
        from src.intelligence.aligner import align_derivatives
        aligned = align_derivatives(args.symbol.lower(), timeframe="15m")
        for name, df in aligned.items():
            print(f"  aligned {name}: {len(df)} rows")


if __name__ == "__main__":
    main()