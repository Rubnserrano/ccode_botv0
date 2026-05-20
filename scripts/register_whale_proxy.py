"""Register Whale Activity Proxy — computed from existing BTC volume data.

No necesita APIs externas. Usa los datos de volumen que ya tenemos.
La lógica: volumen anómalo en velas de 15m = probable actividad ballena.

Métricas (storeadas como onchain:proxy:whale_activity):
  - volume_zscore       → desviación del volumen respecto a su media histórica
  - volume_ratio        → volumen / media móvil de 48h
  - whale_alert         → 1 si volume_zscore > 2 (actividad anómala)
  - whale_signal        → 1 si whale_alert + dirección del precio confirman

Usage:
    PYTHONPATH="$PWD" .venv/bin/python3 scripts/register_whale_proxy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TS_ASSET_ID = "onchain:proxy:whale_activity"


def compute_whale_proxy():
    """Compute whale activity proxy from existing BTC 15m volume data."""
    import pandas as pd
    from src.ts_store import read as ts_read, write as ts_write

    # Read existing BTC 15m data
    df = ts_read("market:binance:btcusdt", frequency="15m")
    if df.empty or len(df) < 500:
        print("  Insufficient BTC 15m data.")
        return 0

    print(f"  Loaded {len(df):,} rows of BTC 15m data")

    # Compute rolling statistics on volume
    df["volume_ma_48h"] = df["volume"].rolling(window=192, min_periods=48).mean()  # 48h = 192 * 15m
    df["volume_std_48h"] = df["volume"].rolling(window=192, min_periods=48).std()
    df["volume_zscore"] = ((df["volume"] - df["volume_ma_48h"]) / df["volume_std_48h"].replace(0, 1)).round(2)
    df["volume_ratio"] = (df["volume"] / df["volume_ma_48h"].replace(0, 1)).round(2)
    df["whale_alert"] = (df["volume_zscore"] > 2).astype(int)

    # Whale signal: whale_alert + price moving in same direction
    df["price_change"] = df["close"].pct_change().fillna(0)
    df["whale_signal"] = ((df["whale_alert"] == 1) & (df["price_change"].abs() > 0.001)).astype(int)

    # Also add: large single-candle volume (in USD)
    df["volume_usd"] = df["volume"] * df["close"]
    df["volume_usd_ma"] = df["volume_usd"].rolling(window=192, min_periods=48).mean()
    df["volume_usd_ratio"] = (df["volume_usd"] / df["volume_usd_ma"].replace(0, 1)).round(2)

    # Build output
    out = df[["ts", "volume_zscore", "volume_ratio", "whale_alert",
              "whale_signal", "volume_usd", "volume_usd_ratio"]].copy()
    out = out.dropna(subset=["volume_zscore"]).reset_index(drop=True)

    n = ts_write(TS_ASSET_ID, out, frequency="15m")
    print(f"  Written {n} rows to {TS_ASSET_ID}")

    # Stats
    alerts = out["whale_alert"].sum()
    signals = out["whale_signal"].sum()
    print(f"  Whale alerts: {alerts} ({alerts/len(out)*100:.1f}% of rows)")
    print(f"  Whale signals: {signals}")
    print(f"  Avg volume Z-score (last 96 bars): {out['volume_zscore'].tail(96).mean():.2f}")
    return n


def main():
    print(f"🐋 Computing whale activity proxy from BTC volume data...")
    n = compute_whale_proxy()
    if n:
        from src.ts_catalog import refresh_catalog
        refresh_catalog()
        print(f"\n✅ Whale proxy registered as '{TS_ASSET_ID}'")
        print(f"  Agent can query: query_market(assets={{'whale': '{TS_ASSET_ID}'}})")
    else:
        print("❌ Failed to compute whale proxy.")


if __name__ == "__main__":
    main()
