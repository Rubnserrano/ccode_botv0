# ccode_botv0 — BTC Data Ingestion Engine

Clean rewrite of ccode_bot. Focused on BTC data ingestion:
- **Historical**: Download OHLCV from Binance REST → monthly-partitioned Parquet
- **Real-time**: WebSocket streams for price, trades, and 1m candles
- **Lightweight**: Minimal dependencies, no external databases

Layout: `data/raw/{exchange}/{symbol}/{YYYY-MM}.parquet`
