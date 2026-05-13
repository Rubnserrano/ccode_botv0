"""
Dashboard for paper trading — reads data/paper_state.json + equity.parquet.

Completely decoupled: zero imports from src/.
If you replace this with Next.js / React / whatever, the paper engine
doesn't need any changes — it only writes files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STATE_PATH = DATA_DIR / "paper_state.json"
EQUITY_PATH = DATA_DIR / "parquet" / "paper_equity.parquet"


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"error": "No paper state yet — start the paper runner first"}


def _load_equity() -> pd.DataFrame:
    if EQUITY_PATH.exists():
        return pd.read_parquet(EQUITY_PATH)
    return pd.DataFrame()


st.set_page_config(page_title="Paper Trading", layout="wide")
st.title("📊 Paper Trading")

state = _load_state()
equity = _load_equity()

if "error" in state:
    st.warning(state["error"])
    st.stop()

# ── Top metrics ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Capital", f"${state.get('capital', 0):.2f}")
col2.metric("Equity", f"${state.get('equity', 0):.2f}",
            delta=f"{state.get('equity', 0) - state.get('capital', 0):+.2f}")
col3.metric("Total P&L", f"${state.get('total_pnl', 0):.2f}")
col4.metric("Trades", f"{state.get('n_trades', 0)}")

# ── Position ──
pos = state.get("position")
if pos:
    st.subheader("Open Position")
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Side", pos["side"])
    pc2.metric("Entry", f"${pos['entry_price']:.2f}")
    pc3.metric("Size", f"${pos['size_usdc']:.2f}")
    pc4.metric("Held", f"{pos['bars_held']}/{pos['horizon_bars']} bars")
else:
    st.info("No open position")

# ── Equity curve ──
if not equity.empty:
    st.subheader("Equity Curve")

    chart_df = equity.set_index("ts")[["capital", "equity"]].reset_index()

    min_val = chart_df[["capital", "equity"]].min().min()
    max_val = chart_df[["capital", "equity"]].max().max()
    pad = (max_val - min_val) * 0.1 or 100

    chart = (
        alt.Chart(chart_df)
        .transform_fold(["capital", "equity"], as_=["variable", "value"])
        .mark_line()
        .encode(
            x=alt.X("ts:T", title=None),
            y=alt.Y("value:Q", title="USD", scale=alt.Scale(domain=[min_val - pad, max_val + pad])),
            color=alt.Color("variable:N", legend=alt.Legend(title=None)),
        )
        .properties(height=400)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    with st.expander("Recent Data"):
        st.dataframe(equity.tail(20).round(2), use_container_width=True)

# ── Raw state ──
with st.expander("Raw State"):
    st.json(state)
