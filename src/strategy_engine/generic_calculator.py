"""Generic Calculator — evaluates mathematical formulas on DataFrames.

Allows agents to define new indicators as formulas (strings) instead of
writing Python code. Formulas are evaluated in a restricted environment
with access only to DataFrame columns and basic math operations.

Usage:
    from src.strategy_engine.generic_calculator import calc_formula

    # Agent defines:
    formula = "close - close.shift(12)"  # momentum 12-period
    result = calc_formula(df, formula)

    # Or with parameters:
    formula = "(close - ema(close, {period})) / close * 100"
    result = calc_formula(df, formula, {"period": 20})
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Allowed functions in formulas
_ALLOWED_FUNCS = {
    # Math
    "abs": abs, "max": max, "min": min,
    "sum": sum, "round": round, "mean": np.mean,
    "sqrt": np.sqrt, "log": np.log, "log10": np.log10,
    # Pandas rolling
    "sma": lambda s, p: s.rolling(p, min_periods=p).mean(),
    "ema": lambda s, p: s.ewm(span=p, min_periods=p).mean(),
    "std": lambda s, p: s.rolling(p, min_periods=p).std(),
    "max_roll": lambda s, p: s.rolling(p, min_periods=p).max(),
    "min_roll": lambda s, p: s.rolling(p, min_periods=p).min(),
    # Shifts and diffs
    "shift": lambda s, p: s.shift(p),
    "diff": lambda s, p: s.diff(p),
    "pct_change": lambda s, p: s.pct_change(p),
    # Column access
    "col": lambda df, n: df[n] if isinstance(n, str) else df.iloc[:, n],
}

# Restricted builtins for eval
_RESTRICTED_BUILTINS = {
    "True": True, "False": False, "None": None,
    "abs": abs, "int": int, "float": float, "bool": bool,
    "len": len, "range": range, "list": list, "dict": dict,
    "min": min, "max": max, "sum": sum, "round": round,
    "isinstance": isinstance, "type": type,
}


def calc_formula(df: pd.DataFrame, formula: str, params: dict | None = None) -> pd.Series:
    """Evaluate a formula string against a DataFrame and return a Series.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + existing indicators. Column names are available as variables.
    formula : str
        Mathematical expression. Examples:
        - "close - close.shift(12)"
        - "sma(close, 20)"
        - "(close - ema(close, {period})) / close * 100"
        - "col(df, 'volume') > sma(col(df, 'volume'), 50)"
    params : dict, optional
        Parameters to substitute as {key} in the formula.

    Returns
    -------
    pd.Series
        Result of the formula evaluation.
    """
    if params:
        for k, v in params.items():
            formula = formula.replace(f"{{{k}}}", str(v))

    # Build namespace: all DataFrame columns as variables
    namespace = dict(_RESTRICTED_BUILTINS)
    namespace["df"] = df
    namespace["_all_funcs"] = _ALLOWED_FUNCS
    namespace["np"] = np
    namespace["pd"] = pd

    for col in df.columns:
        safe_name = col.replace(" ", "_").replace("-", "_")
        namespace[safe_name] = df[col]
        if safe_name != col:
            namespace[col] = df[col]

    for name, func in _ALLOWED_FUNCS.items():
        namespace[name] = func

    # Resolve indicator names from registry so formulas can use
    # shorthand names like "atr", "rsi", "vwap", etc.
    from src.strategy_engine.registry import INDICATOR_REGISTRY, calc_indicator
    for ind_name, ind_fn in INDICATOR_REGISTRY.items():
        if ind_name not in namespace:
            try:
                namespace[ind_name] = ind_fn(df, {})
            except Exception:
                pass

    # Safety checks
    _validate_formula(formula)

    try:
        result = eval(formula, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(f"Formula evaluation failed: {formula}\n  Error: {e}")

    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    if isinstance(result, (int, float, np.number)):
        return pd.Series(result, index=df.index)
    return pd.Series(result)


def _validate_formula(formula: str) -> None:
    """Basic safety validation: no imports, no exec, no file I/O."""
    forbidden = ["import ", "exec(", "open(", "write(", "__import__", "eval("]
    for token in forbidden:
        if token in formula:
            raise ValueError(f"Formula contains forbidden token: '{token}'")


def register_dynamic_indicator(name: str, formula: str, params: dict | None = None,
                                description: str = "") -> None:
    """Register a new indicator at runtime (no code restart needed).

    After registration, any strategy JSON can use the new indicator.

    Parameters
    ----------
    name : str
        Indicator name (used in strategy JSON as ``indicator`` field).
    formula : str
        Formula expression.
    params : dict, optional
        Default parameters.
    description : str
        Human-readable description.
    """
    from src.strategy_engine.registry import INDICATOR_REGISTRY

    def _dynamic_fn(df, p):
        merged = dict(params or {})
        merged.update(p or {})
        return calc_formula(df, formula, merged)

    INDICATOR_REGISTRY[name] = _dynamic_fn
    logger.info("Dynamic indicator registered: '%s' = %s", name, formula)
