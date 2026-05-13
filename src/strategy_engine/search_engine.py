"""Search engine — expands hypotheses into concrete strategy implementations.

Modes:
  - exploit: vary parameters within ranges (20-50 variants)
  - mutation: add volume filter, regime filter, invert conditions, vary lookbacks
"""
from __future__ import annotations

import copy
import logging
import random
from typing import Any

from src.strategy_engine.hypothesis import (
    HypothesisFamily, get_template, get_family,
)

logger = logging.getLogger(__name__)

_MAX_STRATEGIES_PER_HYPOTHESIS = 40


def expand_hypothesis(
    family_name: str,
    params: dict[str, Any] | None = None,
    n_variants: int = 20,
) -> list[dict]:
    """Expand a hypothesis into N concrete strategy definitions.

    Args:
        family_name: Name of the hypothesis family.
        params: Optional override params from LLM.
        n_variants: How many concrete strategies to generate.

    Returns:
        List of strategy dicts.
    """
    family = get_family(family_name)
    if family is None:
        logger.warning("search: unknown family '%s'", family_name)
        return []

    template = get_template(family)
    if not template:
        logger.warning("search: no template for '%s'", family_name)
        return []

    n_variants = min(n_variants, _MAX_STRATEGIES_PER_HYPOTHESIS)

    # Merge LLM params with template defaults
    merged_params = {}
    for pname, pdef in template.get("params", {}).items():
        if params and pname in params:
            merged_params[pname] = params[pname]
        else:
            merged_params[pname] = pdef["default"]

    strategies = []
    base_name = family_name.replace("_", " ").title()

    # 1) Exploit variants: vary parameters smoothly
    for i in range(n_variants // 2):
        variant_params = _vary_params(merged_params, template["params"], i)
        strategy = _build_strategy(base_name, i, template, variant_params)
        if strategy:
            strategies.append(strategy)

    # 2) Mutation variants: structural changes
    for i in range(n_variants // 2):
        base = _build_strategy(base_name, i + 100, template, merged_params)
        if base:
            mutated = _apply_mutation(base, template, merged_params, i)
            if mutated:
                strategies.append(mutated)

    logger.info("search: expanded '%s' into %d strategies", family_name, len(strategies))
    return strategies


def _vary_params(
    base: dict[str, Any],
    param_defs: dict[str, dict],
    seed: int,
) -> dict[str, Any]:
    """Slightly vary parameters around the base values."""
    rng = random.Random(seed)
    varied = {}
    for pname, pdef in param_defs.items():
        base_val = base.get(pname, pdef["default"])
        r = pdef.get("range", [base_val * 0.5, base_val * 1.5])
        ptype = pdef.get("type", float)

        if ptype == int:
            low = max(int(r[0]), int(base_val * 0.7))
            high = min(int(r[1]), int(base_val * 1.3))
            if low >= high:
                low, high = int(r[0]), int(r[1])
            varied[pname] = rng.randint(low, high)
        else:
            low = max(float(r[0]), float(base_val) * 0.7)
            high = min(float(r[1]), float(base_val) * 1.3)
            if low >= high:
                low, high = float(r[0]), float(r[1])
            varied[pname] = round(rng.uniform(low, high), 4)

    return varied


def _apply_mutation(
    strategy: dict,
    template: dict,
    params: dict[str, Any],
    seed: int,
) -> dict | None:
    """Apply simple mutations to a base strategy.

    Mutations:
      - Invert a condition (gt → lt, cross_above → cross_below)
      - Add volume filter
      - Add regime filter
      - Vary lookback periods dynamically
    """
    rng = random.Random(seed + 999)
    mutated = copy.deepcopy(strategy)
    conditions = mutated.get("entry_conditions", [])
    if not conditions:
        return None

    mutation_type = rng.randint(0, 3)

    if mutation_type == 0 and conditions:
        # Invert first condition
        c = conditions[0]
        invert_map = {"gt": "lt", "lt": "gt", "gte": "lte", "lte": "gte",
                      "cross_above": "cross_below", "cross_below": "cross_above"}
        if c.get("op") in invert_map:
            c["op"] = invert_map[c["op"]]
            if isinstance(c.get("value"), (int, float)):
                c["value"] = round(c["value"] * rng.uniform(0.8, 1.2), 2)
            mutated["name"] = mutated.get("name", "?") + "_inv"
            return mutated

    elif mutation_type == 1 and len(conditions) <= 2:
        # Add volume filter (only if not already present)
        has_vol = any(c.get("indicator") == "volume" for c in conditions)
        if not has_vol:
            vol_period = rng.randint(10, 50)
            conditions.append({
                "indicator": "volume",
                "op": "gt_rolling",
                "rolling": "sma",
                "period": vol_period,
            })
            mutated["name"] = mutated.get("name", "?") + "_vol"
            return mutated

    elif mutation_type == 2:
        # Add regime filter
        regime_filter = {"indicator": "regime", "op": "in", "value": [1, 2]}
        if regime_filter not in conditions:
            conditions.append(regime_filter)
            mutated["name"] = mutated.get("name", "?") + "_reg"
            return mutated

    elif mutation_type == 3:
        # Vary TP/SL ratio
        tp = params.get("tp_pct", 0.04)
        sl = params.get("sl_pct", 0.02)
        new_tp = round(tp * rng.uniform(1.5, 3.0), 3)
        new_sl = round(sl * rng.uniform(0.8, 1.5), 3)
        mutated["exit"]["tp_pct"] = new_tp
        mutated["exit"]["sl_pct"] = new_sl
        mutated["name"] = mutated.get("name", "?") + "_ts"
        return mutated

    return None


def _build_strategy(
    base_name: str,
    index: int,
    template: dict,
    params: dict[str, Any],
) -> dict | None:
    """Build a strategy dict from a template + filled params."""
    conds_template = template.get("conditions_template", [])
    if not conds_template:
        return None

    conditions = []
    for ct in conds_template:
        c = copy.deepcopy(ct)
        for key in list(c.keys()):
            val = c[key]
            if isinstance(val, str) and val.startswith("{") and val.endswith("}"):
                pname = val[1:-1]
                c[key] = params.get(pname, val)
        conditions.append(c)

    strategy = {
        "name": f"{base_name.replace(' ', '_').lower()}_v{index + 1}",
        "entry_conditions": conditions,
        "exit": {
            "tp_pct": params.get("tp_pct", 0.04),
            "sl_pct": params.get("sl_pct", 0.02),
            "horizon_bars": params.get("horizon_bars", 24),
        },
        "archetype": base_name.lower().replace(" ", "_"),
    }

    feature_reqs = template.get("feature_requirements", [])
    config = template.get("config", {})
    needs_indicator = config.get("needs_new_indicator", False) or bool(feature_reqs)

    if needs_indicator and feature_reqs:
        req = feature_reqs[0]
        formula = req["formula"]
        for pname, pval in params.items():
            formula = formula.replace(f"{{{pname}}}", str(pval))
        strategy["new_indicator"] = {
            "name": req["name"],
            "formula": formula,
            "description": req.get("description", ""),
        }
    elif needs_indicator and config.get("new_indicator_formula"):
        formula = config["new_indicator_formula"]
        for pname, pval in params.items():
            formula = formula.replace(f"{{{pname}}}", str(pval))
        strategy["new_indicator"] = {
            "name": f"{strategy['archetype']}_signal",
            "formula": formula,
        }

    return strategy
