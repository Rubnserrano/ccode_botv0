"""Evolution engine — mutate, crossover, and select strategies.

Operates on parameter spaces, not DSL trees. Each strategy has a
``param_space`` dict that defines valid ranges. The engine explores
variations by tweaking parameters within those ranges.

Future: when DSL arrives, this same engine will mutate/crossover
DSL AST nodes instead of parameter dicts.
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field

import pandas as pd

from src.backtesting.engine import backtest as run_backtest
from src.backtesting.strategies import STRATEGY_REGISTRY, FN_MAP
from src.backtesting.strategy import Strategy

logger = logging.getLogger(__name__)

# Default parameter spaces for each strategy
PARAM_SPACES: dict[str, dict] = {
    "rsi_mean_reversion": {
        "rsi_period": (5, 30),
        "oversold": (15, 40),
        "overbought": (60, 85),
    },
    "macd_crossover": {
        "fast": (5, 20),
        "slow": (15, 40),
        "signal": (5, 15),
    },
    "ema_trend": {
        "fast": (3, 15),
        "medium": (15, 30),
        "slow": (30, 100),
    },
    "vwap_bounce": {
        "deviation_pct": (0.005, 0.03),
    },
    "heikin_ashi_streak": {
        "min_streak": (2, 6),
    },
}

# Probability of mutating a given param
_MUTATION_RATE = 0.4


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def mutate_params(params: dict, param_space: dict) -> dict:
    """Return a new params dict with random mutations applied."""
    new = dict(params)
    for key, (lo, hi) in param_space.items():
        if random.random() < _MUTATION_RATE:
            value = new.get(key, (lo + hi) / 2)
            if isinstance(value, int):
                step = max(1, int((hi - lo) * 0.1))
                new[key] = _clamp(value + random.choice([-step, step]), lo, hi)
            else:
                step = (hi - lo) * 0.1
                new[key] = round(_clamp(value + random.uniform(-step, step), lo, hi), 4)
    return new


def crossover_params(p1: dict, p2: dict) -> dict:
    """Crossover: for each param, pick randomly from one parent."""
    child = {}
    for key in set(list(p1.keys()) + list(p2.keys())):
        child[key] = random.choice([p1.get(key), p2.get(key)])
    return child


def make_strategy(
    base_name: str,
    params: dict | None = None,
    name: str | None = None,
) -> Strategy | None:
    """Create a new ``Strategy`` from a template name and optional params.

    If ``base_name`` is an existing strategy, starts from its current params.
    """
    if base_name not in STRATEGY_REGISTRY:
        logger.warning("evolve: unknown strategy '%s'", base_name)
        return None
    base = STRATEGY_REGISTRY[base_name]
    merged = dict(base.params)
    if params:
        merged.update(params)
    return Strategy(
        name=name or f"{base_name}_evolved",
        params=merged,
        description=base.description,
        fn=FN_MAP.get(base_name),
    )


@dataclass
class EvolveConfig:
    n_generations: int = 5
    pop_size: int = 20
    elite_ratio: float = 0.2
    mutation_rate: float = 0.4
    horizon: int = 12
    warmup: int = 50
    cooldown: int = 2
    size_usdc: float = 50.0


@dataclass
class Individual:
    strategy: Strategy
    fitness: float = -999.0
    sharpe: float = 0.0
    n_trades: int = 0


def evolve(
    df: pd.DataFrame,
    base_name: str,
    config: EvolveConfig | None = None,
) -> list[Individual]:
    """Run evolution for a single base strategy.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicators.
    base_name : str
        Strategy name in ``STRATEGY_REGISTRY``.
    config : EvolveConfig, optional

    Returns
    -------
    list[Individual]
        Population of the final generation, sorted by fitness descending.
    """
    cfg = config or EvolveConfig()
    space = PARAM_SPACES.get(base_name)
    if space is None:
        logger.warning("evolve: no param space for '%s'", base_name)
        return []

    base = STRATEGY_REGISTRY[base_name]
    pop: list[Individual] = []

    def _fitness(ind: Individual) -> float:
        """Evaluate fitness by running backtest."""
        trades, summary = run_backtest(
            df, ind.strategy,
            horizon=cfg.horizon,
            warmup=cfg.warmup,
            cooldown=cfg.cooldown,
            size_usdc=cfg.size_usdc,
        )
        ind.n_trades = summary.get("n_trades", 0)
        ind.sharpe = summary.get("sharpe", 0.0)
        if ind.n_trades < 30:
            ind.fitness = -999.0
        else:
            ind.fitness = ind.sharpe
        return ind.fitness

    def _random_params() -> dict:
        return {k: random.randint(lo, hi) if isinstance(lo, int) else random.uniform(lo, hi)
                for k, (lo, hi) in space.items()}

    logger.info("evolve: starting '%s' pop=%d gen=%d", base_name, cfg.pop_size, cfg.n_generations)

    # Initialize population
    for i in range(cfg.pop_size):
        params = _random_params()
        s = make_strategy(base_name, params, name=f"{base_name}_gen0_{i}")
        if s:
            pop.append(Individual(strategy=s))

    n_gen = 0
    while n_gen < cfg.n_generations:
        n_gen += 1

        # Fitness (sequential)
        for ind in pop:
            _fitness(ind)

        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        logger.info("  gen %d: best fitness=%.4f (sharpe=%.2f, trades=%d)",
                     n_gen, best.fitness, best.sharpe, best.n_trades)

        if n_gen == cfg.n_generations:
            break

        # Selection & breeding
        n_elite = max(2, int(cfg.pop_size * cfg.elite_ratio))
        elite = pop[:n_elite]
        rest = pop[n_elite:]

        # Mutate elite
        children = []
        for ind in elite:
            mutated = mutate_params(ind.strategy.params, space)
            child = make_strategy(base_name, mutated, name=f"{base_name}_gen{n_gen}_mut")
            if child:
                children.append(Individual(strategy=child))

        # Crossover
        while len(children) < cfg.pop_size:
            p1, p2 = random.sample(elite, 2)
            c_params = crossover_params(p1.strategy.params, p2.strategy.params)
            child = make_strategy(base_name, c_params, name=f"{base_name}_gen{n_gen}_cross")
            if child:
                children.append(Individual(strategy=child))

        # Fill remaining with random
        while len(children) < cfg.pop_size:
            rnd_params = _random_params()
            child = make_strategy(base_name, rnd_params, name=f"{base_name}_gen{n_gen}_rand")
            if child:
                children.append(Individual(strategy=child))

        pop = children[:cfg.pop_size]

    # Final fitness
    for ind in pop:
        _fitness(ind)
    pop.sort(key=lambda x: x.fitness, reverse=True)
    return pop
