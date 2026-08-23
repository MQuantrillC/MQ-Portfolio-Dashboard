"""Portfolio mathematics.

Pure functions over numbers: no Streamlit, no network, no global state. This is
the layer worth testing, and it's where the arithmetic bugs live.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core.config import TRADING_DAYS


# --------------------------------------------------------------------------
# Annualisation
# --------------------------------------------------------------------------

def annualize_returns(daily_returns: pd.Series) -> float:
    """Arithmetic annualised return (the right input for the GBM drift)."""
    return float(daily_returns.mean() * TRADING_DAYS)


def annualize_volatility(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def compound_annual_growth_rate(daily_returns: pd.Series) -> float:
    """Geometric annualised return (CAGR) — what users read 'annual return' as."""
    if daily_returns.empty:
        return float("nan")
    total_growth = float((1 + daily_returns).prod())
    if total_growth <= 0:
        return float("nan")
    years = len(daily_returns) / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return total_growth ** (1 / years) - 1


# --------------------------------------------------------------------------
# Portfolio value
# --------------------------------------------------------------------------

def portfolio_index(prices: pd.DataFrame, weights: Sequence[float], base: float = 100.0) -> pd.Series:
    """Weighted portfolio value over time, normalised to ``base`` at the start.

    Each holding is converted to a growth multiple *first*, then weighted. Taking
    the dot product of raw prices and weights instead would build a
    price-weighted index, in which a $1,000 stock dominates a $20 stock
    regardless of how much of the portfolio each one is.
    """
    if prices.empty:
        return pd.Series(dtype="float64")

    weights = np.asarray(weights, dtype="float64")
    if weights.shape[0] != prices.shape[1]:
        raise ValueError(
            f"weights has {weights.shape[0]} entries but prices has "
            f"{prices.shape[1]} columns"
        )

    # Every holding needs a price on the base date. A stock that listed after
    # the start of the range has NaN there, and dividing by it turns the whole
    # weighted line into NaN — a silently blank chart rather than an error.
    prices = prices.dropna(how="any")
    if prices.empty:
        return pd.Series(dtype="float64")

    growth = prices / prices.iloc[0]
    return growth.dot(weights) * base


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------

def portfolio_volatility(weights, cov_matrix) -> float:
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def portfolio_return(weights, mean_returns) -> float:
    return float(np.dot(weights, mean_returns))


def get_optimized_weights(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_level: str,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Optimise weights for a risk level using mean-variance analysis.

    ``max_weight`` caps any single position. It matters most for the "High"
    level: maximising expected return with no cap has a trivial corner solution
    (everything in the single highest-return asset), which is not a portfolio.
    """
    n_assets = len(mean_returns)
    if n_assets == 0:
        return np.array([])
    if n_assets == 1:
        return np.array([1.0])

    mean_returns = np.asarray(mean_returns, dtype="float64")
    cov_matrix = np.asarray(cov_matrix, dtype="float64")

    # A cap below 1/n makes the problem infeasible; never go below equal weight.
    effective_cap = max(max_weight, 1.0 / n_assets)
    bounds = tuple((0.0, effective_cap) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    initial_weights = np.full(n_assets, 1.0 / n_assets)

    if risk_level == "Low":
        objective = lambda w: portfolio_volatility(w, cov_matrix)
    elif risk_level == "High":
        objective = lambda w: -portfolio_return(w, mean_returns)
    else:  # Moderate — maximise Sharpe
        def objective(w):
            vol = portfolio_volatility(w, cov_matrix)
            if vol <= 0:
                return 0.0
            return -(portfolio_return(w, mean_returns) - risk_free_rate) / vol

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        return initial_weights

    # Clean up float noise and renormalise so the weights sum to exactly 1.
    weights = np.clip(result.x, 0.0, effective_cap)
    total = weights.sum()
    return weights / total if total > 0 else initial_weights


def normalize_custom_weights(
    custom_weights: Dict[str, float], tickers: Sequence[str]
) -> np.ndarray:
    """Turn a {ticker: percent} mapping into weights for ``tickers``, summing to 1.

    Tickers missing from the mapping get 0. Renormalising means a portfolio whose
    data partly failed to load still reports coherent metrics.
    """
    raw = np.array([float(custom_weights.get(t, 0.0)) for t in tickers], dtype="float64")
    total = raw.sum()
    if total <= 0:
        return np.full(len(tickers), 1.0 / len(tickers)) if len(tickers) else np.array([])
    return raw / total


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def calculate_portfolio_metrics(
    returns: pd.DataFrame,
    weights: Sequence[float],
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Annual return, volatility, Sharpe, CAGR and max drawdown."""
    if returns.empty:
        return {}

    weights = np.asarray(weights, dtype="float64")
    if weights.shape[0] != returns.shape[1]:
        raise ValueError(
            f"weights has {weights.shape[0]} entries but returns has "
            f"{returns.shape[1]} columns"
        )

    portfolio_returns = returns.dot(weights)

    annual_return = annualize_returns(portfolio_returns)
    annual_vol = annualize_volatility(portfolio_returns)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0

    cumulative = (1 + portfolio_returns).cumprod()
    drawdowns = cumulative / cumulative.expanding().max() - 1
    max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "cagr": compound_annual_growth_rate(portfolio_returns),
        "risk_free_rate": risk_free_rate,
    }


def calculate_beta(
    asset_returns: pd.Series, market_returns: pd.Series, min_observations: int = 50
) -> Optional[float]:
    """Beta of an asset against the market, or None if there isn't enough overlap."""
    common = asset_returns.index.intersection(market_returns.index)
    if len(common) < min_observations:
        return None

    asset = asset_returns.loc[common]
    market = market_returns.loc[common]
    market_variance = float(market.var())
    if market_variance <= 0:
        return None

    return round(float(asset.cov(market)) / market_variance, 2)


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------

def monte_carlo_simulation(
    start_value: float,
    mean_return: float,
    volatility: float,
    years: int = 10,
    simulations: int = 500,
    steps_per_year: int = TRADING_DAYS,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate portfolio growth under geometric Brownian motion.

    ``mean_return`` is the arithmetic annual return, so the log-drift carries the
    -0.5 * sigma^2 Ito correction.

    Vectorised: one RNG draw and one cumulative sum, rather than a Python loop
    per simulation per step (which at the UI maximum is ~38 million iterations).

    Returns an array of shape (simulations, steps).
    """
    steps = int(steps_per_year * years)
    if steps <= 0 or simulations <= 0:
        return np.empty((max(simulations, 0), max(steps, 0)))

    rng = np.random.default_rng(seed)
    dt = 1.0 / steps_per_year
    drift = (mean_return - 0.5 * volatility ** 2) * dt
    diffusion = volatility * np.sqrt(dt)

    paths = np.empty((simulations, steps), dtype="float64")
    paths[:, 0] = start_value
    if steps > 1:
        shocks = rng.normal(drift, diffusion, size=(simulations, steps - 1))
        paths[:, 1:] = start_value * np.exp(np.cumsum(shocks, axis=1))
    return paths
