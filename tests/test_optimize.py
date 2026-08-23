"""Tests for the portfolio maths.

Every function under test is pure, so there's no network and no Streamlit here.
Several of these pin down bugs that shipped in the original single-file version.
"""

import numpy as np
import pandas as pd
import pytest

from core.optimize import (
    calculate_beta,
    calculate_portfolio_metrics,
    get_optimized_weights,
    monte_carlo_simulation,
    normalize_custom_weights,
    portfolio_index,
)


# --------------------------------------------------------------------------
# portfolio_index — the benchmark chart bug
# --------------------------------------------------------------------------

def test_portfolio_index_weights_returns_not_prices():
    """A 50/50 split of a +40% stock and a flat stock is worth +20%.

    The original code took the dot product of raw prices and weights, which made
    the answer depend on nominal share price: it returned 100.4 here because the
    $1,000 stock swamped the $10 one.
    """
    prices = pd.DataFrame({"CHEAP": [10.0, 14.0], "PRICEY": [1000.0, 1000.0]})
    result = portfolio_index(prices, [0.5, 0.5])
    assert result.iloc[0] == pytest.approx(100.0)
    assert result.iloc[-1] == pytest.approx(120.0)


def test_portfolio_index_is_invariant_to_share_price():
    """Splitting a stock 10-for-1 must not change portfolio performance."""
    prices = pd.DataFrame({"A": [100.0, 150.0], "B": [50.0, 60.0]})
    split = pd.DataFrame({"A": [10.0, 15.0], "B": [50.0, 60.0]})
    weights = [0.5, 0.5]
    assert portfolio_index(prices, weights).iloc[-1] == pytest.approx(
        portfolio_index(split, weights).iloc[-1]
    )


def test_portfolio_index_rejects_mismatched_weights():
    prices = pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, 2.0]})
    with pytest.raises(ValueError, match="3 entries"):
        portfolio_index(prices, [0.3, 0.3, 0.4])


def test_portfolio_index_handles_empty_input():
    assert portfolio_index(pd.DataFrame(), []).empty


def test_portfolio_index_survives_a_holding_that_listed_late():
    """A stock with no price on the base date used to blank the entire chart.

    Dividing by a NaN base makes every weighted value NaN, so the benchmark
    comparison rendered as an empty plot with no error anywhere.
    """
    prices = pd.DataFrame({
        "OLD": [100.0, 110.0, 120.0, 130.0],
        "NEW": [np.nan, np.nan, 50.0, 60.0],
    })
    result = portfolio_index(prices, [0.5, 0.5])
    assert not result.empty
    assert not result.isna().any()
    assert result.iloc[0] == pytest.approx(100.0)
    # Measured from the first date both holdings exist: OLD +8.33%, NEW +20%.
    assert result.iloc[-1] == pytest.approx(0.5 * (130 / 120) * 100 + 0.5 * (60 / 50) * 100)


def test_portfolio_index_returns_empty_when_holdings_never_overlap():
    prices = pd.DataFrame({
        "A": [100.0, 110.0, np.nan, np.nan],
        "B": [np.nan, np.nan, 50.0, 60.0],
    })
    assert portfolio_index(prices, [0.5, 0.5]).empty


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------

@pytest.fixture
def sample_market():
    mean_returns = pd.Series({"LOW": 0.05, "MID": 0.10, "HIGH": 0.30})
    cov = pd.DataFrame(
        [[0.010, 0.002, 0.004],
         [0.002, 0.040, 0.010],
         [0.004, 0.010, 0.160]],
        index=mean_returns.index, columns=mean_returns.index,
    )
    return mean_returns, cov


def test_weights_always_sum_to_one(sample_market):
    mean_returns, cov = sample_market
    for level in ("Low", "Moderate", "High"):
        weights = get_optimized_weights(mean_returns, cov, level, max_weight=0.4)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= -1e-9).all()


def test_high_risk_respects_the_position_cap(sample_market):
    """Uncapped return-maximisation collapses to 100% in one stock.

    That is a mathematically correct answer to the wrong question, and it made
    the 'High' strategy produce a one-slice pie chart.
    """
    mean_returns, cov = sample_market
    weights = get_optimized_weights(mean_returns, cov, "High", max_weight=0.4)
    assert weights.max() <= 0.4 + 1e-6
    assert (weights > 0.01).sum() >= 2


def test_low_risk_is_less_volatile_than_high_risk(sample_market):
    mean_returns, cov = sample_market
    cov_array = cov.to_numpy()

    def vol(w):
        return float(np.sqrt(w @ cov_array @ w))

    low = get_optimized_weights(mean_returns, cov, "Low", max_weight=1.0)
    high = get_optimized_weights(mean_returns, cov, "High", max_weight=1.0)
    assert vol(low) < vol(high)


def test_cap_below_equal_weight_is_ignored(sample_market):
    """A 10% cap across 3 assets is infeasible; fall back to equal weight."""
    mean_returns, cov = sample_market
    weights = get_optimized_weights(mean_returns, cov, "High", max_weight=0.1)
    assert weights.sum() == pytest.approx(1.0)


def test_single_asset_gets_everything():
    mean_returns = pd.Series({"ONLY": 0.1})
    cov = pd.DataFrame([[0.04]], index=["ONLY"], columns=["ONLY"])
    assert get_optimized_weights(mean_returns, cov, "Moderate") == pytest.approx([1.0])


def test_normalize_custom_weights_renormalises_after_a_dropped_ticker():
    weights = normalize_custom_weights({"A": 50.0, "B": 30.0, "C": 20.0}, ["A", "B"])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.625)


def test_normalize_custom_weights_falls_back_to_equal_weight():
    weights = normalize_custom_weights({}, ["A", "B", "C", "D"])
    assert weights == pytest.approx([0.25] * 4)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_sharpe_ratio_uses_the_risk_free_rate():
    """A zero risk-free rate overstates Sharpe whenever short rates aren't zero."""
    returns = pd.DataFrame({"A": np.full(252, 0.001)})
    zero_rate = calculate_portfolio_metrics(returns, [1.0], risk_free_rate=0.0)
    with_rate = calculate_portfolio_metrics(returns, [1.0], risk_free_rate=0.04)
    assert with_rate["sharpe_ratio"] < zero_rate["sharpe_ratio"]


def test_max_drawdown_is_negative_after_a_fall():
    returns = pd.Series([0.10, 0.10, -0.50, 0.05])
    metrics = calculate_portfolio_metrics(returns.to_frame("A"), [1.0])
    assert metrics["max_drawdown"] == pytest.approx(-0.5 * 1.0, abs=0.02)


def test_metrics_reject_mismatched_weights():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, 0.02]})
    with pytest.raises(ValueError):
        calculate_portfolio_metrics(returns, [1.0])


def test_cagr_reflects_volatility_drag_at_low_drift():
    """With a small drift and high volatility, compounding trails the mean.

    Not a universal law — at high drift the exponential in exp(mu - sigma^2/2)
    can push CAGR back above the arithmetic figure. Both are reported in the UI
    for exactly that reason.
    """
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.02, 2520))
    metrics = calculate_portfolio_metrics(returns.to_frame("A"), [1.0])
    assert metrics["cagr"] < metrics["annual_return"]


def test_beta_of_an_asset_against_itself_is_one():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0, 0.01, 300), index=pd.RangeIndex(300))
    assert calculate_beta(series, series) == pytest.approx(1.0)


def test_beta_returns_none_without_enough_overlap():
    a = pd.Series([0.01] * 10, index=range(10))
    b = pd.Series([0.01] * 10, index=range(100, 110))
    assert calculate_beta(a, b) is None


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------

def test_monte_carlo_shape_and_start_value():
    paths = monte_carlo_simulation(1000, 0.08, 0.15, years=2, simulations=50, seed=7)
    assert paths.shape == (50, 504)
    assert (paths[:, 0] == 1000).all()


def test_monte_carlo_is_reproducible_with_a_seed():
    first = monte_carlo_simulation(1000, 0.08, 0.15, years=1, simulations=20, seed=42)
    second = monte_carlo_simulation(1000, 0.08, 0.15, years=1, simulations=20, seed=42)
    np.testing.assert_allclose(first, second)


def test_monte_carlo_median_tracks_expected_growth():
    """With zero volatility the paths are deterministic: value = e^(mu*t)."""
    paths = monte_carlo_simulation(100, 0.10, 0.0, years=5, simulations=10, seed=1)
    assert paths[:, -1] == pytest.approx(100 * np.exp(0.10 * 5), rel=0.01)


def test_monte_carlo_paths_stay_positive():
    paths = monte_carlo_simulation(1000, -0.20, 0.60, years=10, simulations=200, seed=3)
    assert (paths > 0).all()


def test_monte_carlo_handles_zero_horizon():
    assert monte_carlo_simulation(1000, 0.08, 0.15, years=0, simulations=10).size == 0
