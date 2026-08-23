# 📊 MQ Portfolio Dashboard

An interactive Streamlit dashboard for portfolio optimisation, benchmarking and
fundamental analysis across the S&P 500.

> **Educational project — not investment advice.** Every figure is derived from
> historical Yahoo Finance data, which may be delayed by 15–20 minutes.
> Optimised allocations are fitted to the past, and past performance does not
> predict future returns.

## Features

- **Mean-variance optimisation** — minimum volatility, maximum Sharpe, or
  maximum return under a configurable per-position cap
- **Custom weights** — set your own allocation and compare it against the
  optimised ones
- **Benchmark comparison** — portfolio vs. S&P 500, plus a calendar-year
  breakdown per holding
- **Monte Carlo projection** — geometric Brownian motion with a fixed seed, so
  runs are reproducible
- **Fundamental analysis** — income statement, balance sheet and cash flow with
  horizontal (year-over-year) and vertical (common-size) analysis
- **14 financial ratios** with interpretation against benchmark ranges
- **Market overview** — indices, commodities, FX, crypto and sector performance

## Quick start

```bash
git clone https://github.com/MQuantrillC/MQ-Portfolio-Dashboard.git
cd MQ-Portfolio-Dashboard
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 is the tested version (see `runtime.txt`).

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

76 tests covering the portfolio maths, ratio analysis and timeframe handling.
They're fast because everything under `core/` is a pure function — no network,
no Streamlit, no mocking.

## Project structure

```
app.py                  Entry point — page config and routing only
core/                   Pure functions. No Streamlit, no network. All tested.
  config.py             PortfolioConfig and shared constants
  formatting.py         Number → string, at the render boundary only
  optimize.py           Weights, metrics, portfolio index, Monte Carlo
  ratios.py             Statement parsing, ratio maths, interpretation bands
  timeframes.py         Timeframe → date range, market sentiment labels
data/                   Everything that touches the network. All cached.
  market.py             yfinance access — prices, profiles, statements
  universe.py           S&P 500 constituents, with a visible fallback
  analysis.py           Composes data + core into one PortfolioAnalysis
views/                  Rendering. Reads data, never fetches it directly.
  setup.py              The setup form → PortfolioConfig
  portfolio.py          Metrics, holdings, allocation, benchmark, charts
  financials.py         Statements, common-size analysis, ratios
  market.py             Market overview
  montecarlo.py         Monte Carlo projection
  styles.py             The (small) stylesheet
tests/                  pytest suite over core/
```

The rule that keeps this honest: **`core/` takes numbers and returns numbers.**
If a function needs Streamlit or the network, it doesn't belong there.

## Modelling assumptions

These are limitations of the approach, not bugs:

- **Mean-variance optimisation is sensitive to the estimation window.** Expected
  returns estimated from historical data are noisy, and the optimiser amplifies
  that noise. The position cap exists partly to blunt it.
- **The Monte Carlo assumes normally-distributed returns and constant
  volatility.** Real markets have fatter tails, so the downside cases are
  optimistic.
- **Survivorship bias is baked in.** Today's S&P 500 members are, by
  construction, companies that did well enough to still be in the index.
- **The Sharpe ratio uses a risk-free rate you set in the sidebar** (default 4%),
  not a live short rate.

## Data source

Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance), and the
constituent list from Wikipedia. Both are unofficial and can rate-limit or
change format. When the constituent scrape fails, the app says so and falls back
to a list of 50 major constituents rather than pretending nothing happened.
