"""Monte Carlo projection of portfolio value."""

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st

from core.formatting import format_percent
from core.optimize import monte_carlo_simulation
from data.analysis import PortfolioAnalysis
from views.styles import figure_note

MAX_PLOTTED_PATHS = 120


def render_monte_carlo(analysis: PortfolioAnalysis) -> None:
    """Project the portfolio forward under geometric Brownian motion."""
    st.markdown("### Monte Carlo projection")
    st.markdown(
        "Projects portfolio value forward using the historical return and "
        "volatility you see above, with random shocks drawn from a normal "
        "distribution (geometric Brownian motion)."
    )

    metrics = analysis.metrics
    if not metrics:
        st.warning("Portfolio metrics are unavailable, so there's nothing to project.")
        return

    col1, col2 = st.columns(2)
    with col1:
        start_value = st.number_input(
            "Starting value ($)",
            min_value=1_000, max_value=10_000_000, value=10_000, step=1_000,
        )
    with col2:
        years = st.slider("Horizon (years)", min_value=1, max_value=30, value=10)

    col3, col4 = st.columns(2)
    with col3:
        simulations = st.slider(
            "Simulations", min_value=100, max_value=10_000, value=1_000, step=100
        )
    with col4:
        seed = st.number_input(
            "Random seed", min_value=0, max_value=999_999, value=42,
            help="Fixing the seed makes the projection reproducible. Change it to "
                 "draw a different set of paths.",
        )

    st.caption(
        f"Drift: {format_percent(metrics['annual_return'])} annual return · "
        f"Volatility: {format_percent(metrics['annual_vol'])}"
    )

    if not st.button("Run projection", use_container_width=True):
        return

    with st.spinner("Running simulation…"):
        results = monte_carlo_simulation(
            start_value=float(start_value),
            mean_return=metrics["annual_return"],
            volatility=metrics["annual_vol"],
            years=int(years),
            simulations=int(simulations),
            seed=int(seed),
        )

    if results.size == 0:
        st.warning("The simulation produced no paths.")
        return

    percentiles = np.percentile(results, [5, 50, 95], axis=0)
    x = np.arange(results.shape[1]) / 252

    fig = go.Figure()
    for path in results[:MAX_PLOTTED_PATHS]:
        fig.add_trace(go.Scatter(
            x=x, y=path, mode="lines",
            line=dict(width=1, color="rgba(120,140,220,0.18)"),
            showlegend=False, hoverinfo="skip",
        ))
    for values, name, color, dash in (
        (percentiles[2], "95th percentile", "#2EA057", "dash"),
        (percentiles[1], "Median", "#4666B0", None),
        (percentiles[0], "5th percentile", "#D64038", "dash"),
    ):
        fig.add_trace(go.Scatter(
            x=x, y=values, mode="lines", name=name,
            line=dict(width=2.5, color=color, dash=dash),
            hovertemplate=f"{name}: $%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{simulations:,} simulated paths over {years} years",
        xaxis_title="Years from now",
        yaxis_title="Portfolio value ($)",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    figure_note(f"Showing {min(MAX_PLOTTED_PATHS, simulations):,} individual paths for legibility; "
                f"the percentile bands are computed from all {simulations:,}.")

    final_values = results[:, -1]

    def summarise(value: float) -> str:
        return f"${value:,.0f}"

    def delta(value: float) -> str:
        return f"{(value - start_value) / start_value * 100:+.1f}%"

    col1, col2, col3, col4 = st.columns(4)
    median = float(np.median(final_values))
    worst = float(np.percentile(final_values, 5))
    best = float(np.percentile(final_values, 95))
    loss_probability = float((final_values < start_value).mean())

    col1.metric("Median outcome", summarise(median), delta(median),
                help="Half of the simulations end above this value.")
    col2.metric("5th percentile", summarise(worst), delta(worst),
                help="Only 5% of simulations end below this value.")
    col3.metric("95th percentile", summarise(best), delta(best),
                help="Only 5% of simulations end above this value.")
    col4.metric("Chance of a loss", format_percent(loss_probability, decimals=1),
                help="Share of simulations ending below the starting value.")

    st.markdown("##### Distribution of final values")
    fig_hist = px.histogram(x=final_values, nbins=60,
                            color_discrete_sequence=["#4666B0"])
    fig_hist.add_vline(x=start_value, line_dash="dash",
                       line_color="rgba(128,128,128,0.9)",
                       annotation_text="Starting value")
    fig_hist.update_layout(
        height=380, showlegend=False,
        xaxis_title="Final portfolio value ($)",
        yaxis_title="Simulations",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "This model assumes normally-distributed returns and constant volatility. "
        "Real markets have fatter tails, so the worst cases here are optimistic."
    )
