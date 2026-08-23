"""Global page styling.

Deliberately small. The old stylesheet hardcoded ``rgba(255,255,255,0.8)`` and
``#666``, which meant captions were invisible on one theme or the other. Almost
everything here now goes through native Streamlit components, which follow the
viewer's theme on their own; what's left uses Streamlit's theme variables with a
safe fallback.
"""

import streamlit as st

_CSS = """
<style>
  .block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    padding-left: 2rem;
    padding-right: 2rem;
  }

  /* Full-width buttons and tables */
  div.stButton > button { width: 100%; }
  div[data-testid="stDataFrame"] { width: 100%; }

  /* Caption under a table or chart — theme-aware, never a hardcoded grey */
  .figure-note {
    font-size: 0.85em;
    line-height: 1.5;
    color: var(--text-color, inherit);
    opacity: 0.7;
    margin: -0.5rem 0 1.25rem;
  }

  /* Keep numeric columns aligned */
  div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

  /* Footer */
  .app-footer {
    margin-top: 4rem;
    padding-top: 1.25rem;
    border-top: 1px solid rgba(128, 128, 128, 0.25);
    text-align: center;
    font-size: 0.9rem;
    color: var(--text-color, inherit);
    opacity: 0.75;
  }
  .app-footer a {
    color: inherit;
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-thickness: 1px;
  }
  .app-footer a:hover { opacity: 1; }
  .app-footer a:focus-visible {
    outline: 2px solid currentColor;
    outline-offset: 3px;
    border-radius: 2px;
  }
</style>
"""

PORTFOLIO_URL = "https://marco-portfolio-azure.vercel.app/"


def apply_page_style() -> None:
    """Inject the stylesheet. Call once, immediately after ``set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)


def figure_note(text: str) -> None:
    """Render an explanatory note under a chart or table."""
    st.markdown(f'<div class="figure-note">{text}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Page footer. Rendered on every path, including the early returns."""
    st.markdown(
        f'<div class="app-footer">Made with ❤️ by '
        f'<a href="{PORTFOLIO_URL}" target="_blank" rel="noopener noreferrer">'
        f'Marco Quantrill</a></div>',
        unsafe_allow_html=True,
    )
