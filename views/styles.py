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
</style>
"""


def apply_page_style() -> None:
    """Inject the stylesheet. Call once, immediately after ``set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)


def figure_note(text: str) -> None:
    """Render an explanatory note under a chart or table."""
    st.markdown(f'<div class="figure-note">{text}</div>', unsafe_allow_html=True)
