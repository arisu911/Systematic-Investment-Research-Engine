"""UI styling, Plotly layout standards, and Streamlit component helpers."""

from typing import Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go


def apply_terminal_theme():
    """Inject institutional dark terminal styling."""
    st.markdown(
        """
        <style>
        /* Quantitative Research Terminal Theme */
        .main {
            background-color: #0B0E14;
            color: #E6E8EB;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        div[data-testid="stMetric"] {
            background-color: #151A24;
            border: 1px solid #232B3B;
            padding: 12px 16px;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
            color: #8C9BAE;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #F0F3F6;
            font-weight: 700;
            font-family: "SF Mono", Monaco, Inconsolata, "Fira Code", monospace;
        }
        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .badge-live {
            background-color: rgba(0, 212, 178, 0.15);
            color: #00D4B2;
            border: 1px solid #00D4B2;
        }
        .badge-demo {
            background-color: rgba(255, 171, 0, 0.15);
            color: #FFAB00;
            border: 1px solid #FFAB00;
        }
        .badge-warning {
            background-color: rgba(255, 86, 48, 0.15);
            color: #FF5630;
            border: 1px solid #FF5630;
        }
        .terminal-header {
            border-bottom: 1px solid #232B3B;
            padding-bottom: 12px;
            margin-bottom: 18px;
        }
        .section-header {
            font-size: 1.05rem;
            font-weight: 600;
            color: #00D4B2;
            border-left: 3px solid #00D4B2;
            padding-left: 8px;
            margin-top: 20px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_terminal_header(title: str, subtitle: str, is_demo: bool = False):
    """Render standardized top terminal bar with badges."""
    badge_html = (
        '<span class="status-badge badge-demo">DEMO MODE (SYNTHETIC/CACHED)</span>'
        if is_demo
        else '<span class="status-badge badge-live">LIVE DATA READY</span>'
    )
    st.markdown(
        f"""
        <div class="terminal-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; font-size: 1.6rem; color: #FFFFFF; font-weight: 700;">{title}</h2>
                    <div style="color: #8C9BAE; font-size: 0.88rem; margin-top: 4px;">{subtitle}</div>
                </div>
                <div>{badge_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_terminal_plotly_layout(height: int = 420, title: Optional[str] = None) -> Dict[str, Any]:
    """Standardized Plotly theme dictionary for dark high-density charting."""
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "#0B0E14",
        "plot_bgcolor": "#121620",
        "font": {"family": "sans-serif", "color": "#C5CEE0", "size": 11},
        "margin": {"l": 50, "r": 30, "t": 40 if title else 25, "b": 35},
        "height": height,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 10},
        },
        "xaxis": {
            "gridcolor": "#1C2433",
            "zerolinecolor": "#232B3B",
            "showline": True,
            "linecolor": "#232B3B",
        },
        "yaxis": {
            "gridcolor": "#1C2433",
            "zerolinecolor": "#232B3B",
            "showline": True,
            "linecolor": "#232B3B",
        },
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"size": 13, "color": "#FFFFFF"},
            "x": 0.01,
            "y": 0.98,
        }
    return layout


def render_disclaimer():
    """Mandatory financial research integrity and non-solicitation disclaimer."""
    st.markdown(
        """
        <div style="margin-top: 40px; padding: 12px 16px; border-top: 1px solid #232B3B; font-size: 0.75rem; color: #6B778C; text-align: center;">
            <strong>RESEARCH METHODOLOGY DISCLAIMER:</strong> This terminal is an empirical quantitative research framework intended strictly for academic hypothesis testing and backtest simulation. Historical returns and simulated backtests do not guarantee future performance. Strategy calculations reflect modeled transactions and estimated slippage assumptions. Not financial or investment advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
