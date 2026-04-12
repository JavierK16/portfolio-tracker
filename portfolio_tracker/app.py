"""
app.py — Streamlit dashboard entry point.
Run with: streamlit run app.py
"""

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone, timedelta, date
from typing import Dict, List, Optional

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.logging_setup import setup_logging
setup_logging()

from config import (
    INVESTOR_PROFILE, PORTFOLIO, SECTOR_CONFIG,
    TRANCHE_2_DAYS, TRANCHE_3_DAYS,
)
from src.database import (
    init_db, get_config_value, set_config_value,
    get_signal_history, get_sector_score_history, get_price_history,
)
from src.price_engine import get_price_engine, PositionData
from src.geo_scorer import get_geo_scorer, GeoContext
from src.signal_engine import get_signal_engine, SignalResult
from src.alert_manager import get_alert_manager
from src.prediction_engine import get_prediction_engine, PricePrediction, SectorPrediction, PortfolioPrediction
from src.prediction_accuracy import get_accuracy_tracker

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolitical Portfolio Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS / COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card { background:#1e1e2e; border-radius:8px; padding:12px 16px; margin:4px 0; }
  .signal-BUY    { color:#00ff88; font-weight:700; }
  .signal-ADD    { color:#88ff44; font-weight:700; }
  .signal-HOLD   { color:#aaaaaa; font-weight:700; }
  .signal-REDUCE { color:#ffaa00; font-weight:700; }
  .signal-SELL   { color:#ff4444; font-weight:700; }
  .geo-RED   { color:#ff4444; }
  .geo-AMBER { color:#ffaa00; }
  .geo-GREEN { color:#00ff88; }
  .severity-CRITICAL { background:#5c0000; color:#ff8888; padding:2px 6px; border-radius:4px; font-size:0.75rem; }
  .severity-HIGH     { background:#4a2800; color:#ffaa44; padding:2px 6px; border-radius:4px; font-size:0.75rem; }
  .severity-MEDIUM   { background:#2a2200; color:#ffe066; padding:2px 6px; border-radius:4px; font-size:0.75rem; }
  .severity-REMINDER { background:#002a40; color:#66ccff; padding:2px 6px; border-radius:4px; font-size:0.75rem; }
  .stale-badge { background:#444; color:#ccc; padding:2px 6px; border-radius:4px; font-size:0.7rem; margin-left:6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# INITIALISATION (run once per server process)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def _init_system():
    """Start all background engines. Cached so they run only once."""
    init_db()

    # Store system start date on first run
    if not get_config_value("system_start_date"):
        set_config_value("system_start_date", date.today().isoformat())

    fx = __import__("src.fx_engine", fromlist=["get_fx_engine"]).get_fx_engine()
    fx.start_background_refresh()

    pe = get_price_engine()
    pe.backfill_history(years=2)  # One-time 2-year history download
    pe.start_background_refresh()

    gs = get_geo_scorer()
    gs.start_background_refresh()

    se = get_signal_engine()
    se.start_background_refresh()

    am = get_alert_manager()
    am.start_background_checks()

    pred = get_prediction_engine()
    pred.start_background_refresh()

    acc = get_accuracy_tracker()
    acc.start_background_checks()

    return pe, gs, se, am, pred, acc


price_engine, geo_scorer, signal_engine, alert_manager, prediction_engine, accuracy_tracker = _init_system()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

SIGNAL_COLOURS = {
    "BUY":    "#00ff88",
    "ADD":    "#88ff44",
    "HOLD":   "#aaaaaa",
    "REDUCE": "#ffaa00",
    "SELL":   "#ff4444",
}

GEO_STATUS_COLOUR = {
    # HORMUZ_STATUS
    "CLOSED":  "geo-RED", "PARTIAL": "geo-AMBER", "OPEN": "geo-GREEN",
    # IRAN_CONFLICT
    "ACTIVE":  "geo-RED", "CEASEFIRE": "geo-AMBER", "RESOLVED": "geo-GREEN",
    # UKRAINE_WAR
    "ESCALATING": "geo-RED", "STALEMATE": "geo-AMBER",
    "DE-ESCALATING": "geo-GREEN",
    # US_CHINA_RELATIONS
    "HOSTILE": "geo-RED", "TENSE": "geo-AMBER",
    "NEUTRAL": "geo-GREEN", "COOPERATIVE": "geo-GREEN",
    # NATO_SPENDING
    "DECLINING": "geo-RED", "STABLE": "geo-AMBER",
    "INCREASING": "geo-GREEN", "ACCELERATING": "geo-GREEN",
}


def _fmt_eur(v: Optional[float], decimals: int = 0) -> str:
    if v is None:
        return "N/A"
    return f"€{v:,.{decimals}f}"


def _fmt_pct(v: Optional[float], show_plus: bool = True) -> str:
    if v is None:
        return "N/A"
    prefix = "+" if (show_plus and v > 0) else ""
    return f"{prefix}{v:.2f}%"


def _delta_colour(v: Optional[float], invert: bool = False) -> str:
    if v is None:
        return "normal"
    if invert:
        return "inverse"
    return "normal"


def _signal_badge(signal: str) -> str:
    colour = SIGNAL_COLOURS.get(signal, "#888")
    return f'<span style="color:{colour};font-weight:700">{signal}</span>'


def _severity_badge(severity: str) -> str:
    return f'<span class="severity-{severity}">{severity}</span>'


def _hours_ago(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    h = delta.total_seconds() / 3600
    if h < 1:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if h < 24:
        return f"{int(h)}h ago"
    return f"{int(h/24)}d ago"


def _get_dominant_signal(sector: str, signals: Dict[str, SignalResult]) -> str:
    sig_vals = {"BUY": 5, "ADD": 4, "HOLD": 3, "REDUCE": 2, "SELL": 1}
    from config import PORTFOLIO
    tickers = [p["ticker"] for p in PORTFOLIO if p["sector"] == sector]
    best = "HOLD"
    best_val = 3
    for t in tickers:
        sig = signals.get(t)
        if sig and sig_vals.get(sig.signal, 0) > best_val:
            best = sig.signal
            best_val = sig_vals[sig.signal]
    return best


# ─────────────────────────────────────────────────────────────
# HEADER BAR
# ─────────────────────────────────────────────────────────────

def render_header(geo_context: GeoContext) -> None:
    positions  = price_engine.get_all_positions()
    total_val  = price_engine.get_portfolio_value()
    day_pnl    = price_engine.get_portfolio_day_pnl()
    total_pnl  = price_engine.get_portfolio_total_pnl()
    total_cost = sum((p.shares_units or 0) * (p.entry_price_eur or 0) for p in positions)
    alerts     = alert_manager.get_unacknowledged()
    now        = datetime.now(timezone.utc)
    last_ref   = price_engine.last_refresh()

    n_critical = sum(1 for a in alerts if a.severity == "CRITICAL")
    n_high     = sum(1 for a in alerts if a.severity == "HIGH")
    n_medium   = sum(1 for a in alerts if a.severity == "MEDIUM")

    col_refresh, col_val, col_dpnl, col_tpnl, col_alerts, col_btn = st.columns([2, 2, 2, 2, 2, 1])

    market_open = price_engine.is_market_hours()
    mkt_label   = "🟢 Markets open" if market_open else "🔴 Markets closed"
    mkt_colour  = "#00ff88"         if market_open else "#ff6644"

    with col_refresh:
        st.markdown(
            f"**Last refresh:** {_hours_ago(last_ref)}<br>"
            f"<small>Signals: {_hours_ago(signal_engine.last_refresh_time())}</small><br>"
            f'<span style="color:{mkt_colour};font-size:0.8rem">{mkt_label}</span>',
            unsafe_allow_html=True,
        )

    with col_val:
        day_pct = (day_pnl / total_cost * 100) if total_cost else 0
        day_label = _fmt_pct(day_pct) + " today" if market_open else "Mkt closed — 0% today"
        st.metric("Portfolio Value", _fmt_eur(total_val), day_label)

    with col_dpnl:
        dpnl_label = _fmt_eur(day_pnl, 0) if market_open else "€0 (closed)"
        st.metric("Day P&L", dpnl_label,
                  delta=_fmt_pct(day_pct) if market_open else None,
                  help="Profit/loss today based on live price moves. Zero when markets are closed.")

    with col_tpnl:
        total_pct = (total_pnl / total_cost * 100) if total_cost else 0
        st.metric(
            "Total P&L since entry",
            _fmt_eur(total_pnl, 0),
            f"{_fmt_pct(total_pct)}",
            help=(
                "Total profit/loss in EUR across all positions since your entry prices "
                "were recorded on first run. "
                f"Cost basis: {_fmt_eur(total_cost)} | "
                "P&L = current value − cost basis."
            ),
        )

    with col_alerts:
        badge_parts = []
        if n_critical:
            badge_parts.append(f'<span class="severity-CRITICAL">{n_critical} CRITICAL</span>')
        if n_high:
            badge_parts.append(f'<span class="severity-HIGH">{n_high} HIGH</span>')
        if n_medium:
            badge_parts.append(f'<span class="severity-MEDIUM">{n_medium} MEDIUM</span>')
        if not badge_parts:
            badge_parts = ['<span style="color:#00ff88">No alerts</span>']
        st.markdown("**Alerts**<br>" + " ".join(badge_parts), unsafe_allow_html=True)

    with col_btn:
        if st.button("🔄 Refresh Now"):
            price_engine.refresh_all()
            geo_scorer.refresh()
            signal_engine.refresh_all()
            st.rerun()


# ─────────────────────────────────────────────────────────────
# SECTION 1 — GEOPOLITICAL SITUATION BOARD
# ─────────────────────────────────────────────────────────────

def render_geo_board(geo_context: GeoContext) -> None:
    st.subheader("Geopolitical Situation Board")

    var_labels = {
        "HORMUZ_STATUS":      "Hormuz Strait",
        "IRAN_CONFLICT":      "Iran Conflict",
        "UKRAINE_WAR":        "Ukraine War",
        "US_CHINA_RELATIONS": "US-China Relations",
        "NATO_SPENDING":      "NATO Spending",
    }

    cols = st.columns(5)
    for i, (var, label) in enumerate(var_labels.items()):
        info = geo_context.variables.get(var, {})
        value = info.get("value", "UNKNOWN")
        colour_cls = GEO_STATUS_COLOUR.get(value, "geo-AMBER")
        changed = info.get("changed_hours_ago")
        changed_str = f"{changed:.0f}h ago" if changed is not None else "unknown"
        headline = (info.get("headline") or "No recent trigger")[:80]

        with cols[i]:
            st.markdown(
                f"**{label}**<br>"
                f'<span class="{colour_cls}" style="font-size:1.1rem">{value}</span><br>'
                f"<small>Changed: {changed_str}</small><br>"
                f"<small style='color:#888'>{headline}</small>",
                unsafe_allow_html=True,
            )

    # News feed
    st.markdown("---")
    st.markdown("**Recent Geopolitical News (last 48h)**")

    if not geo_context.recent_news:
        st.caption("No matching news items yet. Feed refreshes every 15 minutes.")
        return

    for item in geo_context.recent_news[:20]:
        sector_badges = " ".join(
            f'<span style="background:#333;padding:1px 5px;border-radius:3px;font-size:0.7rem">{s}</span>'
            for s in item.sectors_matched
        )
        delta_str = f"+{item.sentiment_delta:.1f}" if item.sentiment_delta >= 0 else f"{item.sentiment_delta:.1f}"
        delta_col = "#00ff88" if item.sentiment_delta > 0 else "#ff4444" if item.sentiment_delta < 0 else "#888"
        pub_str = _hours_ago(item.published) if item.published else "unknown"

        st.markdown(
            f"{sector_badges} "
            f'<span style="color:{delta_col};font-size:0.75rem">[{delta_str}]</span> '
            f"<small>{pub_str}</small> — {item.title[:120]}",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# SECTION 2 — SECTOR OVERVIEW
# ─────────────────────────────────────────────────────────────

def render_sector_overview(
    geo_context: GeoContext,
    signals: Dict[str, SignalResult],
) -> None:
    st.subheader("Sector Overview")
    positions = price_engine.get_all_positions()

    cols = st.columns(5)
    for i, (sector, cfg) in enumerate(SECTOR_CONFIG.items()):
        sector_positions = [p for p in positions if p.sector == sector]
        alloc_eur   = sum(p.current_value_eur or 0 for p in sector_positions)
        total_cost  = sum((p.shares_units or 0) * (p.entry_price_eur or 0) for p in sector_positions)
        pnl_eur     = alloc_eur - total_cost
        total_val   = price_engine.get_portfolio_value()
        alloc_pct   = (alloc_eur / total_val * 100) if total_val else 0

        geo_score   = max(0.0, min(10.0, geo_context.sector_scores.get(sector, cfg["base_score"])))
        score_delta = geo_context.sector_score_changes.get(sector, 0)
        arrow = "▲" if score_delta > 0 else ("▼" if score_delta < 0 else "→")
        arrow_col = "#00ff88" if score_delta > 0 else ("#ff4444" if score_delta < 0 else "#888")
        dom_signal  = _get_dominant_signal(sector, signals)

        with cols[i]:
            st.markdown(
                f"**{sector}**<br>"
                f"{_fmt_eur(alloc_eur)} ({alloc_pct:.1f}%)<br>"
                f"Geo Score: {geo_score:.1f}/10 "
                f'<span style="color:{arrow_col}">{arrow}{abs(score_delta):.2f}</span><br>'
                f"P&L: {_fmt_eur(pnl_eur)} ({_fmt_pct(pnl_eur/total_cost*100 if total_cost else 0)})<br>"
                f"Signal: {_signal_badge(dom_signal)}",
                unsafe_allow_html=True,
            )

            # ── Sector sparkline ──────────────────────────────
            # Build per-ticker daily price series, align to common
            # date index, forward-fill gaps, then sum to get total
            # sector value per day. Avoids spikes from misaligned timestamps.
            ticker_series = {}
            for p in sector_positions:
                hist = get_price_history(p.ticker, days=30)
                if not hist:
                    continue
                shares = p.shares_units or 0
                s = pd.Series(
                    {h.timestamp: (h.price_eur or 0) * shares for h in hist},
                    name=p.ticker,
                )
                # Resample to 1-hour buckets to smooth intra-day noise
                s.index = pd.to_datetime(s.index, utc=True)
                s = s.resample("1h").last().ffill()
                ticker_series[p.ticker] = s

            if ticker_series:
                combined = pd.DataFrame(ticker_series)
                # Forward-fill so every ticker has a value at every timestamp
                combined = combined.ffill().dropna(how="all")
                sector_total = combined.sum(axis=1)

                # Resample to daily for a clean sparkline
                daily = sector_total.resample("1D").last().dropna()

                # Need at least 2 points to draw a line
                if len(daily) >= 2:
                    line_col = SIGNAL_COLOURS.get(dom_signal, "#888")
                    fig = go.Figure(go.Scatter(
                        x=daily.index, y=daily.values,
                        mode="lines",
                        line=dict(color=line_col, width=2),
                        fill="tozeroy",
                        fillcolor=line_col.replace(")", ",0.15)").replace("rgb", "rgba")
                                  if line_col.startswith("rgb") else line_col + "26",
                    ))
                    fig.update_layout(
                        height=70,
                        margin=dict(l=0, r=0, t=4, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                   key=f"spark_{sector}")
                else:
                    st.caption("📊 Building history…")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — POSITION TABLE
# ─────────────────────────────────────────────────────────────

def render_position_table(signals: Dict[str, SignalResult]) -> Optional[str]:
    st.subheader("Position Table")

    positions  = price_engine.get_all_positions()
    total_val  = price_engine.get_portfolio_value()

    rows = []
    for pos in positions:
        sig     = signals.get(pos.ticker)
        signal  = sig.signal if sig else "HOLD"
        score   = sig.composite_score if sig else None
        flags   = sig.flags if sig else []

        drift   = pos.drift_from_target or 0
        drift_col = "#ff4444" if abs(drift) > 3 else ("#00ff88" if abs(drift) <= 1 else "#ffaa00")

        # Helper: format only when value is not None (avoids 0.0 being treated as falsy)
        def _v(val, fmt):
            return fmt.format(val) if val is not None else "N/A"

        rows.append({
            "Ticker":    pos.ticker,
            "Name":      pos.name[:30],
            "Sector":    pos.sector,
            "Type":      pos.instrument_type,
            "Ccy":       pos.currency,
            "Shares":    _v(pos.shares_units,        "{:.2f}"),
            "Entry €":   _v(pos.entry_price_eur,     "{:.2f}"),
            "Price €":   _v(pos.current_price_eur,   "{:.2f}"),
            "Value €":   _v(pos.current_value_eur,   "{:,.0f}"),
            "P&L €":     _v(pos.pnl_eur,             "{:+,.0f}"),
            "P&L %":     _v(pos.pnl_pct,             "{:+.2f}%"),
            "Day %":     _v(pos.day_change_pct,      "{:+.2f}%"),
            "Week %":    _v(pos.week_change_pct,     "{:+.2f}%"),
            "Weight %":  _v(pos.weight_current_pct,  "{:.1f}%"),
            "Target %":  f"{pos.target_pct:.1f}%",
            "Drift":     f"{drift:+.1f}%",
            "Signal":    signal,
            "Score":     f"{score:.0f}" if score is not None else "—",
            "Tranche":   str(pos.tranche),
            "Status":    pos.data_status,
            "_flags":    flags,
        })

    df = pd.DataFrame(rows)

    # Colour styling is limited in st.dataframe — use HTML table for rich formatting
    # But for interactivity we use st.dataframe with a selection mechanism

    # Signal and drift colour mapping
    def highlight_row(row):
        sig = row.get("Signal", "HOLD")
        colours = {
            "BUY":    "background-color: #002a15",
            "ADD":    "background-color: #001a08",
            "HOLD":   "",
            "REDUCE": "background-color: #2a1500",
            "SELL":   "background-color: #2a0000",
        }
        return [colours.get(sig, "")] * len(row)

    display_cols = [c for c in df.columns if not c.startswith("_")]
    display_df   = df[display_cols].copy()

    st.dataframe(
        display_df.style.apply(highlight_row, axis=1),
        use_container_width=True,
        height=500,
    )

    # Row selection for detail panel
    selected = st.selectbox(
        "Select position to view signal rationale:",
        options=[""] + [p.ticker for p in positions],
        format_func=lambda x: x if x else "— select ticker —",
        key="selected_ticker",
    )

    # Show flags summary
    all_flags = []
    for _, row in df.iterrows():
        for flag in row["_flags"]:
            all_flags.append(f"**{row['Ticker']}**: {flag}")
    if all_flags:
        with st.expander(f"⚠ Active override flags ({len(all_flags)})", expanded=False):
            for f in all_flags:
                st.warning(f)

    return selected if selected else None


# ─────────────────────────────────────────────────────────────
# SECTION 4 — SIGNAL RATIONALE PANEL
# ─────────────────────────────────────────────────────────────

def render_signal_rationale(ticker: str, geo_context: GeoContext) -> None:
    sig = signal_engine.get_signal(ticker)
    pos = price_engine.get_position(ticker)
    if not sig or not pos:
        st.info(f"No signal data for {ticker} yet.")
        return

    st.subheader(f"Signal Rationale — {ticker} ({sig.signal})")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Factor Scores**")
        from config import SIGNAL_WEIGHTS
        factor_df = pd.DataFrame([
            {"Factor": "Sector Geo Score",  "Score": sig.factors.geo_score,       "Weight": "35%"},
            {"Factor": "Price Momentum",    "Score": sig.factors.momentum_score,  "Weight": "20%"},
            {"Factor": "Position Drift",    "Score": sig.factors.drift_score,     "Weight": "15%"},
            {"Factor": "Drawdown Risk",     "Score": sig.factors.drawdown_score,  "Weight": "15%"},
            {"Factor": "Volatility Regime", "Score": sig.factors.volatility_score,"Weight": "15%"},
            {"Factor": "COMPOSITE",         "Score": sig.composite_score,         "Weight": "—"},
        ])
        st.dataframe(factor_df, use_container_width=True, hide_index=True)

        if sig.flags:
            st.markdown("**Override Flags**")
            for flag in sig.flags:
                st.warning(flag)

    with col2:
        st.markdown("**Active Geopolitical Variables**")
        for var, info in geo_context.variables.items():
            value = info.get("value", "—")
            col_cls = GEO_STATUS_COLOUR.get(value, "geo-AMBER")
            st.markdown(
                f"**{var}:** "
                f'<span class="{col_cls}">{value}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("**Geo Triggers (last 24h)**")
        sector = pos.sector
        triggered = [
            item for item in geo_context.recent_news
            if sector in item.sectors_matched
        ][:5]
        if triggered:
            for item in triggered:
                st.caption(f"• {item.title[:100]}")
        else:
            st.caption("No specific triggers in last 24h")

    st.markdown("**What Would Change the Signal**")
    st.text(sig.what_would_change)

    # Signal history
    hist = get_signal_history(ticker, limit=20)
    if hist:
        st.markdown("**Signal History**")
        hist_df = pd.DataFrame([{
            "Time":      h.timestamp.strftime("%Y-%m-%d %H:%M") if h.timestamp else "—",
            "Signal":    h.signal,
            "Score":     f"{h.composite_score:.0f}" if h.composite_score else "—",
            "Geo":       f"{h.geo_score:.0f}" if h.geo_score else "—",
            "Momentum":  f"{h.momentum_score:.0f}" if h.momentum_score else "—",
        } for h in hist])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# SECTION 5 — ALERTS FEED
# ─────────────────────────────────────────────────────────────

def render_alerts_feed() -> None:
    st.subheader("Alerts Feed")

    col_sev, col_sec, _ = st.columns([2, 2, 6])
    with col_sev:
        sev_filter = st.selectbox(
            "Filter severity", ["ALL", "CRITICAL", "HIGH", "MEDIUM", "REMINDER"],
            key="alert_sev_filter"
        )
    with col_sec:
        sec_filter = st.selectbox(
            "Filter sector", ["ALL"] + list(SECTOR_CONFIG.keys()),
            key="alert_sec_filter"
        )

    alerts = alert_manager.get_unacknowledged(limit=200)

    if sev_filter != "ALL":
        alerts = [a for a in alerts if a.severity == sev_filter]
    if sec_filter != "ALL":
        alerts = [a for a in alerts if a.sector == sec_filter]

    if not alerts:
        st.success("No unacknowledged alerts.")
        return

    for alert in alerts[:50]:
        col_badge, col_msg, col_ack = st.columns([1.5, 8, 1])
        ts_str = alert.timestamp.strftime("%m/%d %H:%M") if alert.timestamp else "—"
        with col_badge:
            st.markdown(
                f'{_severity_badge(alert.severity)} <small>{ts_str}</small>',
                unsafe_allow_html=True
            )
        with col_msg:
            ticker_str = f"[{alert.ticker}] " if alert.ticker else ""
            st.markdown(f"{ticker_str}{alert.message}")
        with col_ack:
            if st.button("✓", key=f"ack_{alert.id}"):
                alert_manager.acknowledge(alert.id)
                st.rerun()


# ─────────────────────────────────────────────────────────────
# SECTION 6 — PORTFOLIO CHARTS
# ─────────────────────────────────────────────────────────────

def render_charts(geo_context: GeoContext) -> None:
    st.subheader("Portfolio Charts")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Value", "Sector Allocation",
        "Correlation Matrix", "P&L Waterfall", "Geo Score History"
    ])

    positions = price_engine.get_all_positions()
    total_val = price_engine.get_portfolio_value()

    # ── Tab 1: Portfolio value over time ──────────────────────
    with tab1:
        hist_df = price_engine.get_historical_portfolio_values(days=30)
        if not hist_df.empty:
            fig = go.Figure(go.Scatter(
                x=hist_df["timestamp"], y=hist_df["total_value_eur"],
                mode="lines", fill="tozeroy",
                line=dict(color="#00aaff", width=2),
                name="Portfolio Value",
            ))
            # Reference line at initial capital
            fig.add_hline(
                y=INVESTOR_PROFILE["total_capital"],
                line_dash="dash", line_color="#ff8844",
                annotation_text="Initial Capital €100k",
            )
            fig.update_layout(
                title="Portfolio Value (EUR)", height=400,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#ccc"),
                yaxis_title="EUR",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient history data — check back after first price refresh.")

    # ── Tab 2: Sector allocation ─────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Actual Allocation**")
            sector_vals = {}
            for p in positions:
                sector_vals[p.sector] = sector_vals.get(p.sector, 0) + (p.current_value_eur or 0)
            if sector_vals:
                fig = go.Figure(go.Pie(
                    labels=list(sector_vals.keys()),
                    values=list(sector_vals.values()),
                    textinfo="label+percent",
                    hole=0.3,
                ))
                fig.update_layout(height=350, paper_bgcolor="#0e1117", font=dict(color="#ccc"))
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("**Target Allocation**")
            sector_targets = {s: c["weight_in_portfolio"] * 100
                              for s, c in SECTOR_CONFIG.items()}
            fig = go.Figure(go.Pie(
                labels=list(sector_targets.keys()),
                values=list(sector_targets.values()),
                textinfo="label+percent",
                hole=0.3,
            ))
            fig.update_layout(height=350, paper_bgcolor="#0e1117", font=dict(color="#ccc"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Correlation matrix ─────────────────────────────
    with tab3:
        try:
            price_dfs = {}
            for p in positions:
                hist = get_price_history(p.ticker, days=30)
                if len(hist) >= 5:
                    price_dfs[p.ticker] = pd.Series(
                        [h.price_eur for h in hist],
                        index=[h.timestamp for h in hist],
                        name=p.ticker,
                    )
            if len(price_dfs) >= 3:
                combined = pd.DataFrame(price_dfs).dropna()
                corr = combined.pct_change().dropna().corr()
                fig = px.imshow(
                    corr, text_auto=".2f",
                    color_continuous_scale="RdYlGn",
                    zmin=-1, zmax=1,
                    title="30-Day Rolling Correlation",
                )
                fig.update_layout(height=500, paper_bgcolor="#0e1117", font=dict(color="#ccc"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 3 positions with 5+ days of history.")
        except Exception as e:
            st.warning(f"Correlation matrix unavailable: {e}")

    # ── Tab 4: P&L waterfall ──────────────────────────────────
    with tab4:
        pos_sorted = sorted(
            [p for p in positions if p.pnl_eur is not None],
            key=lambda x: x.pnl_eur or 0, reverse=True
        )
        if pos_sorted:
            fig = go.Figure(go.Bar(
                x=[p.ticker for p in pos_sorted],
                y=[p.pnl_eur for p in pos_sorted],
                marker_color=["#00ff88" if (p.pnl_eur or 0) >= 0 else "#ff4444"
                              for p in pos_sorted],
                text=[_fmt_eur(p.pnl_eur, 0) for p in pos_sorted],
                textposition="outside",
            ))
            fig.update_layout(
                title="P&L by Position (EUR)", height=400,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#ccc"), yaxis_title="P&L (EUR)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Geo score history ──────────────────────────────
    with tab5:
        fig = go.Figure()
        colours = {
            "ENERGY": "#ff8844", "DEFENSE": "#4488ff",
            "METALS": "#aaaaaa", "GOLD": "#ffdd44", "BIOTECH": "#44ffaa",
        }
        for sector in SECTOR_CONFIG:
            hist = get_sector_score_history(sector, days=30)
            if hist:
                fig.add_trace(go.Scatter(
                    x=[h.timestamp for h in hist],
                    y=[h.geo_score for h in hist],
                    mode="lines+markers",
                    name=sector,
                    line=dict(color=colours.get(sector, "#888"), width=2),
                    marker=dict(size=4),
                ))
        fig.update_layout(
            title="Geopolitical Sector Score History (30 days)",
            height=400, yaxis=dict(range=[0, 10], title="Geo Score"),
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#ccc"), legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# SECTION 7 — TRANCHE DEPLOYMENT TRACKER
# ─────────────────────────────────────────────────────────────

def render_tranche_tracker() -> None:
    st.subheader("Tranche Deployment Tracker")

    start_str = get_config_value("system_start_date")
    if not start_str:
        st.warning("System start date not set. Run the system to initialise.")
        return

    start_date = date.fromisoformat(start_str)
    today      = date.today()
    days_since = (today - start_date).days

    tranches = {
        1: {"label": "Tranche 1 — Immediate", "days": 0,           "colour": "#00ff88"},
        2: {"label": f"Tranche 2 — Day {TRANCHE_2_DAYS}", "days": TRANCHE_2_DAYS, "colour": "#ffaa00"},
        3: {"label": f"Tranche 3 — Day {TRANCHE_3_DAYS}", "days": TRANCHE_3_DAYS, "colour": "#ff4444"},
    }

    cols = st.columns(3)
    for i, (tranche_num, info) in enumerate(tranches.items()):
        tranche_positions = [p for p in PORTFOLIO if p["tranche"] == tranche_num]
        tranche_eur = sum(p["allocation_eur"] for p in tranche_positions)
        tickers     = [p["ticker"] for p in tranche_positions]

        days_remaining = info["days"] - days_since
        if days_remaining <= 0:
            status = "DEPLOY NOW" if tranche_num > 1 else "DEPLOYED"
            status_col = "#00ff88"
        else:
            status = f"{days_remaining} days remaining"
            status_col = info["colour"]

        with cols[i]:
            st.markdown(
                f"**{info['label']}**<br>"
                f'<span style="color:{status_col}">{status}</span><br>'
                f"Total: **€{tranche_eur:,}**<br>"
                f"Positions: {', '.join(tickers)}",
                unsafe_allow_html=True,
            )
            # Progress bar
            if tranche_num > 1 and info["days"] > 0:
                progress = min(1.0, days_since / info["days"])
                st.progress(progress)


# ─────────────────────────────────────────────────────────────
# SECTION 8 — PREDICTIONS
# ─────────────────────────────────────────────────────────────

def render_predictions(geo_context: GeoContext) -> None:
    st.subheader("Predictions")
    st.caption(
        "These are statistical model estimates, not financial advice. "
        "Past performance and model predictions do not guarantee future results. "
        "Always consult a licensed financial advisor before making investment decisions."
    )

    # ── Market Regime Indicator ───────────────────────────────
    regime = prediction_engine.get_current_regime()
    if regime:
        regime_colours = {
            "CALM": "#00ff88", "ELEVATED": "#ffaa00",
            "STRESSED": "#ff6644", "CRISIS": "#ff0000",
        }
        rc = regime_colours.get(regime.regime_name, "#888")
        pattern_str = ", ".join(
            f"{p[0]} ({p[1]:.0%})" for p in regime.active_patterns
        ) if regime.active_patterns else "None"

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.markdown(
                f'**Market Regime:** <span style="color:{rc};font-size:1.1rem">'
                f'{regime.regime_name}</span> ({regime.regime_score:.0f}/100)',
                unsafe_allow_html=True,
            )
        with rc2:
            st.markdown(
                f"**Cross-Sector Correlation:** {regime.cross_sector_correlation:.2f}<br>"
                f"<small>VIX: {regime.vix_level:.0f} | Vol z-score: {regime.vol_regime_z:.1f}</small>",
                unsafe_allow_html=True,
            )
        with rc3:
            contagion_col = "#ff4444" if regime.contagion_risk > 0.5 else (
                "#ffaa00" if regime.contagion_risk > 0.25 else "#00ff88"
            )
            st.markdown(
                f'**Contagion Risk:** <span style="color:{contagion_col}">'
                f'{regime.contagion_risk:.0%}</span>',
                unsafe_allow_html=True,
            )
        with rc4:
            st.markdown(f"**Active Patterns:** {pattern_str}")

        # Show regime factors
        if regime.factors:
            with st.expander("Regime Analysis Detail", expanded=False):
                for f in regime.factors:
                    st.text(f"  {f}")
                if regime.sector_adjustments:
                    st.markdown("**Crisis-adjusted sector biases (weekly %):**")
                    adj_parts = []
                    for sec, adj in sorted(regime.sector_adjustments.items(),
                                           key=lambda x: x[1], reverse=True):
                        col = "#00ff88" if adj > 0 else "#ff4444" if adj < 0 else "#888"
                        adj_parts.append(
                            f'<span style="color:{col}">{sec}: {adj:+.1f}%</span>'
                        )
                    st.markdown(" | ".join(adj_parts), unsafe_allow_html=True)

        st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Forecast Overview", "Position Forecasts",
        "Confidence Intervals", "Prediction Accuracy",
        "Geopolitical Scenario Analysis",
    ])

    positions = price_engine.get_all_positions()
    pos_preds = prediction_engine.get_all_position_predictions()
    sec_preds = prediction_engine.get_all_sector_predictions()
    port_preds = prediction_engine.get_all_portfolio_predictions()

    # ── Tab 1: Forecast Overview ──────────────────────────────
    with tab1:
        if not port_preds:
            st.info("Predictions are being generated. Check back in a few minutes.")
        else:
            cols = st.columns(3)
            for i, h in enumerate(["24h", "1w", "1m"]):
                pp = port_preds.get(h)
                if not pp:
                    continue
                labels = {"24h": "Next 24 Hours", "1w": "Next 1 Week", "1m": "Next 1 Month"}
                dir_arrow = {"UP": "^", "DOWN": "v", "FLAT": "~"}.get(pp.direction, "~")
                dir_col = {"UP": "#00ff88", "DOWN": "#ff4444", "FLAT": "#888"}.get(pp.direction, "#888")
                conf_col = {"HIGH": "#00ff88", "MEDIUM": "#ffaa00", "LOW": "#ff4444"}.get(pp.overall_confidence, "#888")

                with cols[i]:
                    st.markdown(
                        f"**{labels.get(h, h)}**<br>"
                        f"Predicted: **{_fmt_eur(pp.predicted_value_eur)}**<br>"
                        f'<span style="color:{dir_col};font-size:1.3rem">{dir_arrow} {pp.predicted_change_pct:+.2f}%</span><br>'
                        f'Confidence: <span style="color:{conf_col}">{pp.overall_confidence}</span><br>'
                        f"P&L: {_fmt_eur(pp.predicted_pnl_eur)}<br>"
                        f"<small>95% CI: {_fmt_eur(pp.ci_95[0])} — {_fmt_eur(pp.ci_95[1])}</small><br>"
                        f"<small>{pp.risk_summary}</small>",
                        unsafe_allow_html=True,
                    )

            # Sector forecast bars
            st.markdown("---")
            st.markdown("**Sector Forecast (% Change)**")
            for h in ["24h", "1w", "1m"]:
                sector_data = []
                for sector in SECTOR_CONFIG:
                    sp = sec_preds.get(sector, {}).get(h)
                    if sp:
                        sector_data.append({
                            "Sector": sector,
                            "Change %": sp.predicted_change_pct,
                            "Direction": sp.direction,
                        })
                if sector_data:
                    df = pd.DataFrame(sector_data)
                    colours = ["#00ff88" if r >= 0 else "#ff4444" for r in df["Change %"]]
                    fig = go.Figure(go.Bar(
                        x=df["Change %"], y=df["Sector"],
                        orientation="h",
                        marker_color=colours,
                        text=[f"{v:+.2f}%" for v in df["Change %"]],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title=f"Sector Forecast — {h}",
                        height=200, margin=dict(l=80, r=40, t=40, b=20),
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="#ccc"), xaxis_title="Predicted Change %",
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"sec_bar_{h}")

    # ── Tab 2: Position Forecasts ─────────────────────────────
    with tab2:
        if not pos_preds:
            st.info("No position predictions yet.")
        else:
            rows = []
            for pos in positions:
                tp = pos_preds.get(pos.ticker, {})
                row = {
                    "Ticker": pos.ticker,
                    "Current EUR": f"{pos.current_price_eur:.2f}" if pos.current_price_eur else "N/A",
                }
                for h in ["24h", "1w", "1m"]:
                    pred = tp.get(h)
                    if pred:
                        row[f"Pred {h} EUR"] = f"{pred.predicted_price_eur:.2f}"
                        row[f"{h} %"] = f"{pred.predicted_change_pct:+.2f}%"
                    else:
                        row[f"Pred {h} EUR"] = "N/A"
                        row[f"{h} %"] = "N/A"

                # Use first available horizon for direction/confidence/factor
                first_pred = tp.get("24h") or tp.get("1w") or tp.get("1m")
                row["Direction"] = first_pred.direction if first_pred else "N/A"
                row["Confidence"] = first_pred.confidence_level if first_pred else "N/A"
                row["Top Factor"] = first_pred.factors[0][:60] if first_pred and first_pred.factors else "N/A"
                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True, height=500)

            # Expandable detail
            sel = st.selectbox(
                "Expand position detail:",
                options=[""] + list(pos_preds.keys()),
                format_func=lambda x: x if x else "-- select --",
                key="pred_detail_ticker",
            )
            if sel and sel in pos_preds:
                for h in ["24h", "1w", "1m"]:
                    pred = pos_preds[sel].get(h)
                    if not pred:
                        continue
                    with st.expander(f"{sel} — {h}", expanded=(h == "24h")):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(
                                f"**Direction:** {pred.direction} | **Confidence:** {pred.confidence_level}<br>"
                                f"**Predicted:** {pred.predicted_price_eur:.2f} EUR ({pred.predicted_change_pct:+.2f}%)<br>"
                                f"**80% CI:** {pred.ci_80[0]:.2f} — {pred.ci_80[1]:.2f}<br>"
                                f"**95% CI:** {pred.ci_95[0]:.2f} — {pred.ci_95[1]:.2f}<br>"
                                f"**Model:** {pred.model_used}",
                                unsafe_allow_html=True,
                            )
                        with c2:
                            st.markdown("**Model Scores (% change)**")
                            for model, score in pred.model_scores.items():
                                st.text(f"  {model}: {score:+.2f}%")
                            st.markdown("**Top Factors**")
                            for f in pred.factors:
                                st.text(f"  - {f}")
                        if pred.warning:
                            st.warning(pred.warning)

    # ── Tab 3: Confidence Intervals (fan charts) ─────────────
    with tab3:
        if not pos_preds:
            st.info("No predictions yet for fan charts.")
        else:
            sel_fan = st.selectbox(
                "Select position for fan chart:",
                options=list(pos_preds.keys()),
                key="fan_ticker",
            )
            if sel_fan and sel_fan in pos_preds:
                preds = pos_preds[sel_fan]
                pos_obj = next((p for p in positions if p.ticker == sel_fan), None)
                current = pos_obj.current_price_eur if pos_obj else 0

                if current and preds:
                    # Build forward-looking points
                    today = datetime.now(timezone.utc)
                    dates = [today]
                    central = [current]
                    ci80_lo = [current]
                    ci80_hi = [current]
                    ci95_lo = [current]
                    ci95_hi = [current]

                    for h_name, h_days in [("24h", 1), ("1w", 5), ("1m", 21)]:
                        pred = preds.get(h_name)
                        if pred:
                            dates.append(today + timedelta(days=h_days))
                            central.append(pred.predicted_price_eur)
                            ci80_lo.append(pred.ci_80[0])
                            ci80_hi.append(pred.ci_80[1])
                            ci95_lo.append(pred.ci_95[0])
                            ci95_hi.append(pred.ci_95[1])

                    fig = go.Figure()
                    # 95% CI band
                    fig.add_trace(go.Scatter(
                        x=dates + dates[::-1],
                        y=ci95_hi + ci95_lo[::-1],
                        fill="toself", fillcolor="rgba(100,100,255,0.1)",
                        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
                    ))
                    # 80% CI band
                    fig.add_trace(go.Scatter(
                        x=dates + dates[::-1],
                        y=ci80_hi + ci80_lo[::-1],
                        fill="toself", fillcolor="rgba(100,100,255,0.25)",
                        line=dict(color="rgba(0,0,0,0)"), name="80% CI",
                    ))
                    # Central line
                    fig.add_trace(go.Scatter(
                        x=dates, y=central,
                        mode="lines+markers", name="Predicted",
                        line=dict(color="#00aaff", width=2),
                    ))
                    # Current price dot
                    fig.add_trace(go.Scatter(
                        x=[today], y=[current],
                        mode="markers", name="Current",
                        marker=dict(color="red", size=10),
                    ))
                    fig.update_layout(
                        title=f"{sel_fan} — Price Forecast Fan Chart",
                        height=400, yaxis_title="Price (EUR)",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="#ccc"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Prediction Accuracy ────────────────────────────
    with tab4:
        metrics = accuracy_tracker.get_metrics(days=30)
        if not metrics:
            st.info(
                "No matured predictions yet. Accuracy tracking begins after predictions "
                "reach their target dates (24h for first results)."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Direction Accuracy", f"{metrics['direction_accuracy_pct']:.1f}%")
            c2.metric("80% CI Hit Rate", f"{metrics['ci80_hit_rate']:.1f}%",
                       help="Target: ~80%")
            c3.metric("MAE", f"{metrics['mae']:.2f}%")
            c4.metric("Total Predictions", str(metrics["total_predictions"]))

            # By horizon
            by_h = metrics.get("by_horizon", {})
            if by_h:
                st.markdown("**Accuracy by Horizon**")
                h_rows = []
                for h, m in by_h.items():
                    h_rows.append({
                        "Horizon": h, "Count": m["total"],
                        "Direction %": f"{m['direction_accuracy']:.1f}%",
                        "CI80 Hit %": f"{m['ci80_hit_rate']:.1f}%",
                        "MAE": f"{m['mae']:.2f}%",
                    })
                st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

            # By sector
            by_s = metrics.get("by_sector", {})
            if by_s:
                st.markdown("**Accuracy by Sector**")
                s_rows = []
                for sec, m in by_s.items():
                    s_rows.append({
                        "Sector": sec, "Count": m["total"],
                        "Direction %": f"{m['direction_accuracy']:.1f}%",
                        "MAE": f"{m['mae']:.2f}%",
                    })
                st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

    # ── Tab 5: Geopolitical Scenario Analysis ─────────────────
    with tab5:
        st.markdown("**What-If Scenario Analysis**")
        st.caption("Adjust geopolitical variables to see how predictions would change.")

        from config import GEO_VARIABLES_DEFAULT

        geo_options = {
            "HORMUZ_STATUS":      ["OPEN", "PARTIAL", "CLOSED"],
            "IRAN_CONFLICT":      ["ACTIVE", "CEASEFIRE", "RESOLVED"],
            "UKRAINE_WAR":        ["ESCALATING", "STALEMATE", "DE-ESCALATING", "RESOLVED"],
            "US_CHINA_RELATIONS": ["HOSTILE", "TENSE", "NEUTRAL", "COOPERATIVE"],
            "NATO_SPENDING":      ["DECLINING", "STABLE", "INCREASING", "ACCELERATING"],
        }

        # Current values
        db_states = {}
        try:
            from src.database import get_geo_states as _get_gs
            raw = _get_gs()
            db_states = {k: v.current_value for k, v in raw.items()}
        except Exception:
            pass

        scenario_vars = {}
        cols = st.columns(5)
        for i, (var, options) in enumerate(geo_options.items()):
            current = db_states.get(var, GEO_VARIABLES_DEFAULT.get(var, options[0]))
            idx = options.index(current) if current in options else 0
            with cols[i]:
                scenario_vars[var] = st.selectbox(
                    var.replace("_", " ").title(),
                    options=options,
                    index=idx,
                    key=f"scenario_{var}",
                )

        if st.button("Run Scenario", key="run_scenario"):
            with st.spinner("Running scenario analysis..."):
                scenario_results = prediction_engine.run_scenario(scenario_vars)

            if not scenario_results:
                st.warning("Could not generate scenario predictions.")
            else:
                st.markdown("---")
                st.markdown("**Scenario vs Current Forecast**")

                rows = []
                for h in ["24h", "1w", "1m"]:
                    current_pp = port_preds.get(h)
                    scenario_pp = scenario_results.get(h)
                    if current_pp and scenario_pp:
                        delta = scenario_pp.predicted_change_pct - current_pp.predicted_change_pct
                        rows.append({
                            "Horizon": h,
                            "Current Forecast": f"{current_pp.predicted_change_pct:+.2f}%",
                            "Scenario Forecast": f"{scenario_pp.predicted_change_pct:+.2f}%",
                            "Delta": f"{delta:+.2f}%",
                            "Scenario Direction": scenario_pp.direction,
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Sector breakdown
                st.markdown("**Sector Impact**")
                sec_rows = []
                for sector in SECTOR_CONFIG:
                    for h in ["1w"]:  # Show 1-week horizon for sector detail
                        cur_sp = sec_preds.get(sector, {}).get(h)
                        # Get scenario sector predictions
                        scen_pp = scenario_results.get(h)
                        if cur_sp and scen_pp:
                            scen_sp = next(
                                (s for s in scen_pp.sector_predictions if s.sector == sector),
                                None,
                            )
                            if scen_sp:
                                delta = scen_sp.predicted_change_pct - cur_sp.predicted_change_pct
                                sec_rows.append({
                                    "Sector": sector,
                                    "Current 1W": f"{cur_sp.predicted_change_pct:+.2f}%",
                                    "Scenario 1W": f"{scen_sp.predicted_change_pct:+.2f}%",
                                    "Delta": f"{delta:+.2f}%",
                                })
                if sec_rows:
                    st.dataframe(pd.DataFrame(sec_rows), use_container_width=True, hide_index=True)

                    # Highlight biggest movers
                    biggest = max(sec_rows, key=lambda r: abs(float(r["Delta"].replace("%", ""))))
                    delta_val = float(biggest["Delta"].replace("%", ""))
                    verb = "gain" if delta_val > 0 else "lose"
                    st.success(
                        f"Biggest mover: **{biggest['Sector']}** would {verb} "
                        f"**{biggest['Delta']}** under this scenario."
                    )


# ─────────────────────────────────────────────────────────────
# MAIN RENDER LOOP
# ─────────────────────────────────────────────────────────────

def main():
    geo_context = geo_scorer.get_geo_context()
    signals     = signal_engine.get_all_signals()

    st.title("Geopolitical Investment Tracker")
    st.caption(
        f"Luxembourg investor | €100k portfolio | HIGH risk | 5-year horizon | "
        f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    render_header(geo_context)
    st.markdown("---")

    render_geo_board(geo_context)
    st.markdown("---")

    render_sector_overview(geo_context, signals)
    st.markdown("---")

    selected_ticker = render_position_table(signals)

    if selected_ticker:
        st.markdown("---")
        render_signal_rationale(selected_ticker, geo_context)

    st.markdown("---")
    render_alerts_feed()
    st.markdown("---")

    render_charts(geo_context)
    st.markdown("---")

    render_predictions(geo_context)
    st.markdown("---")

    render_tranche_tracker()

    # Auto-refresh every 60 seconds
    st.markdown(
        """
        <script>
          setTimeout(function() { window.location.reload(); }, 60000);
        </script>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
