"""
run_terminal.py — Rich terminal dashboard.
Run with: python run_terminal.py
Backtesting: python run_terminal.py --backtest
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta, date
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.logging_setup import setup_logging
setup_logging()

from config import INVESTOR_PROFILE, PORTFOLIO, SECTOR_CONFIG
from src.database import init_db, get_config_value, set_config_value, get_signal_history

console = Console()

logging.basicConfig(level=logging.WARNING)  # Quiet in terminal mode

# ─────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────

SIGNAL_STYLES = {
    "BUY":    "bold green",
    "ADD":    "green",
    "HOLD":   "dim white",
    "REDUCE": "yellow",
    "SELL":   "bold red",
}

GEO_STYLES = {
    "CLOSED":  "bold red", "PARTIAL": "yellow", "OPEN": "bold green",
    "ACTIVE":  "bold red", "CEASEFIRE": "yellow", "RESOLVED": "bold green",
    "ESCALATING": "bold red", "STALEMATE": "yellow",
    "DE-ESCALATING": "green", "RESOLVED": "bold green",
    "HOSTILE": "bold red", "TENSE": "yellow",
    "NEUTRAL": "green", "COOPERATIVE": "bold green",
    "DECLINING": "bold red", "STABLE": "yellow",
    "INCREASING": "green", "ACCELERATING": "bold green",
}

SEV_STYLES = {
    "CRITICAL": "bold red",
    "HIGH":     "bold yellow",
    "MEDIUM":   "yellow",
    "REMINDER": "cyan",
}


def _pct_style(v: Optional[float]) -> str:
    if v is None:
        return "dim"
    return "green" if v >= 0 else "red"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.1f}%"


def _fmt_eur(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"€{v:,.0f}"


# ─────────────────────────────────────────────────────────────
# PANELS
# ─────────────────────────────────────────────────────────────

def build_summary_panel(price_engine, geo_scorer, signal_engine, alert_manager) -> Panel:
    total_val  = price_engine.get_portfolio_value()
    day_pnl    = price_engine.get_portfolio_day_pnl()
    total_pnl  = price_engine.get_portfolio_total_pnl()
    alerts     = alert_manager.get_unacknowledged(limit=100)
    n_crit     = sum(1 for a in alerts if a.severity == "CRITICAL")
    n_high     = sum(1 for a in alerts if a.severity == "HIGH")
    n_med      = sum(1 for a in alerts if a.severity == "MEDIUM")

    geo_context = geo_scorer.get_geo_context()

    # Portfolio metrics
    port_text = Text()
    port_text.append("Portfolio: ", style="bold")
    port_text.append(f"{_fmt_eur(total_val)}  ", style="bold cyan")
    port_text.append("Day P&L: ", style="bold")
    port_text.append(f"{_fmt_eur(day_pnl)} ", style=_pct_style(day_pnl))
    port_text.append(" | Total P&L: ", style="bold")
    port_text.append(f"{_fmt_eur(total_pnl)}", style=_pct_style(total_pnl))

    # Alerts
    alert_text = Text()
    alert_text.append("Alerts: ", style="bold")
    if n_crit:
        alert_text.append(f"{n_crit} CRITICAL  ", style="bold red")
    if n_high:
        alert_text.append(f"{n_high} HIGH  ", style="bold yellow")
    if n_med:
        alert_text.append(f"{n_med} MEDIUM  ", style="yellow")
    if not (n_crit or n_high or n_med):
        alert_text.append("None", style="bold green")

    # Geo status row
    geo_text = Text()
    var_order = [
        "HORMUZ_STATUS", "IRAN_CONFLICT", "UKRAINE_WAR",
        "US_CHINA_RELATIONS", "NATO_SPENDING"
    ]
    for var in var_order:
        info  = geo_context.variables.get(var, {})
        value = info.get("value", "?")
        style = GEO_STYLES.get(value, "white")
        short_var = var.replace("_", " ").title().replace("Hormuz Status", "Hormuz").replace(
            "Iran Conflict", "Iran").replace("Ukraine War", "Ukraine").replace(
            "Us China Relations", "US-CN").replace("Nato Spending", "NATO")[:7]
        geo_text.append(f"{short_var}: ", style="bold")
        geo_text.append(f"{value}  ", style=style)

    content = Text()
    content.append_text(port_text)
    content.append("\n")
    content.append_text(alert_text)
    content.append("\n")
    content.append_text(geo_text)

    return Panel(content, title="[bold]Portfolio Summary & Geo Status[/bold]",
                 border_style="cyan", padding=(0, 1))


def build_position_table(price_engine, signal_engine) -> Table:
    positions = price_engine.get_all_positions()
    signals   = signal_engine.get_all_signals()

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        row_styles=["", "dim"],
        expand=True,
    )

    cols = [
        ("Ticker",  6),  ("Sector", 8), ("Value €", 10), ("P&L €",  9),
        ("P&L %",   7),  ("Day %",  7),  ("Wk %",    7),  ("Wt%",    5),
        ("Tgt%",    5),  ("Drift",  6),  ("Signal",  8),  ("Score",  5),
        ("Status",  7),
    ]
    for name, _ in cols:
        table.add_column(name, no_wrap=True)

    for pos in sorted(positions, key=lambda p: p.sector):
        sig     = signals.get(pos.ticker)
        signal  = sig.signal if sig else "HOLD"
        score   = f"{sig.composite_score:.0f}" if sig and sig.composite_score else "—"
        drift   = pos.drift_from_target or 0
        drift_s = "red" if abs(drift) > 3 else ("green" if abs(drift) <= 1 else "yellow")

        table.add_row(
            pos.ticker,
            pos.sector,
            _fmt_eur(pos.current_value_eur),
            Text(_fmt_eur(pos.pnl_eur), style=_pct_style(pos.pnl_eur)),
            Text(_fmt_pct(pos.pnl_pct), style=_pct_style(pos.pnl_pct)),
            Text(_fmt_pct(pos.day_change_pct), style=_pct_style(pos.day_change_pct)),
            Text(_fmt_pct(pos.week_change_pct), style=_pct_style(pos.week_change_pct)),
            f"{pos.weight_current_pct:.1f}%" if pos.weight_current_pct else "—",
            f"{pos.target_pct:.1f}%",
            Text(f"{drift:+.1f}%", style=drift_s),
            Text(signal, style=SIGNAL_STYLES.get(signal, "white")),
            score,
            Text(pos.data_status, style="green" if pos.data_status == "LIVE" else "yellow"),
        )

    return table


def build_alerts_panel(alert_manager) -> Panel:
    alerts = alert_manager.get_unacknowledged(limit=5)
    if not alerts:
        return Panel(Text("No unacknowledged alerts.", style="green"),
                     title="[bold]Last Alerts[/bold]", border_style="green")

    text = Text()
    for a in alerts[:5]:
        ts  = a.timestamp.strftime("%m/%d %H:%M") if a.timestamp else "—"
        sty = SEV_STYLES.get(a.severity, "white")
        text.append(f"[{a.severity}] ", style=sty)
        text.append(f"{ts} ", style="dim")
        if a.ticker:
            text.append(f"[{a.ticker}] ", style="bold")
        text.append(f"{a.message[:80]}\n")

    return Panel(text, title="[bold]Last 5 Alerts[/bold]", border_style="yellow", padding=(0, 1))


# ─────────────────────────────────────────────────────────────
# BACKTEST MODE
# ─────────────────────────────────────────────────────────────

def run_backtest(price_engine):
    console.print(Panel("[bold cyan]BACKTEST MODE — Replaying last 30 days[/bold]",
                        border_style="cyan"))

    hist_df = price_engine.get_historical_portfolio_values(days=30)
    if hist_df.empty:
        console.print("[red]No historical data available for backtesting.[/red]")
        return

    initial = hist_df["total_value_eur"].iloc[0] if not hist_df.empty else INVESTOR_PROFILE["total_capital"]

    table = Table(box=box.SIMPLE, title="Portfolio Value History")
    table.add_column("Date")
    table.add_column("Value €")
    table.add_column("vs Initial")
    table.add_column("P&L €")

    prev = initial
    for _, row in hist_df.iterrows():
        val  = row["total_value_eur"]
        pnl  = val - initial
        chg  = val - prev
        prev = val
        ts   = row["timestamp"]
        date_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, 'strftime') else str(ts)
        pnl_style = "green" if pnl >= 0 else "red"
        table.add_row(
            date_str,
            _fmt_eur(val),
            Text(_fmt_pct((val - initial) / initial * 100), style=pnl_style),
            Text(_fmt_eur(pnl), style=pnl_style),
        )

    console.print(table)
    console.print(f"\n[bold]Period: {len(hist_df)} data points | "
                  f"Net P&L: {_fmt_eur(hist_df['total_value_eur'].iloc[-1] - initial)}[/bold]")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio Tracker Terminal Dashboard")
    parser.add_argument("--backtest", action="store_true", help="Replay last 30 days")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    args = parser.parse_args()

    # Init
    init_db()
    if not get_config_value("system_start_date"):
        set_config_value("system_start_date", date.today().isoformat())

    from src.fx_engine import get_fx_engine
    fx = get_fx_engine()
    fx.start_background_refresh()

    from src.price_engine import get_price_engine
    from src.geo_scorer import get_geo_scorer
    from src.signal_engine import get_signal_engine
    from src.alert_manager import get_alert_manager

    price_engine  = get_price_engine()
    geo_scorer    = get_geo_scorer()
    signal_engine = get_signal_engine()
    alert_manager = get_alert_manager()

    if args.backtest:
        # Do a quick price refresh first to populate history
        console.print("[cyan]Fetching price data for backtest...[/cyan]")
        price_engine.refresh_all()
        run_backtest(price_engine)
        return

    console.print("[cyan]Starting background data refresh...[/cyan]")
    price_engine.start_background_refresh()
    geo_scorer.start_background_refresh()
    signal_engine.start_background_refresh()
    alert_manager.start_background_checks()

    console.print("[green]Dashboard running. Press Ctrl+C to exit.[/green]")
    time.sleep(2)  # Let initial refresh complete

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                summary = build_summary_panel(
                    price_engine, geo_scorer, signal_engine, alert_manager
                )
                table   = build_position_table(price_engine, signal_engine)
                alerts  = build_alerts_panel(alert_manager)

                last_ref = price_engine.last_refresh()
                last_str = last_ref.strftime("%H:%M:%S UTC") if last_ref else "—"
                footer   = Text(
                    f"  Last price update: {last_str} | "
                    f"Refreshes every {args.interval}s | "
                    f"VIX: {price_engine.get_vix() or 'N/A'}  "
                    f"Brent: ${price_engine.get_brent() or 'N/A'}",
                    style="dim",
                )

                from rich.layout import Layout
                layout = Layout()
                layout.split_column(
                    Layout(summary, size=6),
                    Layout(Panel(table, title="[bold]Portfolio Positions[/bold]",
                                 border_style="blue")),
                    Layout(alerts, size=9),
                    Layout(footer, size=1),
                )
                live.update(layout)

            except Exception as e:
                live.update(Text(f"[red]Render error: {e}[/red]"))

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
