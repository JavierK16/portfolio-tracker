"""
alert_manager.py — Alert generation, desktop notifications, and log writing.
Monitors price moves, signal changes, geo events, drawdowns, and rebalancing needs.
"""

import json
import logging
import logging.handlers
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from src.database import save_alert, get_unacknowledged_alerts, acknowledge_alert
from src.price_engine import PriceEngine, PositionData, get_price_engine
from src.geo_scorer import GeopoliticalScorer, get_geo_scorer, GeoContext
from src.signal_engine import SignalEngine, SignalResult, get_signal_engine

logger = logging.getLogger(__name__)

# Keep track of last known state to detect changes
_last_day_change: Dict[str, float] = {}
_last_pnl_pct: Dict[str, float] = {}
_last_signals: Dict[str, str] = {}
_last_geo_vars: Dict[str, str] = {}
_last_portfolio_value: Optional[float] = None
_tranche_reminders_sent: Dict[int, bool] = {2: False, 3: False}

_lock = threading.Lock()


def _setup_alert_log() -> None:
    from config import ALERT_LOG, LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)
    handler = logging.handlers.TimedRotatingFileHandler(
        ALERT_LOG, when="midnight", backupCount=30, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    alert_logger = logging.getLogger("alerts")
    alert_logger.setLevel(logging.INFO)
    if not alert_logger.handlers:
        alert_logger.addHandler(handler)


_setup_alert_log()
_alert_log = logging.getLogger("alerts")


def _desktop_notify(title: str, message: str) -> None:
    """Send desktop notification via plyer (best-effort)."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message[:256],
            app_name="Portfolio Tracker",
            timeout=8,
        )
    except Exception as e:
        logger.debug("Desktop notification failed: %s", e)


def _emit_alert(
    severity: str,
    alert_type: str,
    message: str,
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
    notify: bool = False,
) -> None:
    """Save alert to DB, log it, and optionally send desktop notification."""
    alert_id = save_alert(severity, alert_type, message, ticker, sector)
    log_msg = f"[{severity}] {alert_type} | {ticker or ''} | {message}"
    _alert_log.info(log_msg)
    if severity in ("HIGH", "CRITICAL") or notify:
        _desktop_notify(f"[{severity}] {alert_type}", message[:200])
    logger.info("Alert emitted: %s", log_msg)


# ─────────────────────────────────────────────────────────────
# PRICE ALERTS
# ─────────────────────────────────────────────────────────────

def check_price_alerts(positions: List[PositionData]) -> None:
    global _last_portfolio_value
    from config import (
        ALERT_DAY_MOVE_MEDIUM, ALERT_DAY_MOVE_HIGH, ALERT_PORTFOLIO_DAY_PCT
    )

    total_value = sum(p.current_value_eur or 0 for p in positions)
    total_cost  = sum((p.shares_units or 0) * (p.entry_price_eur or 0) for p in positions)

    # Portfolio day change
    with _lock:
        if _last_portfolio_value and _last_portfolio_value > 0:
            pct_change = ((total_value - _last_portfolio_value) / _last_portfolio_value) * 100
            if abs(pct_change) >= ALERT_PORTFOLIO_DAY_PCT:
                _emit_alert(
                    "HIGH", "PORTFOLIO_DAY_CHANGE",
                    f"Portfolio value changed {pct_change:+.2f}% "
                    f"(€{total_value:,.0f} vs €{_last_portfolio_value:,.0f})",
                    notify=True,
                )
        _last_portfolio_value = total_value

    for pos in positions:
        if pos.day_change_pct is None or pos.data_status == "N/A":
            continue

        with _lock:
            prev = _last_day_change.get(pos.ticker)
            _last_day_change[pos.ticker] = pos.day_change_pct

        abs_chg = abs(pos.day_change_pct)
        if abs_chg >= ALERT_DAY_MOVE_HIGH:
            _emit_alert(
                "HIGH", "PRICE_MOVE",
                f"{pos.ticker} moved {pos.day_change_pct:+.2f}% today "
                f"(€{pos.current_price_eur or 0:.2f})",
                ticker=pos.ticker, sector=pos.sector, notify=True,
            )
        elif abs_chg >= ALERT_DAY_MOVE_MEDIUM:
            _emit_alert(
                "MEDIUM", "PRICE_MOVE",
                f"{pos.ticker} moved {pos.day_change_pct:+.2f}% today",
                ticker=pos.ticker, sector=pos.sector,
            )


# ─────────────────────────────────────────────────────────────
# SIGNAL CHANGE ALERTS
# ─────────────────────────────────────────────────────────────

def check_signal_alerts(signals: Dict[str, SignalResult]) -> None:
    for ticker, result in signals.items():
        if result.signal_change and result.previous_signal:
            _emit_alert(
                "HIGH", "SIGNAL_CHANGE",
                f"{ticker}: signal changed from {result.previous_signal} → {result.signal} "
                f"(composite score: {result.composite_score:.0f})",
                ticker=ticker, notify=True,
            )


# ─────────────────────────────────────────────────────────────
# GEOPOLITICAL ALERTS
# ─────────────────────────────────────────────────────────────

def check_geo_alerts(geo_context: GeoContext) -> None:
    with _lock:
        for var, info in geo_context.variables.items():
            current = info.get("value")
            prev = _last_geo_vars.get(var)

            if prev and prev != current:
                severity = "CRITICAL" if var == "HORMUZ_STATUS" else "HIGH"
                _emit_alert(
                    severity, "GEO_VARIABLE_CHANGE",
                    f"{var} changed: {prev} → {current} | "
                    f"Trigger: {info.get('headline', 'N/A')[:120]}",
                    notify=True,
                )
                if var == "HORMUZ_STATUS":
                    _emit_alert(
                        "CRITICAL", "HORMUZ_STATUS_CHANGE",
                        f"CRITICAL: Hormuz Status changed to {current}. "
                        "Recalculating energy sector signals immediately.",
                        sector="ENERGY", notify=True,
                    )

            _last_geo_vars[var] = current or ""

    # Alert on news matching >3 trigger keywords
    for item in geo_context.recent_news:
        if len(item.sectors_matched) >= 3:
            _emit_alert(
                "MEDIUM", "MULTI_SECTOR_NEWS",
                f"Article matches {len(item.sectors_matched)} sectors: "
                f"{', '.join(item.sectors_matched)} | {item.title[:100]}",
            )


# ─────────────────────────────────────────────────────────────
# REBALANCING ALERTS
# ─────────────────────────────────────────────────────────────

def check_rebalancing_alerts(positions: List[PositionData]) -> None:
    from config import ALERT_DRIFT_THRESHOLD, TRANCHE_2_DAYS, TRANCHE_3_DAYS
    from src.database import get_config_value

    for pos in positions:
        if pos.drift_from_target is None:
            continue
        if abs(pos.drift_from_target) >= ALERT_DRIFT_THRESHOLD:
            direction = "overweight" if pos.drift_from_target > 0 else "underweight"
            action    = "TRIM" if pos.drift_from_target > 0 else "ADD"
            _emit_alert(
                "MEDIUM", "REBALANCING",
                f"{pos.ticker} is {direction} by {abs(pos.drift_from_target):.1f}% "
                f"(current: {pos.weight_current_pct or 0:.1f}%, "
                f"target: {pos.target_pct:.1f}%) — consider {action}",
                ticker=pos.ticker, sector=pos.sector,
            )

    # Tranche deployment reminders
    start_str = get_config_value("system_start_date")
    if not start_str:
        return
    try:
        from datetime import date
        start_date = date.fromisoformat(start_str)
        today = date.today()
        days_elapsed = (today - start_date).days

        for tranche_num, tranche_days in [(2, TRANCHE_2_DAYS), (3, TRANCHE_3_DAYS)]:
            if days_elapsed >= tranche_days and not _tranche_reminders_sent.get(tranche_num):
                from config import PORTFOLIO
                tranche_tickers = [
                    p["ticker"] for p in PORTFOLIO if p["tranche"] == tranche_num
                ]
                tranche_eur = sum(
                    p["allocation_eur"] for p in PORTFOLIO if p["tranche"] == tranche_num
                )
                _emit_alert(
                    "REMINDER", "TRANCHE_DEPLOYMENT",
                    f"Tranche {tranche_num} deployment due (day {tranche_days}). "
                    f"Positions: {', '.join(tranche_tickers)} — Total: €{tranche_eur:,}",
                    notify=True,
                )
                with _lock:
                    _tranche_reminders_sent[tranche_num] = True
    except Exception as e:
        logger.debug("Tranche reminder check error: %s", e)


# ─────────────────────────────────────────────────────────────
# DRAWDOWN ALERTS
# ─────────────────────────────────────────────────────────────

def check_drawdown_alerts(positions: List[PositionData]) -> None:
    from config import ALERT_DRAWDOWN_MEDIUM, ALERT_DRAWDOWN_CRITICAL

    for pos in positions:
        if pos.pnl_pct is None:
            continue

        with _lock:
            prev_pnl = _last_pnl_pct.get(pos.ticker)

        if prev_pnl is not None:
            # Only alert when crossing threshold for first time (edge trigger)
            if pos.pnl_pct <= ALERT_DRAWDOWN_CRITICAL < prev_pnl:
                _emit_alert(
                    "CRITICAL", "STOP_LOSS_REVIEW",
                    f"{pos.ticker} down {pos.pnl_pct:.1f}% from entry — "
                    "STOP-LOSS REVIEW REQUIRED. Reassess investment thesis.",
                    ticker=pos.ticker, sector=pos.sector, notify=True,
                )
            elif pos.pnl_pct <= ALERT_DRAWDOWN_MEDIUM < prev_pnl:
                _emit_alert(
                    "MEDIUM", "DRAWDOWN_WARNING",
                    f"{pos.ticker} down {pos.pnl_pct:.1f}% from entry — "
                    "monitor closely",
                    ticker=pos.ticker, sector=pos.sector,
                )

        with _lock:
            _last_pnl_pct[pos.ticker] = pos.pnl_pct


# ─────────────────────────────────────────────────────────────
# MAIN CHECK RUNNER
# ─────────────────────────────────────────────────────────────

class AlertManager:
    """Orchestrates all alert checks on a background thread."""

    def __init__(
        self,
        price_engine: PriceEngine,
        geo_scorer: GeopoliticalScorer,
        signal_engine: SignalEngine,
    ):
        self._price  = price_engine
        self._geo    = geo_scorer
        self._signal = signal_engine

    def run_checks(self) -> None:
        positions    = self._price.get_all_positions()
        geo_context  = self._geo.get_geo_context()
        signals      = self._signal.get_all_signals()

        check_price_alerts(positions)
        check_signal_alerts(signals)
        check_geo_alerts(geo_context)
        check_rebalancing_alerts(positions)
        check_drawdown_alerts(positions)

    def get_unacknowledged(self, limit: int = 200):
        return get_unacknowledged_alerts(limit)

    def acknowledge(self, alert_id: int) -> None:
        acknowledge_alert(alert_id)

    def start_background_checks(self, interval: int = 300) -> None:
        def _loop():
            time.sleep(30)  # Small initial delay
            while True:
                try:
                    self.run_checks()
                except Exception as e:
                    logger.error("AlertManager check error: %s", e)
                time.sleep(interval)

        t = threading.Thread(target=_loop, daemon=True, name="alert-checks")
        t.start()
        logger.info("AlertManager background checks started")


# Module-level singleton
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(
            get_price_engine(), get_geo_scorer(), get_signal_engine()
        )
    return _alert_manager
