"""
price_engine.py — Fetches real-time and historical prices via yfinance.
Calculates P&L, drift, momentum, and stores snapshots in SQLite.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf
import pandas as pd
import numpy as np

from src.database import (
    get_entry_price, save_entry_price, save_price_snapshot,
    get_price_history, get_config_value, set_config_value,
)
from src.fx_engine import get_fx_engine

logger = logging.getLogger(__name__)

VIX_TICKER = "^VIX"
BRENT_TICKER = "BZ=F"   # Brent crude futures on yfinance


@dataclass
class PositionData:
    ticker: str
    name: str
    sector: str
    instrument_type: str
    currency: str
    allocation_eur: float
    target_pct: float
    tranche: int
    # Live data (populated after refresh)
    current_price_local: Optional[float] = None
    current_price_eur: Optional[float] = None
    current_value_eur: Optional[float] = None
    shares_units: Optional[float] = None
    entry_price_local: Optional[float] = None
    entry_price_eur: Optional[float] = None
    pnl_eur: Optional[float] = None
    pnl_pct: Optional[float] = None
    weight_current_pct: Optional[float] = None
    drift_from_target: Optional[float] = None
    day_change_pct: Optional[float] = None
    week_change_pct: Optional[float] = None
    month_change_pct: Optional[float] = None
    volume: Optional[float] = None
    last_updated: Optional[datetime] = None
    data_status: str = "PENDING"   # LIVE / DELAYED / N/A / STALE
    error_msg: Optional[str] = None


class PriceEngine:
    """
    Fetches prices for all portfolio positions via yfinance.
    Runs in a background thread and refreshes every PRICE_REFRESH_INTERVAL seconds.
    """

    def __init__(self):
        self._positions: Dict[str, PositionData] = {}
        self._portfolio_value_eur: float = 0.0
        self._portfolio_day_pnl_eur: float = 0.0
        self._portfolio_total_pnl_eur: float = 0.0
        self._vix: Optional[float] = None
        self._brent_usd: Optional[float] = None
        self._lock = threading.RLock()
        self._last_refresh: Optional[datetime] = None
        self._fx = get_fx_engine()

        # Load portfolio from config
        self._load_portfolio()

    # ─────────────────────────────────────────────────────────
    # INITIALISATION
    # ─────────────────────────────────────────────────────────

    def _load_portfolio(self) -> None:
        from config import PORTFOLIO
        for pos in PORTFOLIO:
            self._positions[pos["ticker"]] = PositionData(
                ticker=pos["ticker"],
                name=pos["name"],
                sector=pos["sector"],
                instrument_type=pos["type"],
                currency=pos["currency"],
                allocation_eur=pos["allocation_eur"],
                target_pct=pos["target_pct"],
                tranche=pos["tranche"],
            )

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def refresh_all(self) -> None:
        """Fetch all prices, calculate metrics, store snapshots."""
        from config import YF_TICKER_MAP, YF_CURRENCY_OVERRIDE, YF_PENCE_TICKERS
        logger.info("PriceEngine: starting refresh")

        # Refresh FX rates if stale
        if self._fx.is_stale():
            self._fx.refresh()

        portfolio_tickers = list(self._positions.keys())

        # Build YF fetch ticker list (apply overrides)
        yf_tickers = [YF_TICKER_MAP.get(t, t) for t in portfolio_tickers]
        # Reverse map: yf_ticker → portfolio_ticker (for extraction)
        yf_to_portfolio = {YF_TICKER_MAP.get(t, t): t for t in portfolio_tickers}

        # Download VIX and Brent separately (special chars disrupt bulk)
        self._fetch_vix_brent()

        # Bulk download (default group_by='column' → level 1 = ticker)
        raw = self._bulk_download(yf_tickers)

        # Build per-portfolio-ticker series cache
        series_cache: Dict[str, pd.DataFrame] = {}
        for yf_sym in yf_tickers:
            port_ticker = yf_to_portfolio.get(yf_sym, yf_sym)
            series = self._extract_series(raw, yf_sym)
            if series is not None and not series.empty:
                # Apply pence → GBP conversion for LSE tickers priced in GBp
                if yf_sym in YF_PENCE_TICKERS:
                    series = series.copy()
                    series["Close"] = series["Close"] / 100.0
                series_cache[port_ticker] = series
            else:
                # Per-ticker fallback
                fb = self._individual_download(yf_sym)
                if fb is not None and not fb.empty:
                    if yf_sym in YF_PENCE_TICKERS:
                        fb = fb.copy()
                        fb["Close"] = fb["Close"] / 100.0
                    series_cache[port_ticker] = fb
                    logger.info("Used individual fallback for %s (→%s)", port_ticker, yf_sym)

        with self._lock:
            total_value = 0.0
            total_pnl   = 0.0
            total_day_pnl = 0.0

            for ticker, pos in self._positions.items():
                try:
                    series = series_cache.get(ticker)
                    if series is None or series.empty:
                        pos.data_status = "N/A"
                        pos.error_msg = "No data from yfinance"
                        logger.warning("No price data for %s", ticker)
                        continue

                    self._update_position(pos, series)

                    if pos.current_value_eur:
                        total_value += pos.current_value_eur
                    if pos.pnl_eur:
                        total_pnl += pos.pnl_eur

                    # Day P&L approximation
                    if (pos.day_change_pct is not None and
                            pos.current_value_eur is not None):
                        day_pnl = pos.current_value_eur * (pos.day_change_pct / 100)
                        total_day_pnl += day_pnl

                except Exception as e:
                    logger.error("Error updating %s: %s", ticker, e)
                    pos.data_status = "N/A"
                    pos.error_msg = str(e)

            # Calculate portfolio-level weights
            if total_value > 0:
                for pos in self._positions.values():
                    if pos.current_value_eur is not None:
                        pos.weight_current_pct = (pos.current_value_eur / total_value) * 100
                        pos.drift_from_target  = pos.weight_current_pct - pos.target_pct

            self._portfolio_value_eur   = total_value
            self._portfolio_total_pnl_eur = total_pnl
            self._portfolio_day_pnl_eur  = total_day_pnl
            self._last_refresh = datetime.now(timezone.utc)

        logger.info(
            "PriceEngine refresh complete — portfolio value: €%.2f",
            total_value,
        )

    def get_position(self, ticker: str) -> Optional[PositionData]:
        with self._lock:
            return self._positions.get(ticker)

    def get_all_positions(self) -> List[PositionData]:
        with self._lock:
            return list(self._positions.values())

    def get_portfolio_value(self) -> float:
        with self._lock:
            return self._portfolio_value_eur

    def get_portfolio_day_pnl(self) -> float:
        with self._lock:
            return self._portfolio_day_pnl_eur

    def get_portfolio_total_pnl(self) -> float:
        with self._lock:
            return self._portfolio_total_pnl_eur

    def get_vix(self) -> Optional[float]:
        with self._lock:
            return self._vix

    def get_brent(self) -> Optional[float]:
        with self._lock:
            return self._brent_usd

    def last_refresh(self) -> Optional[datetime]:
        with self._lock:
            return self._last_refresh

    def is_market_hours(self) -> bool:
        """Check if current CET time is within market hours."""
        from config import MARKET_HOURS_START, MARKET_HOURS_END
        now_cet = datetime.now(timezone.utc) + timedelta(hours=1)
        return (
            now_cet.weekday() < 5 and
            MARKET_HOURS_START <= now_cet.hour < MARKET_HOURS_END
        )

    # ─────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _bulk_download(tickers: List[str]) -> pd.DataFrame:
        """
        Download all tickers in one call. Default group_by='column' gives:
          Level 0 = price field (Close, Volume, …)
          Level 1 = ticker symbol
        """
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    tickers=tickers,
                    period="2mo",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    # group_by default = 'column' → tickers at level 1
                )
            return raw
        except Exception as e:
            logger.error("Bulk yfinance download failed: %s", e)
            return pd.DataFrame()

    @staticmethod
    def _individual_download(ticker: str) -> Optional[pd.DataFrame]:
        """Per-ticker fallback using yf.Ticker().history()."""
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = yf.Ticker(ticker).history(period="2mo", interval="1d", auto_adjust=True)
            if hist.empty:
                return None
            # Normalise column names (history() returns flat columns)
            cols = {}
            for c in hist.columns:
                cl = c.lower()
                if "close" in cl:
                    cols[c] = "Close"
                elif "volume" in cl:
                    cols[c] = "Volume"
            hist = hist.rename(columns=cols)
            if "Close" not in hist.columns:
                return None
            if "Volume" not in hist.columns:
                hist["Volume"] = 0.0
            return hist[["Close", "Volume"]].dropna(subset=["Close"])
        except Exception as e:
            logger.warning("Individual download failed for %s: %s", ticker, e)
            return None

    def _extract_series(self, raw: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Extract Close/Volume series for one ticker from a bulk-downloaded DataFrame.
        Handles both MultiIndex (multiple tickers) and flat (single ticker) results.
        """
        if raw is None or raw.empty:
            return None
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # Default group_by='column': level 0 = field, level 1 = ticker
                tickers_available = raw.columns.get_level_values(1).unique()
                if ticker not in tickers_available:
                    return None
                sub = raw.xs(ticker, axis=1, level=1)
                # Normalise: prefer 'Close', fall back to 'Adj Close'
                if "Close" not in sub.columns and "Adj Close" in sub.columns:
                    sub = sub.rename(columns={"Adj Close": "Close"})
                if "Close" not in sub.columns:
                    return None
                if "Volume" not in sub.columns:
                    sub["Volume"] = 0.0
                return sub[["Close", "Volume"]].dropna(subset=["Close"])
            else:
                # Single-ticker download (flat columns)
                col_map = {}
                for c in raw.columns:
                    cl = str(c).lower()
                    if cl == "close":
                        col_map[c] = "Close"
                    elif cl == "adj close":
                        col_map[c] = "Close"
                    elif cl == "volume":
                        col_map[c] = "Volume"
                sub = raw.rename(columns=col_map)
                if "Close" not in sub.columns:
                    return None
                if "Volume" not in sub.columns:
                    sub["Volume"] = 0.0
                return sub[["Close", "Volume"]].dropna(subset=["Close"])
        except Exception as e:
            logger.debug("_extract_series error for %s: %s", ticker, e)
            return None

    def _fetch_vix_brent(self) -> None:
        """Fetch VIX and Brent individually (special chars cause issues in bulk)."""
        for sym, attr in [(VIX_TICKER, "_vix"), (BRENT_TICKER, "_brent_usd")]:
            try:
                series = self._individual_download(sym)
                if series is not None and not series.empty:
                    setattr(self, attr, float(series["Close"].iloc[-1]))
                    logger.debug("%s = %.2f", sym, getattr(self, attr))
            except Exception as e:
                logger.debug("Could not fetch %s: %s", sym, e)

    def _update_position(self, pos: PositionData, series: pd.DataFrame) -> None:
        """Populate all live metrics for a position from its price series."""
        closes = series["Close"].dropna()
        if closes.empty:
            pos.data_status = "N/A"
            return

        current_price_local = float(closes.iloc[-1])
        volume = None
        if "Volume" in series.columns and not series["Volume"].empty:
            try:
                volume = float(series["Volume"].iloc[-1])
            except Exception:
                pass

        # Use currency override if the YF ticker was remapped to a different exchange
        from config import YF_CURRENCY_OVERRIDE
        effective_currency = YF_CURRENCY_OVERRIDE.get(pos.ticker, pos.currency)
        price_eur = self._fx.to_eur(current_price_local, effective_currency)

        # ── Entry price (first run only) ──────────────────────
        entry = get_entry_price(pos.ticker)
        if entry is None:
            # First run — derive shares from allocation
            if price_eur > 0:
                shares = pos.allocation_eur / price_eur
            else:
                shares = 0.0
            save_entry_price(
                pos.ticker, current_price_local, price_eur, shares
            )
            entry = get_entry_price(pos.ticker)

        pos.shares_units      = entry.shares_units if entry else 0.0
        pos.entry_price_local = entry.entry_price_local if entry else current_price_local
        pos.entry_price_eur   = entry.entry_price_eur if entry else price_eur

        # ── Current value ──────────────────────────────────────
        pos.current_price_local = current_price_local
        pos.current_price_eur   = price_eur
        pos.current_value_eur   = (pos.shares_units or 0.0) * price_eur
        pos.volume              = volume
        pos.last_updated        = datetime.now(timezone.utc)

        # ── P&L ────────────────────────────────────────────────
        cost_basis = (pos.shares_units or 0.0) * (pos.entry_price_eur or price_eur)
        pos.pnl_eur = pos.current_value_eur - cost_basis
        if cost_basis > 0:
            pos.pnl_pct = (pos.pnl_eur / cost_basis) * 100
        else:
            pos.pnl_pct = 0.0

        # ── Momentum ───────────────────────────────────────────
        # Day change is only meaningful when markets are actually open today.
        # With daily data, iloc[-1] is the last *trading* day's close — on a
        # weekend or holiday that is Friday, so "day %" would show Friday's
        # move vs Thursday, which is misleading. Force 0.0 when closed.
        market_open = self.is_market_hours()
        pos.day_change_pct   = self._pct_change(closes, 1) if market_open else 0.0
        pos.week_change_pct  = self._pct_change(closes, 5)
        pos.month_change_pct = self._pct_change(closes, 21)

        # ── Data status ────────────────────────────────────────
        pos.data_status = "LIVE" if market_open else "DELAYED"
        pos.error_msg   = None

        # ── Persist snapshot ───────────────────────────────────
        save_price_snapshot(pos.ticker, current_price_local, price_eur, volume)

    @staticmethod
    def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
        """Calculate % change over N periods (trading days)."""
        if len(series) > periods:
            old = series.iloc[-(periods + 1)]
            new = series.iloc[-1]
            if old and old != 0:
                return ((new - old) / abs(old)) * 100
        return None

    # ─────────────────────────────────────────────────────────
    # BACKGROUND REFRESH
    # ─────────────────────────────────────────────────────────

    def start_background_refresh(self) -> None:
        from config import PRICE_REFRESH_INTERVAL
        self.refresh_all()   # Immediate first run

        def _loop():
            while True:
                time.sleep(PRICE_REFRESH_INTERVAL)
                try:
                    self.refresh_all()
                except Exception as e:
                    logger.error("PriceEngine background refresh error: %s", e)

        t = threading.Thread(target=_loop, daemon=True, name="price-refresh")
        t.start()
        logger.info("PriceEngine background refresh started")

    # ─────────────────────────────────────────────────────────
    # BACKTESTING SUPPORT
    # ─────────────────────────────────────────────────────────

    def get_historical_portfolio_values(self, days: int = 30) -> pd.DataFrame:
        """Return a DataFrame of (timestamp, total_value_eur) for the last N days."""
        from config import PORTFOLIO
        rows = []
        tickers = [p["ticker"] for p in PORTFOLIO]
        for ticker in tickers:
            hist = get_price_history(ticker, days=days)
            for h in hist:
                rows.append({
                    "ticker": ticker,
                    "timestamp": h.timestamp,
                    "price_eur": h.price_eur,
                })
        if not rows:
            return pd.DataFrame(columns=["timestamp", "total_value_eur"])

        df = pd.DataFrame(rows)
        # Pivot and sum weighted values
        pivot = df.pivot_table(
            index="timestamp", columns="ticker", values="price_eur", aggfunc="last"
        ).ffill()

        # Multiply by shares (use entry prices as proxy for shares count)
        total = pd.Series(0.0, index=pivot.index)
        for ticker in pivot.columns:
            ep = get_entry_price(ticker)
            if ep and ep.shares_units:
                total += pivot[ticker].fillna(0) * ep.shares_units

        result = pd.DataFrame({"timestamp": total.index, "total_value_eur": total.values})
        return result.sort_values("timestamp").reset_index(drop=True)


# Module-level singleton
_price_engine: Optional[PriceEngine] = None


def get_price_engine() -> PriceEngine:
    global _price_engine
    if _price_engine is None:
        _price_engine = PriceEngine()
    return _price_engine
