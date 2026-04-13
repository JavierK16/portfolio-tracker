"""
price_engine.py — Fetches real-time and historical prices via yfinance.
Calculates P&L, drift, momentum, and stores snapshots in SQLite.

Data source priority:
  1. PRIMARY: yfinance — 15-min cache, 0.5s spacing between requests
  2. FALLBACK: Finnhub free API (FINNHUB_API_KEY from .env)
  3. LAST RESORT: show last known price with STALE badge

Resilience:
  - Never crashes on data source failure
  - Logs every failure to logs/data_errors.log
  - On startup, tests all data sources and reports availability
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf
import pandas as pd
import numpy as np
import requests

from src.database import (
    get_entry_price, save_entry_price, save_price_snapshot,
    get_price_history, get_config_value, set_config_value,
    bulk_save_price_history, get_oldest_price_date, get_price_history_count,
)
from src.fx_engine import get_fx_engine

logger = logging.getLogger(__name__)

# ── Data error logger (separate file) ────────────────────────
_data_err_logger = logging.getLogger("data_errors")
if not _data_err_logger.handlers:
    os.makedirs("logs", exist_ok=True)
    _dh = logging.FileHandler("logs/data_errors.log")
    _dh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _data_err_logger.addHandler(_dh)
    _data_err_logger.setLevel(logging.WARNING)

VIX_TICKER = "^VIX"
BRENT_TICKER = "BZ=F"    # Brent crude futures on yfinance
GOLD_TICKER = "GC=F"     # Gold futures (COMEX) on yfinance
COPPER_TICKER = "HG=F"   # Copper futures (COMEX) on yfinance

# ── Finnhub fallback ─────────────────────────────────────────
FINNHUB_API_KEY: Optional[str] = None

# Try loading from .env file
for env_path in [".env", "../.env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")]:
    try:
        if os.path.isfile(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FINNHUB_API_KEY="):
                        FINNHUB_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if FINNHUB_API_KEY:
                break
    except Exception:
        pass

# Also check env var directly
if not FINNHUB_API_KEY:
    FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

# Rate limiting: minimum seconds between yfinance requests
YF_REQUEST_SPACING = 0.5

# Cache duration for yfinance data (seconds) — 15 minutes
YF_CACHE_DURATION = 900


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
    data_status: str = "PENDING"   # LIVE / DELAYED / STALE / N/A
    error_msg: Optional[str] = None
    data_source: str = "NONE"      # YFINANCE / FINNHUB / CACHE


def _finnhub_quote(ticker: str) -> Optional[Dict]:
    """
    Fetch a quote from Finnhub free API.
    Returns {"c": current_price, "pc": previous_close, ...} or None.
    """
    if not FINNHUB_API_KEY:
        return None
    try:
        # Finnhub uses plain symbols — strip exchange suffixes for US tickers
        symbol = ticker
        # For LSE tickers (.L), Finnhub doesn't support them well — skip
        if "." in ticker and not ticker.endswith(".L"):
            pass  # Try as-is for .DE etc.

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        # Finnhub returns c=0 if ticker not found
        if data and data.get("c", 0) > 0:
            return data
        return None
    except Exception as e:
        _data_err_logger.warning("Finnhub failed for %s: %s", ticker, e)
        return None


class PriceEngine:
    """
    Fetches prices for all portfolio positions.

    Data source priority:
      1. yfinance (primary) — with 0.5s spacing between requests
      2. Finnhub (fallback) — when yfinance fails
      3. Last known price (STALE) — when all sources fail

    Runs in a background thread, refreshing every PRICE_REFRESH_INTERVAL seconds.
    Uses a 15-minute yfinance cache to avoid hammering the API.
    """

    def __init__(self):
        self._positions: Dict[str, PositionData] = {}
        self._portfolio_value_eur: float = 0.0
        self._portfolio_day_pnl_eur: float = 0.0
        self._portfolio_total_pnl_eur: float = 0.0
        self._vix: Optional[float] = None
        self._brent_usd: Optional[float] = None
        self._gold_usd: Optional[float] = None
        self._copper_usd: Optional[float] = None
        # Historical commodity series for prediction models
        self._brent_history: Optional[pd.Series] = None
        self._gold_history: Optional[pd.Series] = None
        self._copper_history: Optional[pd.Series] = None
        self._lock = threading.RLock()
        self._last_refresh: Optional[datetime] = None
        self._fx = get_fx_engine()
        # yfinance cache: {ticker: (DataFrame, timestamp)}
        self._yf_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        # Data source availability (set by startup test)
        self._source_status: Dict[str, str] = {}

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
    # STARTUP DIAGNOSTICS
    # ─────────────────────────────────────────────────────────

    def test_data_sources(self) -> Dict[str, str]:
        """
        Test all data sources on startup and report availability.
        Returns {source_name: status_string}.
        """
        results = {}

        # 1. Test yfinance
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test = yf.Ticker("AAPL").history(period="1d", interval="1d")
            if test is not None and not test.empty:
                results["yfinance"] = "OK"
                logger.info("Data source yfinance: OK")
            else:
                results["yfinance"] = "EMPTY_RESPONSE"
                logger.warning("Data source yfinance: returned empty data")
        except Exception as e:
            results["yfinance"] = f"FAIL: {e}"
            logger.error("Data source yfinance: FAILED — %s", e)
            _data_err_logger.error("Startup test: yfinance FAILED — %s", e)

        # 2. Test Finnhub
        if FINNHUB_API_KEY:
            try:
                url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={FINNHUB_API_KEY}"
                resp = requests.get(url, timeout=8)
                resp.raise_for_status()
                data = resp.json()
                if data and data.get("c", 0) > 0:
                    results["finnhub"] = f"OK (AAPL=${data['c']:.2f})"
                    logger.info("Data source Finnhub: OK (AAPL=$%.2f)", data["c"])
                else:
                    results["finnhub"] = "EMPTY_RESPONSE"
                    logger.warning("Data source Finnhub: empty response")
            except Exception as e:
                results["finnhub"] = f"FAIL: {e}"
                logger.error("Data source Finnhub: FAILED — %s", e)
                _data_err_logger.error("Startup test: Finnhub FAILED — %s", e)
        else:
            results["finnhub"] = "NO_API_KEY"
            logger.info("Data source Finnhub: no API key (set FINNHUB_API_KEY in .env)")

        # 3. Test ECB FX
        try:
            from config import ECB_FX_URL
            resp = requests.get(ECB_FX_URL, timeout=10)
            resp.raise_for_status()
            if "currency" in resp.text.lower():
                results["ecb_fx"] = "OK"
                logger.info("Data source ECB FX: OK")
            else:
                results["ecb_fx"] = "UNEXPECTED_RESPONSE"
        except Exception as e:
            results["ecb_fx"] = f"FAIL: {e}"
            logger.error("Data source ECB FX: FAILED — %s", e)
            _data_err_logger.error("Startup test: ECB FX FAILED — %s", e)

        self._source_status = results
        return results

    def get_source_status(self) -> Dict[str, str]:
        """Return data source availability from last startup test."""
        return dict(self._source_status)

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

        # Download VIX, Brent, Gold, Copper individually (special chars disrupt bulk)
        self._fetch_vix_brent()

        # Check if yfinance cache is still fresh (15-min window)
        now = datetime.now(timezone.utc)
        cache_fresh = (
            self._last_refresh is not None
            and (now - self._last_refresh).total_seconds() < YF_CACHE_DURATION
        )

        # Bulk download (skip if cache is fresh and we have data for all tickers)
        if not cache_fresh or not all(t in self._yf_cache for t in portfolio_tickers):
            raw = self._bulk_download(yf_tickers)
        else:
            raw = pd.DataFrame()  # Will use cache

        # Build per-portfolio-ticker series cache
        series_cache: Dict[str, pd.DataFrame] = {}
        for yf_sym in yf_tickers:
            port_ticker = yf_to_portfolio.get(yf_sym, yf_sym)

            # Try extracting from bulk download
            series = self._extract_series(raw, yf_sym)
            if series is not None and not series.empty:
                # Apply pence → GBP conversion for LSE tickers priced in GBp
                if yf_sym in YF_PENCE_TICKERS:
                    series = series.copy()
                    series["Close"] = series["Close"] / 100.0
                series_cache[port_ticker] = series
                self._yf_cache[port_ticker] = (series, now)
                continue

            # Try yfinance individual download (with rate limiting)
            time.sleep(YF_REQUEST_SPACING)
            fb = self._individual_download(yf_sym)
            if fb is not None and not fb.empty:
                if yf_sym in YF_PENCE_TICKERS:
                    fb = fb.copy()
                    fb["Close"] = fb["Close"] / 100.0
                series_cache[port_ticker] = fb
                self._yf_cache[port_ticker] = (fb, now)
                logger.info("Used individual yfinance fallback for %s (→%s)", port_ticker, yf_sym)
                continue

            # Try Finnhub fallback
            fh = self._try_finnhub_fallback(port_ticker, yf_sym)
            if fh is not None and not fh.empty:
                if yf_sym in YF_PENCE_TICKERS:
                    fh = fh.copy()
                    fh["Close"] = fh["Close"] / 100.0
                series_cache[port_ticker] = fh
                logger.info("Used Finnhub fallback for %s", port_ticker)
                continue

            # Use yfinance cache if available (even if stale)
            cached = self._yf_cache.get(port_ticker)
            if cached is not None:
                series_cache[port_ticker] = cached[0]
                _data_err_logger.warning(
                    "All sources failed for %s — using cached data from %s",
                    port_ticker, cached[1].isoformat()
                )
                continue

            # All sources failed, no cache — will show STALE/N/A
            _data_err_logger.error(
                "ALL data sources failed for %s (yf=%s). No cached data available.",
                port_ticker, yf_sym
            )

        with self._lock:
            total_value = 0.0
            total_pnl   = 0.0
            total_day_pnl = 0.0

            for ticker, pos in self._positions.items():
                try:
                    series = series_cache.get(ticker)
                    if series is None or series.empty:
                        # Check if position already has a price from a previous refresh
                        if pos.current_price_eur is not None and pos.current_price_eur > 0:
                            pos.data_status = "STALE"
                            pos.error_msg = "All data sources failed — showing last known price"
                            pos.data_source = "CACHE"
                            # Still count towards portfolio value
                            if pos.current_value_eur:
                                total_value += pos.current_value_eur
                            if pos.pnl_eur:
                                total_pnl += pos.pnl_eur
                        else:
                            pos.data_status = "N/A"
                            pos.error_msg = "No data from any source"
                            pos.data_source = "NONE"
                        logger.warning("No price data for %s", ticker)
                        continue

                    # Determine data source
                    cached_entry = self._yf_cache.get(ticker)
                    if cached_entry and cached_entry[0] is series:
                        pos.data_source = "YFINANCE"
                    else:
                        pos.data_source = "YFINANCE"  # default; Finnhub will override

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
                    _data_err_logger.error("Position update error for %s: %s", ticker, e)
                    # Keep existing data as STALE rather than wiping it
                    if pos.current_price_eur is not None:
                        pos.data_status = "STALE"
                        pos.error_msg = f"Update failed: {e}"
                        if pos.current_value_eur:
                            total_value += pos.current_value_eur
                        if pos.pnl_eur:
                            total_pnl += pos.pnl_eur
                    else:
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

    def get_gold(self) -> Optional[float]:
        with self._lock:
            return self._gold_usd

    def get_copper(self) -> Optional[float]:
        with self._lock:
            return self._copper_usd

    def get_commodity_histories(self) -> Dict[str, Optional[pd.Series]]:
        """Return historical price series for Brent, Gold, Copper."""
        with self._lock:
            return {
                "brent": self._brent_history.copy() if self._brent_history is not None else None,
                "gold": self._gold_history.copy() if self._gold_history is not None else None,
                "copper": self._copper_history.copy() if self._copper_history is not None else None,
            }

    def last_refresh(self) -> Optional[datetime]:
        with self._lock:
            return self._last_refresh

    def is_market_hours(self) -> bool:
        """Check if any relevant market is open (European or US)."""
        from config import MARKET_HOURS_START, MARKET_HOURS_END
        now_utc = datetime.now(timezone.utc)
        # CET is UTC+1 (CEST = UTC+2 in summer, but we use simple +1 as approximation)
        now_cet = now_utc + timedelta(hours=2)  # CEST for April
        return (
            now_cet.weekday() < 5 and
            MARKET_HOURS_START <= now_cet.hour < MARKET_HOURS_END
        )

    # ─────────────────────────────────────────────────────────
    # FINNHUB FALLBACK
    # ─────────────────────────────────────────────────────────

    def _try_finnhub_fallback(self, port_ticker: str, yf_ticker: str) -> Optional[pd.DataFrame]:
        """
        Try fetching current price from Finnhub when yfinance fails.
        Constructs a minimal DataFrame with just the current Close price
        so it can slot into the same pipeline.
        """
        # Map to Finnhub-compatible symbols
        # Finnhub uses different format for non-US tickers
        finnhub_map = {
            # LSE tickers: not well supported on Finnhub free tier
            "SPOG.L": None, "DFNS.L": None, "NATO.L": None,
            "BA.L": None, "IGLN.L": None,
            # XETRA tickers
            "RHM.DE": None, "VVMX.DE": None,
            # Oslo tickers: Finnhub uses EQNR (NYSE) as proxy
            "EQNR.OL": "EQNR",
            # NYSE/NASDAQ — these work on Finnhub
            "XOM": "XOM", "FCX": "FCX",
            "GDX": "GDX", "VRTX": "VRTX",
            # Remapped
            "SETM.L": "SETM", "BTEC.AS": None,
        }

        fh_sym = finnhub_map.get(port_ticker, port_ticker)
        if fh_sym is None:
            return None

        quote = _finnhub_quote(fh_sym)
        if quote is None:
            return None

        price = quote.get("c", 0)
        if price <= 0:
            return None

        # Build a minimal 2-row DataFrame so momentum calcs have something
        prev_close = quote.get("pc", price)
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        df = pd.DataFrame({
            "Close": [prev_close, price],
            "Volume": [0.0, 0.0],
        }, index=pd.DatetimeIndex([yesterday, now], tz="UTC"))

        logger.info("Finnhub quote for %s (%s): $%.2f", port_ticker, fh_sym, price)
        return df

    # ─────────────────────────────────────────────────────────
    # INTERNAL — YFINANCE
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
                )
            return raw
        except Exception as e:
            logger.error("Bulk yfinance download failed: %s", e)
            _data_err_logger.error("Bulk yfinance download failed: %s", e)
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
            logger.warning("Individual yfinance download failed for %s: %s", ticker, e)
            _data_err_logger.warning("Individual yfinance failed for %s: %s", ticker, e)
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
                tickers_available = raw.columns.get_level_values(1).unique()
                if ticker not in tickers_available:
                    return None
                sub = raw.xs(ticker, axis=1, level=1)
                if "Close" not in sub.columns and "Adj Close" in sub.columns:
                    sub = sub.rename(columns={"Adj Close": "Close"})
                if "Close" not in sub.columns:
                    return None
                if "Volume" not in sub.columns:
                    sub["Volume"] = 0.0
                return sub[["Close", "Volume"]].dropna(subset=["Close"])
            else:
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
        """Fetch VIX, Brent, Gold spot, and Copper individually with rate limiting."""
        for sym, attr in [
            (VIX_TICKER, "_vix"), (BRENT_TICKER, "_brent_usd"),
            (GOLD_TICKER, "_gold_usd"), (COPPER_TICKER, "_copper_usd"),
        ]:
            try:
                time.sleep(YF_REQUEST_SPACING)  # Rate limiting
                series = self._individual_download(sym)
                if series is not None and not series.empty:
                    setattr(self, attr, float(series["Close"].iloc[-1]))
                    # Store historical series for commodity-correlation model
                    hist_attr = attr.replace("_usd", "_history").replace("_vix", "_vix_history")
                    if hasattr(self, hist_attr):
                        setattr(self, hist_attr, series["Close"].dropna())
                    logger.debug("%s = %.2f", sym, getattr(self, attr))
                else:
                    _data_err_logger.warning("No data for %s from yfinance", sym)
            except Exception as e:
                logger.debug("Could not fetch %s: %s", sym, e)
                _data_err_logger.warning("Failed to fetch %s: %s", sym, e)

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
        market_open = self.is_market_hours()
        pos.day_change_pct   = self._pct_change(closes, 1) if market_open else 0.0
        pos.week_change_pct  = self._pct_change(closes, 5)
        pos.month_change_pct = self._pct_change(closes, 21)

        # ── Data status ────────────────────────────────────────
        # Check if the latest data point is from today
        if closes.index[-1].tzinfo is None:
            last_ts = closes.index[-1].tz_localize("UTC")
        else:
            last_ts = closes.index[-1]

        age_hours = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600

        if market_open and age_hours < 24:
            pos.data_status = "LIVE"
        elif age_hours < 48:
            pos.data_status = "DELAYED"
        else:
            pos.data_status = "DELAYED"

        pos.error_msg = None

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

        # Run startup diagnostics
        self.test_data_sources()

        self.refresh_all()   # Immediate first run

        def _loop():
            while True:
                time.sleep(PRICE_REFRESH_INTERVAL)
                try:
                    self.refresh_all()
                except Exception as e:
                    logger.error("PriceEngine background refresh error: %s", e)
                    _data_err_logger.error("Background refresh error: %s", e)

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
        pivot = df.pivot_table(
            index="timestamp", columns="ticker", values="price_eur", aggfunc="last"
        ).ffill()

        total = pd.Series(0.0, index=pivot.index)
        for ticker in pivot.columns:
            ep = get_entry_price(ticker)
            if ep and ep.shares_units:
                total += pivot[ticker].fillna(0) * ep.shares_units

        result = pd.DataFrame({"timestamp": total.index, "total_value_eur": total.values})
        return result.sort_values("timestamp").reset_index(drop=True)


    # ─────────────────────────────────────────────────────────
    # HISTORICAL BACKFILL
    # ─────────────────────────────────────────────────────────

    def backfill_history(self, years: int = 2) -> None:
        """
        Download up to `years` of daily history from yfinance for every
        portfolio ticker and store in price_history. Only fetches data older
        than what we already have. Runs once on first startup (flag in DB).
        """
        from config import (
            PORTFOLIO, YF_TICKER_MAP, YF_CURRENCY_OVERRIDE, YF_PENCE_TICKERS,
        )

        flag = get_config_value("history_backfill_done")
        if flag == "true":
            logger.info("History backfill already completed — skipping")
            return

        logger.info("Starting %d-year price history backfill...", years)

        if self._fx.is_stale():
            self._fx.refresh()

        for pos_cfg in PORTFOLIO:
            ticker = pos_cfg["ticker"]
            yf_sym = YF_TICKER_MAP.get(ticker, ticker)

            # Check how much history we already have
            existing_count = get_price_history_count(ticker)
            oldest = get_oldest_price_date(ticker)

            if existing_count >= 400:
                logger.debug("Backfill skip %s — already has %d rows", ticker, existing_count)
                continue

            try:
                time.sleep(YF_REQUEST_SPACING)  # Rate limiting
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hist = yf.Ticker(yf_sym).history(
                        period=f"{years}y", interval="1d", auto_adjust=True,
                    )

                if hist is None or hist.empty:
                    logger.warning("Backfill: no history for %s (%s)", ticker, yf_sym)
                    _data_err_logger.warning("Backfill: no history for %s (%s)", ticker, yf_sym)
                    continue

                col_map = {}
                for c in hist.columns:
                    cl = c.lower()
                    if "close" in cl:
                        col_map[c] = "Close"
                    elif "volume" in cl:
                        col_map[c] = "Volume"
                hist = hist.rename(columns=col_map)
                if "Close" not in hist.columns:
                    continue
                if "Volume" not in hist.columns:
                    hist["Volume"] = 0.0

                hist = hist[["Close", "Volume"]].dropna(subset=["Close"])

                # Pence conversion
                if yf_sym in YF_PENCE_TICKERS or ticker in YF_PENCE_TICKERS:
                    hist["Close"] = hist["Close"] / 100.0

                # Convert to EUR
                effective_ccy = YF_CURRENCY_OVERRIDE.get(ticker, pos_cfg["currency"])

                # Only keep rows older than our oldest existing record
                if oldest is not None:
                    hist = hist[hist.index < pd.Timestamp(oldest, tz="UTC")]
                    if hist.empty:
                        logger.debug("Backfill skip %s — no older data to add", ticker)
                        continue

                records = []
                for ts, row in hist.iterrows():
                    price_local = float(row["Close"])
                    price_eur = self._fx.to_eur(price_local, effective_ccy)
                    vol = float(row["Volume"]) if pd.notna(row["Volume"]) else None
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    records.append({
                        "ticker": ticker,
                        "timestamp": ts.to_pydatetime(),
                        "price_local": price_local,
                        "price_eur": price_eur,
                        "volume": vol,
                    })

                if records:
                    count = bulk_save_price_history(records)
                    logger.info("Backfilled %d daily records for %s (%s)", count, ticker, yf_sym)

            except Exception as e:
                logger.error("Backfill failed for %s: %s", ticker, e)
                _data_err_logger.error("Backfill failed for %s: %s", ticker, e)

        set_config_value("history_backfill_done", "true")
        logger.info("History backfill complete for all tickers")


# Module-level singleton
_price_engine: Optional[PriceEngine] = None


def get_price_engine() -> PriceEngine:
    global _price_engine
    if _price_engine is None:
        _price_engine = PriceEngine()
    return _price_engine
