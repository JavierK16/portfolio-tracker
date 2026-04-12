"""
fx_engine.py — Fetches live EUR FX rates from the ECB XML daily feed.
Provides EUR conversion helpers for all portfolio currencies.
"""

import logging
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# All currencies used in the portfolio
REQUIRED_CURRENCIES = {"USD", "GBP", "NOK", "EUR"}

# Fallback rates (EUR-based) in case ECB feed is unavailable
FALLBACK_RATES: Dict[str, float] = {
    "EUR": 1.0,
    "USD": 1.08,
    "GBP": 0.86,
    "NOK": 11.60,
    "CHF": 0.96,
    "JPY": 164.0,
}


class FXEngine:
    """Thread-safe ECB FX rate provider. Refreshes every 4 hours."""

    def __init__(self):
        self._rates: Dict[str, float] = dict(FALLBACK_RATES)
        self._last_updated: Optional[datetime] = None
        self._lock = threading.Lock()
        self._stale = True

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def refresh(self) -> bool:
        """Fetch latest rates from ECB. Returns True on success."""
        from config import ECB_FX_URL
        try:
            resp = requests.get(ECB_FX_URL, timeout=10)
            resp.raise_for_status()
            new_rates = self._parse_ecb_xml(resp.text)
            if new_rates:
                with self._lock:
                    self._rates = new_rates
                    self._rates["EUR"] = 1.0
                    self._last_updated = datetime.now(timezone.utc)
                    self._stale = False
                logger.info("ECB FX rates updated: %s", self._rates)
                return True
            else:
                logger.warning("ECB XML parsed but no rates found — using fallback")
                return False
        except Exception as e:
            logger.error("ECB FX fetch failed: %s — using fallback rates", e)
            self._stale = True
            return False

    def to_eur(self, amount: float, currency: str) -> float:
        """Convert amount in local currency to EUR."""
        if currency == "EUR":
            return amount
        rate = self.get_rate(currency)
        if rate and rate > 0:
            return amount / rate
        logger.warning("No FX rate for %s — using 1.0", currency)
        return amount

    def get_rate(self, currency: str) -> Optional[float]:
        """Return how many units of currency = 1 EUR."""
        with self._lock:
            return self._rates.get(currency.upper())

    def is_stale(self) -> bool:
        with self._lock:
            return self._stale

    def last_updated(self) -> Optional[datetime]:
        with self._lock:
            return self._last_updated

    def all_rates(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._rates)

    # ─────────────────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_ecb_xml(xml_text: str) -> Dict[str, float]:
        """Parse ECB eurofxref XML into {currency: rate_vs_eur} dict."""
        rates: Dict[str, float] = {"EUR": 1.0}
        try:
            root = ET.fromstring(xml_text)
            ns = {"gesmes": "http://www.gesmes.org/xml/2002-08-01",
                  "ecb":    "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
            # Walk all Cube elements looking for currency/rate attributes
            for cube in root.iter():
                currency = cube.attrib.get("currency")
                rate_str = cube.attrib.get("rate")
                if currency and rate_str:
                    try:
                        rates[currency.upper()] = float(rate_str)
                    except ValueError:
                        pass
        except ET.ParseError as e:
            logger.error("ECB XML parse error: %s", e)
        return rates

    # ─────────────────────────────────────────────────────────
    # BACKGROUND REFRESH
    # ─────────────────────────────────────────────────────────

    def start_background_refresh(self, interval_seconds: int = 14_400) -> None:
        """Start a daemon thread that refreshes FX rates periodically."""
        self.refresh()  # Immediate first fetch

        def _loop():
            while True:
                time.sleep(interval_seconds)
                self.refresh()

        t = threading.Thread(target=_loop, daemon=True, name="fx-refresh")
        t.start()
        logger.info("FX background refresh started (interval=%ds)", interval_seconds)


# Module-level singleton
_fx_engine: Optional[FXEngine] = None


def get_fx_engine() -> FXEngine:
    global _fx_engine
    if _fx_engine is None:
        _fx_engine = FXEngine()
    return _fx_engine
