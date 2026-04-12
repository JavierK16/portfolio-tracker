"""
prediction_accuracy.py — Tracks prediction accuracy by comparing matured predictions
against actual prices. Runs hourly in a background thread.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from src.database import (
    get_matured_predictions, update_prediction_actuals,
    get_prediction_accuracy_metrics,
)

logger = logging.getLogger(__name__)


class PredictionAccuracyTracker:
    """Checks matured predictions, fills in actuals, computes accuracy."""

    def __init__(self):
        self._running = False
        self._last_check: Optional[datetime] = None

    def check_matured(self) -> int:
        """Find predictions past target_date, fetch actual prices, update DB."""
        from src.price_engine import get_price_engine

        pe = get_price_engine()
        positions = {p.ticker: p for p in pe.get_all_positions()}

        matured = get_matured_predictions()
        updated = 0

        for pred in matured:
            try:
                if pred.level == "POSITION" and pred.ticker:
                    pos = positions.get(pred.ticker)
                    if not pos or pos.current_price_eur is None:
                        continue
                    actual_price = pos.current_price_eur
                elif pred.level == "SECTOR" and pred.sector:
                    sector_positions = [p for p in positions.values() if p.sector == pred.sector]
                    actual_price = sum(p.current_value_eur or 0 for p in sector_positions)
                    if actual_price == 0:
                        continue
                elif pred.level == "PORTFOLIO":
                    actual_price = pe.get_portfolio_value() or 0
                    if actual_price == 0:
                        continue
                else:
                    continue

                current = pred.current_price_eur or 0
                if current == 0:
                    continue

                actual_change_pct = ((actual_price / current) - 1) * 100

                # Direction check
                pred_dir = pred.direction or "FLAT"
                if actual_change_pct > 1:
                    actual_dir = "UP"
                elif actual_change_pct < -1:
                    actual_dir = "DOWN"
                else:
                    actual_dir = "FLAT"
                was_correct = (pred_dir == actual_dir)

                # CI80 check
                was_in_ci80 = (
                    pred.ci_80_lower is not None
                    and pred.ci_80_upper is not None
                    and pred.ci_80_lower <= actual_price <= pred.ci_80_upper
                )

                update_prediction_actuals(
                    pred.id, actual_price, actual_change_pct,
                    was_correct, was_in_ci80,
                )
                updated += 1

            except Exception as e:
                logger.error("Error checking prediction %d: %s", pred.id, e)

        if updated:
            logger.info("Updated %d matured predictions with actuals", updated)

        self._last_check = datetime.now(timezone.utc)
        return updated

    def get_metrics(self, days: int = 30) -> dict:
        return get_prediction_accuracy_metrics(days)

    def last_check_time(self) -> Optional[datetime]:
        return self._last_check

    def start_background_checks(self) -> None:
        if self._running:
            return
        self._running = True

        def _loop():
            time.sleep(60)  # Wait for system to stabilize
            while self._running:
                try:
                    self.check_matured()
                except Exception as e:
                    logger.error("Accuracy check error: %s", e)
                time.sleep(3600)  # Every hour

        t = threading.Thread(target=_loop, daemon=True, name="accuracy-tracker")
        t.start()
        logger.info("Prediction accuracy tracker started (1-hour interval)")


_instance: Optional[PredictionAccuracyTracker] = None
_lock = threading.Lock()


def get_accuracy_tracker() -> PredictionAccuracyTracker:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PredictionAccuracyTracker()
    return _instance
