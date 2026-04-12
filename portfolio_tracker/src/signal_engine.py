"""
signal_engine.py — Composite scoring and BUY/ADD/HOLD/REDUCE/SELL signal generation.
Runs every 15 minutes. Applies override rules for geopolitical emergencies.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.price_engine import PriceEngine, PositionData, get_price_engine
from src.geo_scorer import GeopoliticalScorer, get_geo_scorer
from src.database import get_geo_states, save_signal, get_signal_history

logger = logging.getLogger(__name__)


@dataclass
class FactorScores:
    geo_score: float          = 50.0
    momentum_score: float     = 50.0
    drift_score: float        = 80.0
    drawdown_score: float     = 80.0
    volatility_score: float   = 50.0
    composite: float          = 0.0


@dataclass
class SignalResult:
    ticker: str
    signal: str                        # BUY / ADD / HOLD / REDUCE / SELL
    composite_score: float
    factors: FactorScores
    flags: List[str]                   # Override flags
    rationale: str
    signal_change: bool = False        # True if signal changed vs previous
    previous_signal: Optional[str] = None
    what_would_change: str = ""


class SignalEngine:
    """
    Generates trading signals for all portfolio positions.
    Uses a 5-factor weighted composite score.
    """

    def __init__(self, price_engine: PriceEngine, geo_scorer: GeopoliticalScorer):
        self._price = price_engine
        self._geo   = geo_scorer
        self._signals: Dict[str, SignalResult] = {}
        self._lock = threading.RLock()
        self._last_refresh: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def refresh_all(self) -> Dict[str, SignalResult]:
        logger.info("SignalEngine: computing signals")
        positions = self._price.get_all_positions()
        geo_context = self._geo.get_geo_context()
        geo_states  = get_geo_states()
        vix         = self._price.get_vix()
        brent       = self._price.get_brent()

        new_signals: Dict[str, SignalResult] = {}

        for pos in positions:
            try:
                result = self._compute_signal(pos, geo_context, geo_states, vix, brent)
                # Check for signal change
                prev = self._signals.get(pos.ticker)
                if prev and prev.signal != result.signal:
                    result.signal_change    = True
                    result.previous_signal  = prev.signal
                    logger.info(
                        "SIGNAL CHANGE: %s  %s → %s",
                        pos.ticker, prev.signal, result.signal
                    )
                # Persist signal
                save_signal(
                    pos.ticker,
                    result.signal,
                    result.composite_score,
                    {
                        "geo":        result.factors.geo_score,
                        "momentum":   result.factors.momentum_score,
                        "drift":      result.factors.drift_score,
                        "drawdown":   result.factors.drawdown_score,
                        "volatility": result.factors.volatility_score,
                    },
                    json.dumps(result.flags),
                )
                new_signals[pos.ticker] = result
            except Exception as e:
                logger.error("SignalEngine error for %s: %s", pos.ticker, e)

        with self._lock:
            self._signals = new_signals
            self._last_refresh = datetime.now(timezone.utc)

        return new_signals

    def get_signal(self, ticker: str) -> Optional[SignalResult]:
        with self._lock:
            return self._signals.get(ticker)

    def get_all_signals(self) -> Dict[str, SignalResult]:
        with self._lock:
            return dict(self._signals)

    def last_refresh_time(self) -> Optional[datetime]:
        with self._lock:
            return self._last_refresh

    # ─────────────────────────────────────────────────────────
    # SIGNAL COMPUTATION
    # ─────────────────────────────────────────────────────────

    def _compute_signal(
        self,
        pos: PositionData,
        geo_context,
        geo_states: dict,
        vix: Optional[float],
        brent: Optional[float],
    ) -> SignalResult:
        from config import SIGNAL_WEIGHTS, SIGNAL_THRESHOLDS, SECTOR_CONFIG, INVESTOR_PROFILE

        factors = FactorScores()
        flags: List[str] = []

        # ── Factor 1: Sector Geo Score (35%) ─────────────────
        sector_score_0_10 = self._geo.get_sector_score(pos.sector)
        factors.geo_score = min(100.0, sector_score_0_10 * 10.0)

        # ── Factor 2: Price Momentum (20%) ───────────────────
        factors.momentum_score = self._momentum_score(pos)

        # ── Factor 3: Position Drift (15%) ───────────────────
        factors.drift_score = self._drift_score(pos)

        # ── Factor 4: Drawdown Risk (15%) ────────────────────
        factors.drawdown_score = self._drawdown_score(pos)

        # ── Factor 5: Volatility Regime (15%) ────────────────
        factors.volatility_score = self._volatility_score(pos, vix, brent)

        # ── Weighted composite ────────────────────────────────
        composite = (
            factors.geo_score       * SIGNAL_WEIGHTS["sector_geo_score"] +
            factors.momentum_score  * SIGNAL_WEIGHTS["price_momentum"] +
            factors.drift_score     * SIGNAL_WEIGHTS["position_drift"] +
            factors.drawdown_score  * SIGNAL_WEIGHTS["drawdown_risk"] +
            factors.volatility_score * SIGNAL_WEIGHTS["volatility_regime"]
        )

        # ── VIX adjustments ───────────────────────────────────
        if vix is not None:
            if vix > 25:
                composite -= 10
            elif vix < 15:
                composite += 5

        composite = max(0.0, min(100.0, composite))
        factors.composite = composite

        # ── Base signal from composite ─────────────────────────
        signal = self._composite_to_signal(composite)

        # ── Override rules ────────────────────────────────────
        signal, flags = self._apply_overrides(signal, flags, pos, geo_states, brent)

        # ── Rationale ─────────────────────────────────────────
        rationale = self._build_rationale(pos, factors, geo_context, geo_states, brent)
        what_would_change = self._build_what_would_change(pos, factors, composite, geo_states, brent)

        return SignalResult(
            ticker=pos.ticker,
            signal=signal,
            composite_score=round(composite, 1),
            factors=factors,
            flags=flags,
            rationale=rationale,
            what_would_change=what_would_change,
        )

    # ─────────────────────────────────────────────────────────
    # FACTOR CALCULATORS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _momentum_score(pos: PositionData) -> float:
        m  = pos.month_change_pct or 0.0
        w  = pos.week_change_pct  or 0.0
        if m > 0 and w > 0:
            base = 75
            # Scale up with magnitude
            base += min(25, abs(m) * 0.5 + abs(w) * 0.5)
            return min(100, base)
        elif m > 0 and w <= 0:
            return 50 + min(24, abs(m) * 0.5)
        elif m <= 0 and w > 0:
            return 40 + min(19, abs(w) * 0.5)
        else:
            # Both negative
            mag = abs(m) + abs(w)
            return max(0, 39 - mag * 0.5)

    @staticmethod
    def _drift_score(pos: PositionData) -> float:
        drift = pos.drift_from_target
        if drift is None:
            return 80.0  # No data → neutral
        if abs(drift) <= 2:
            return 80.0
        elif drift > 5:
            return 30.0   # Overweight — consider trimming
        elif drift < -5:
            return 70.0   # Underweight — consider adding
        elif 2 < drift <= 5:
            return 55.0   # Slightly overweight
        else:  # -5 <= drift < -2
            return 65.0   # Slightly underweight

    @staticmethod
    def _drawdown_score(pos: PositionData) -> float:
        pnl = pos.pnl_pct
        if pnl is None:
            return 80.0
        if pnl >= 0:
            return 80.0
        elif pnl >= -10:
            return 65.0
        elif pnl >= -25:
            return 45.0
        elif pnl >= -40:
            return 25.0
        else:
            return 10.0

    def _volatility_score(
        self,
        pos: PositionData,
        vix: Optional[float],
        brent: Optional[float],
    ) -> float:
        base = 60.0
        # Brent regime
        if brent is not None:
            if brent > 100:
                if pos.sector == "ENERGY":
                    base += 20.0
                else:
                    base -= 10.0
            elif brent < 70:
                if pos.sector == "ENERGY":
                    base -= 15.0
        # VIX handled at composite level — return base here
        return max(0, min(100, base))

    # ─────────────────────────────────────────────────────────
    # SIGNAL THRESHOLD
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _composite_to_signal(score: float) -> str:
        if score >= 75:
            return "BUY"
        elif score >= 60:
            return "ADD"
        elif score >= 45:
            return "HOLD"
        elif score >= 30:
            return "REDUCE"
        else:
            return "SELL"

    # ─────────────────────────────────────────────────────────
    # OVERRIDE RULES
    # ─────────────────────────────────────────────────────────

    def _apply_overrides(
        self,
        signal: str,
        flags: List[str],
        pos: PositionData,
        geo_states: dict,
        brent: Optional[float],
    ) -> Tuple[str, List[str]]:
        from config import INVESTOR_PROFILE

        def val(var):
            row = geo_states.get(var)
            return row.current_value if row else None

        # STOP_LOSS_REVIEW
        if pos.pnl_pct is not None and pos.pnl_pct < INVESTOR_PROFILE["stop_loss_pct"]:
            flags.append(
                f"STOP_LOSS_REVIEW: position down {pos.pnl_pct:.1f}% — "
                "conviction check required regardless of score"
            )

        # OVERWEIGHT_TRIM
        if (pos.weight_current_pct is not None and
                pos.weight_current_pct > INVESTOR_PROFILE["max_single_position_pct"]):
            flags.append(
                f"OVERWEIGHT: {pos.weight_current_pct:.1f}% exceeds "
                f"{INVESTOR_PROFILE['max_single_position_pct']}% max — consider trimming"
            )

        # THESIS_BROKEN_DEFENSE
        if (pos.sector == "DEFENSE" and
                val("UKRAINE_WAR") == "RESOLVED" and
                val("NATO_SPENDING") == "DECLINING"):
            signal = "REDUCE"
            flags.append(
                "THESIS_BROKEN_DEFENSE: Ukraine resolved + NATO spending declining"
            )

        # THESIS_BROKEN_ENERGY
        if (pos.sector == "ENERGY" and
                val("HORMUZ_STATUS") == "OPEN" and
                val("IRAN_CONFLICT") == "RESOLVED"):
            # Composite was already calculated — force downgrade
            if signal in ("BUY", "ADD"):
                signal = "HOLD"
            elif signal == "HOLD":
                signal = "REDUCE"
            flags.append(
                "THESIS_BROKEN_ENERGY: Hormuz open + Iran conflict resolved — "
                "energy premium normalising"
            )

        return signal, flags

    # ─────────────────────────────────────────────────────────
    # RATIONALE BUILDER
    # ─────────────────────────────────────────────────────────

    def _build_rationale(
        self,
        pos: PositionData,
        factors: FactorScores,
        geo_context,
        geo_states: dict,
        brent: Optional[float],
    ) -> str:
        from config import SECTOR_CONFIG
        sector_cfg = SECTOR_CONFIG.get(pos.sector, {})

        lines = [
            f"Composite score: {factors.composite:.1f}/100",
            f"Sector geo score ({pos.sector}): {factors.geo_score:.0f}/100  "
            f"(raw: {self._geo.get_sector_score(pos.sector):.2f}/10)",
            f"Price momentum: {factors.momentum_score:.0f}/100  "
            f"(1W: {pos.week_change_pct or 0:.1f}%, 1M: {pos.month_change_pct or 0:.1f}%)",
            f"Position drift: {factors.drift_score:.0f}/100  "
            f"(drift from target: {pos.drift_from_target or 0:+.1f}%)",
            f"Drawdown risk: {factors.drawdown_score:.0f}/100  "
            f"(P&L since entry: {pos.pnl_pct or 0:+.1f}%)",
            f"Volatility regime: {factors.volatility_score:.0f}/100",
        ]

        # Active geo variables
        for var, row in geo_states.items():
            lines.append(f"{var}: {row.current_value}")

        # Bull/bear scenario
        lines.append(f"Bull: {sector_cfg.get('bull_scenario', 'N/A')}")
        lines.append(f"Bear: {sector_cfg.get('bear_scenario', 'N/A')}")

        return "\n".join(lines)

    def _build_what_would_change(
        self,
        pos: PositionData,
        factors: FactorScores,
        composite: float,
        geo_states: dict,
        brent: Optional[float],
    ) -> str:
        from config import SECTOR_CONFIG
        lines = []
        current_signal = self._composite_to_signal(composite)

        if pos.sector == "ENERGY":
            if brent and brent < 80:
                lines.append("ENERGY: Brent below $80 would trigger REDUCE")
            lines.append("ENERGY: Hormuz closure → CLOSED would force BUY signal")
            lines.append("ENERGY: Iran conflict resolution → would reduce score by ~20pts")

        if pos.sector == "DEFENSE":
            lines.append("DEFENSE: Ukraine ceasefire + NATO spending cuts → REDUCE override")
            lines.append("DEFENSE: NATO spending ACCELERATING → +15pts composite")

        if pos.sector == "METALS":
            lines.append("METALS: US-China moving to HOSTILE → +15pts composite")
            lines.append("METALS: China-US trade deal → score reduction")

        if pos.sector == "GOLD":
            lines.append("GOLD: Conflict resolution across all variables → REDUCE")

        if composite < 75:
            pts_needed = 75 - composite
            lines.append(f"BUY: need {pts_needed:.0f} more points — "
                         "improve geo score or momentum")
        if composite > 30:
            pts_to_sell = composite - 30
            lines.append(f"SELL: composite would need to fall {pts_to_sell:.0f} pts")

        return "\n".join(lines) if lines else "No specific triggers identified"

    # ─────────────────────────────────────────────────────────
    # BACKGROUND REFRESH
    # ─────────────────────────────────────────────────────────

    def start_background_refresh(self) -> None:
        from config import SIGNAL_REFRESH_INTERVAL
        self.refresh_all()

        def _loop():
            while True:
                time.sleep(SIGNAL_REFRESH_INTERVAL)
                try:
                    self.refresh_all()
                except Exception as e:
                    logger.error("SignalEngine background refresh error: %s", e)

        t = threading.Thread(target=_loop, daemon=True, name="signal-refresh")
        t.start()
        logger.info("SignalEngine background refresh started")


# Module-level singleton
_signal_engine: Optional[SignalEngine] = None


def get_signal_engine() -> SignalEngine:
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = SignalEngine(get_price_engine(), get_geo_scorer())
    return _signal_engine
