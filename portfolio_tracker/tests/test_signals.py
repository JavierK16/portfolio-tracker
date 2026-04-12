"""
test_signals.py — Unit tests for the signal engine scoring logic.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.signal_engine import SignalEngine
from src.price_engine import PositionData


def _make_pos(**kwargs) -> PositionData:
    defaults = dict(
        ticker="TEST",
        name="Test Position",
        sector="ENERGY",
        instrument_type="STOCK",
        currency="USD",
        allocation_eur=10_000,
        target_pct=10.0,
        tranche=1,
    )
    defaults.update(kwargs)
    return PositionData(**defaults)


# ─────────────────────────────────────────────────────────────
# MOMENTUM SCORE
# ─────────────────────────────────────────────────────────────

class TestMomentumScore:
    def test_both_positive_gives_high_score(self):
        pos = _make_pos(month_change_pct=5.0, week_change_pct=2.0)
        score = SignalEngine._momentum_score(pos)
        assert score >= 75

    def test_month_pos_week_neg_midrange(self):
        pos = _make_pos(month_change_pct=5.0, week_change_pct=-1.0)
        score = SignalEngine._momentum_score(pos)
        assert 50 <= score <= 74

    def test_month_neg_week_pos_reversing(self):
        pos = _make_pos(month_change_pct=-3.0, week_change_pct=1.5)
        score = SignalEngine._momentum_score(pos)
        assert 40 <= score <= 59

    def test_both_negative_low_score(self):
        pos = _make_pos(month_change_pct=-8.0, week_change_pct=-3.0)
        score = SignalEngine._momentum_score(pos)
        assert score <= 39

    def test_none_values_handled(self):
        pos = _make_pos(month_change_pct=None, week_change_pct=None)
        score = SignalEngine._momentum_score(pos)
        assert 0 <= score <= 100


# ─────────────────────────────────────────────────────────────
# DRIFT SCORE
# ─────────────────────────────────────────────────────────────

class TestDriftScore:
    def test_on_target_high_score(self):
        pos = _make_pos(drift_from_target=0.5)
        score = SignalEngine._drift_score(pos)
        assert score == 80

    def test_large_overweight_low_score(self):
        pos = _make_pos(drift_from_target=6.0)
        score = SignalEngine._drift_score(pos)
        assert score == 30

    def test_large_underweight_medium_high(self):
        pos = _make_pos(drift_from_target=-6.0)
        score = SignalEngine._drift_score(pos)
        assert score == 70

    def test_none_drift_neutral(self):
        pos = _make_pos(drift_from_target=None)
        score = SignalEngine._drift_score(pos)
        assert score == 80


# ─────────────────────────────────────────────────────────────
# DRAWDOWN SCORE
# ─────────────────────────────────────────────────────────────

class TestDrawdownScore:
    def test_positive_pnl_high_score(self):
        pos = _make_pos(pnl_pct=15.0)
        score = SignalEngine._drawdown_score(pos)
        assert score == 80

    def test_minor_loss(self):
        pos = _make_pos(pnl_pct=-5.0)
        score = SignalEngine._drawdown_score(pos)
        assert score == 65

    def test_moderate_drawdown(self):
        pos = _make_pos(pnl_pct=-20.0)
        score = SignalEngine._drawdown_score(pos)
        assert score == 45

    def test_severe_drawdown(self):
        pos = _make_pos(pnl_pct=-35.0)
        score = SignalEngine._drawdown_score(pos)
        assert score == 25

    def test_stop_loss_territory(self):
        pos = _make_pos(pnl_pct=-45.0)
        score = SignalEngine._drawdown_score(pos)
        assert score == 10

    def test_none_pnl_neutral(self):
        pos = _make_pos(pnl_pct=None)
        score = SignalEngine._drawdown_score(pos)
        assert score == 80


# ─────────────────────────────────────────────────────────────
# COMPOSITE → SIGNAL THRESHOLDS
# ─────────────────────────────────────────────────────────────

class TestSignalThresholds:
    def test_buy_at_75(self):
        assert SignalEngine._composite_to_signal(75) == "BUY"

    def test_buy_at_90(self):
        assert SignalEngine._composite_to_signal(90) == "BUY"

    def test_add_at_60(self):
        assert SignalEngine._composite_to_signal(60) == "ADD"

    def test_add_at_74(self):
        assert SignalEngine._composite_to_signal(74) == "ADD"

    def test_hold_at_45(self):
        assert SignalEngine._composite_to_signal(45) == "HOLD"

    def test_hold_at_59(self):
        assert SignalEngine._composite_to_signal(59) == "HOLD"

    def test_reduce_at_30(self):
        assert SignalEngine._composite_to_signal(30) == "REDUCE"

    def test_reduce_at_44(self):
        assert SignalEngine._composite_to_signal(44) == "REDUCE"

    def test_sell_below_30(self):
        assert SignalEngine._composite_to_signal(29) == "SELL"

    def test_sell_at_zero(self):
        assert SignalEngine._composite_to_signal(0) == "SELL"


# ─────────────────────────────────────────────────────────────
# OVERRIDE RULES (using mocked geo states)
# ─────────────────────────────────────────────────────────────

class FakeGeoRow:
    def __init__(self, value):
        self.current_value = value


class TestOverrideRules:
    def _engine(self):
        """Create a SignalEngine with mocked dependencies."""
        from unittest.mock import MagicMock
        pe = MagicMock()
        gs = MagicMock()
        gs.get_sector_score.return_value = 7.0
        return SignalEngine(pe, gs)

    def test_stop_loss_review_flag(self):
        engine = self._engine()
        pos = _make_pos(ticker="XOM", pnl_pct=-45.0, weight_current_pct=10.0,
                        sector="ENERGY")
        geo_states = {
            "HORMUZ_STATUS": FakeGeoRow("OPEN"),
            "IRAN_CONFLICT": FakeGeoRow("ACTIVE"),
        }
        signal, flags = engine._apply_overrides("HOLD", [], pos, geo_states, brent=80.0)
        flag_text = " ".join(flags)
        assert "STOP_LOSS_REVIEW" in flag_text

    def test_overweight_flag(self):
        engine = self._engine()
        pos = _make_pos(ticker="XOM", pnl_pct=5.0, weight_current_pct=22.0,
                        sector="ENERGY")
        geo_states = {
            "HORMUZ_STATUS": FakeGeoRow("OPEN"),
            "IRAN_CONFLICT": FakeGeoRow("ACTIVE"),
        }
        signal, flags = engine._apply_overrides("BUY", [], pos, geo_states, brent=80.0)
        flag_text = " ".join(flags)
        assert "OVERWEIGHT" in flag_text

    def test_thesis_broken_defense(self):
        engine = self._engine()
        pos = _make_pos(ticker="RHM.DE", pnl_pct=10.0, weight_current_pct=7.0,
                        sector="DEFENSE")
        geo_states = {
            "UKRAINE_WAR":  FakeGeoRow("RESOLVED"),
            "NATO_SPENDING": FakeGeoRow("DECLINING"),
        }
        signal, flags = engine._apply_overrides("BUY", [], pos, geo_states, brent=80.0)
        assert signal == "REDUCE"
        assert any("THESIS_BROKEN_DEFENSE" in f for f in flags)

    def test_thesis_broken_energy_downgrades(self):
        engine = self._engine()
        pos = _make_pos(ticker="XOM", pnl_pct=5.0, weight_current_pct=6.0,
                        sector="ENERGY")
        geo_states = {
            "HORMUZ_STATUS": FakeGeoRow("OPEN"),
            "IRAN_CONFLICT": FakeGeoRow("RESOLVED"),
        }
        # Signal BUY should be downgraded
        signal, flags = engine._apply_overrides("BUY", [], pos, geo_states, brent=80.0)
        assert signal in ("HOLD", "REDUCE")
        assert any("THESIS_BROKEN_ENERGY" in f for f in flags)

    def test_no_override_when_clean(self):
        engine = self._engine()
        pos = _make_pos(ticker="XOM", pnl_pct=5.0, weight_current_pct=6.0,
                        sector="ENERGY")
        geo_states = {
            "HORMUZ_STATUS": FakeGeoRow("PARTIAL"),
            "IRAN_CONFLICT": FakeGeoRow("ACTIVE"),
            "UKRAINE_WAR":   FakeGeoRow("STALEMATE"),
            "NATO_SPENDING": FakeGeoRow("INCREASING"),
        }
        signal, flags = engine._apply_overrides("BUY", [], pos, geo_states, brent=80.0)
        assert signal == "BUY"
        assert flags == []


# ─────────────────────────────────────────────────────────────
# WEIGHTED COMPOSITE SANITY
# ─────────────────────────────────────────────────────────────

class TestWeightedComposite:
    def test_weights_sum_to_one(self):
        from config import SIGNAL_WEIGHTS
        total = sum(SIGNAL_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, not 1.0"

    def test_high_geo_score_pushes_toward_buy(self):
        from src.signal_engine import FactorScores
        # Simulate all max geo score
        factors = FactorScores(
            geo_score=100,
            momentum_score=80,
            drift_score=80,
            drawdown_score=80,
            volatility_score=80,
        )
        from config import SIGNAL_WEIGHTS
        composite = (
            factors.geo_score * SIGNAL_WEIGHTS["sector_geo_score"] +
            factors.momentum_score * SIGNAL_WEIGHTS["price_momentum"] +
            factors.drift_score * SIGNAL_WEIGHTS["position_drift"] +
            factors.drawdown_score * SIGNAL_WEIGHTS["drawdown_risk"] +
            factors.volatility_score * SIGNAL_WEIGHTS["volatility_regime"]
        )
        assert composite >= 75  # Should be BUY

    def test_low_geo_score_pushes_toward_sell(self):
        from src.signal_engine import FactorScores
        factors = FactorScores(
            geo_score=0,
            momentum_score=10,
            drift_score=10,
            drawdown_score=10,
            volatility_score=10,
        )
        from config import SIGNAL_WEIGHTS
        composite = (
            factors.geo_score * SIGNAL_WEIGHTS["sector_geo_score"] +
            factors.momentum_score * SIGNAL_WEIGHTS["price_momentum"] +
            factors.drift_score * SIGNAL_WEIGHTS["position_drift"] +
            factors.drawdown_score * SIGNAL_WEIGHTS["drawdown_risk"] +
            factors.volatility_score * SIGNAL_WEIGHTS["volatility_regime"]
        )
        assert composite < 30  # Should be SELL
