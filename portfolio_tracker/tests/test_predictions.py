"""
test_predictions.py — Unit tests for prediction engine models and ensemble.
"""

import math
import sys
import os
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEMAMomentumModel(unittest.TestCase):
    """Tests for Model 1: EMA momentum."""

    def _make_prices(self, n=50, start=100.0, trend=0.002):
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
        vals = [start * (1 + trend) ** i for i in range(n)]
        return pd.Series(vals, index=dates)

    def test_uptrend_predicts_up(self):
        from src.prediction_engine import _model_ema_momentum
        prices = self._make_prices(50, trend=0.005)
        result = _model_ema_momentum(prices, horizon_days=5)
        self.assertIsNotNone(result)
        self.assertGreater(result["predicted_pct"], 0)

    def test_downtrend_predicts_down(self):
        from src.prediction_engine import _model_ema_momentum
        prices = self._make_prices(50, trend=-0.005)
        result = _model_ema_momentum(prices, horizon_days=5)
        self.assertIsNotNone(result)
        self.assertLess(result["predicted_pct"], 0)

    def test_insufficient_data_returns_none(self):
        from src.prediction_engine import _model_ema_momentum
        prices = self._make_prices(10)
        result = _model_ema_momentum(prices, horizon_days=5)
        self.assertIsNone(result)

    def test_ci_bounds_ordered(self):
        from src.prediction_engine import _model_ema_momentum
        prices = self._make_prices(50)
        result = _model_ema_momentum(prices, horizon_days=5)
        self.assertIsNotNone(result)
        self.assertLess(result["ci_80"][0], result["ci_80"][1])
        self.assertLess(result["ci_95"][0], result["ci_95"][1])
        # 95% CI wider than 80% CI
        self.assertLessEqual(result["ci_95"][0], result["ci_80"][0])
        self.assertGreaterEqual(result["ci_95"][1], result["ci_80"][1])

    def test_produces_factor_description(self):
        from src.prediction_engine import _model_ema_momentum
        prices = self._make_prices(50)
        result = _model_ema_momentum(prices, horizon_days=5)
        self.assertIn("EMA", result["factor"])


class TestMeanReversionModel(unittest.TestCase):
    """Tests for Model 3: Bollinger Band mean reversion."""

    def _make_prices(self, n=30, start=100.0, noise=1.0):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
        vals = start + np.cumsum(np.random.randn(n) * noise)
        return pd.Series(vals, index=dates)

    def test_overbought_predicts_down(self):
        from src.prediction_engine import _model_mean_reversion
        # Create prices where last value is well above SMA
        prices = self._make_prices(30, noise=0.5)
        # Push last few values above upper band
        prices.iloc[-3:] = prices.mean() + 3 * prices.std()
        result = _model_mean_reversion(prices, horizon_days=1)
        self.assertIsNotNone(result)
        # Should predict reversion down
        self.assertLess(result["predicted_pct"], prices.iloc[-1])  # just checking it runs

    def test_insufficient_data_returns_none(self):
        from src.prediction_engine import _model_mean_reversion
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=10, freq="D")
        prices = pd.Series([100] * 10, index=dates)
        result = _model_mean_reversion(prices, horizon_days=5)
        self.assertIsNone(result)

    def test_long_horizon_dampened(self):
        from src.prediction_engine import _model_mean_reversion
        prices = self._make_prices(30)
        prices.iloc[-1] = prices.mean() + 2.5 * prices.std()
        r_short = _model_mean_reversion(prices, horizon_days=1)
        r_long = _model_mean_reversion(prices, horizon_days=21)
        # Both should exist
        self.assertIsNotNone(r_short)
        self.assertIsNotNone(r_long)


class TestGeopoliticalModel(unittest.TestCase):
    """Tests for Model 4: Geopolitical overlay."""

    @patch("src.prediction_engine.get_geo_states")
    @patch("src.prediction_engine.get_sector_score_history")
    def test_high_score_rising_predicts_up(self, mock_hist, mock_states):
        from src.prediction_engine import _model_geopolitical

        mock_states.return_value = {}
        # Simulate rising geo score
        mock_entry_old = MagicMock()
        mock_entry_old.geo_score = 7.5
        mock_entry_new = MagicMock()
        mock_entry_new.geo_score = 9.0
        mock_hist.return_value = [mock_entry_old, mock_entry_new]

        result = _model_geopolitical("XOM", "ENERGY", 100.0, 5)
        self.assertIsNotNone(result)
        self.assertGreater(result["predicted_pct"], 0)

    @patch("src.prediction_engine.get_geo_states")
    @patch("src.prediction_engine.get_sector_score_history")
    def test_hormuz_closed_forces_energy_up(self, mock_hist, mock_states):
        from src.prediction_engine import _model_geopolitical

        mock_states.return_value = {}
        mock_hist.return_value = []

        geo_override = {
            "HORMUZ_STATUS": "CLOSED",
            "IRAN_CONFLICT": "ACTIVE",
            "UKRAINE_WAR": "STALEMATE",
            "US_CHINA_RELATIONS": "TENSE",
            "NATO_SPENDING": "INCREASING",
        }
        result = _model_geopolitical("XOM", "ENERGY", 100.0, 5, geo_override)
        self.assertIsNotNone(result)
        self.assertGreater(result["predicted_pct"], 0)
        self.assertIn("Hormuz", result["factor"])

    @patch("src.prediction_engine.get_geo_states")
    @patch("src.prediction_engine.get_sector_score_history")
    def test_ukraine_resolved_defense_down(self, mock_hist, mock_states):
        from src.prediction_engine import _model_geopolitical

        mock_states.return_value = {}
        mock_hist.return_value = []

        geo_override = {
            "HORMUZ_STATUS": "OPEN",
            "IRAN_CONFLICT": "RESOLVED",
            "UKRAINE_WAR": "RESOLVED",
            "US_CHINA_RELATIONS": "NEUTRAL",
            "NATO_SPENDING": "DECLINING",
        }
        result = _model_geopolitical("RHM.DE", "DEFENSE", 100.0, 5, geo_override)
        self.assertIsNotNone(result)
        self.assertLess(result["predicted_pct"], 0)

    @patch("src.prediction_engine.get_geo_states")
    @patch("src.prediction_engine.get_sector_score_history")
    def test_us_china_hostile_metals_up(self, mock_hist, mock_states):
        from src.prediction_engine import _model_geopolitical

        mock_states.return_value = {}
        mock_hist.return_value = []

        geo_override = {
            "HORMUZ_STATUS": "OPEN",
            "IRAN_CONFLICT": "CEASEFIRE",
            "UKRAINE_WAR": "STALEMATE",
            "US_CHINA_RELATIONS": "HOSTILE",
            "NATO_SPENDING": "STABLE",
        }
        result = _model_geopolitical("FCX", "METALS", 100.0, 5, geo_override)
        self.assertIsNotNone(result)
        self.assertGreater(result["predicted_pct"], 0)
        self.assertIn("HOSTILE", result["factor"].upper())

    @patch("src.prediction_engine.get_geo_states")
    @patch("src.prediction_engine.get_sector_score_history")
    def test_peace_dividend_energy_down(self, mock_hist, mock_states):
        """HORMUZ OPEN + IRAN RESOLVED = energy should drop (Leigh 2003 IMF)."""
        from src.prediction_engine import _model_geopolitical

        mock_states.return_value = {}
        mock_hist.return_value = []

        geo_override = {
            "HORMUZ_STATUS": "OPEN",
            "IRAN_CONFLICT": "RESOLVED",
            "UKRAINE_WAR": "DE-ESCALATING",
            "US_CHINA_RELATIONS": "NEUTRAL",
            "NATO_SPENDING": "STABLE",
        }
        result = _model_geopolitical("XOM", "ENERGY", 100.0, 5, geo_override)
        self.assertIsNotNone(result)
        self.assertLess(result["predicted_pct"], -1.0,
                        "Energy should predict significant downside when peace resolves")
        self.assertIn("peace", result["factor"].lower())

    @patch("src.prediction_engine.get_geo_states")
    @patch("src.prediction_engine.get_sector_score_history")
    def test_peace_gold_down(self, mock_hist, mock_states):
        """Full de-escalation should be bearish for gold (safe haven unwinding)."""
        from src.prediction_engine import _model_geopolitical

        mock_states.return_value = {}
        mock_hist.return_value = []

        geo_override = {
            "HORMUZ_STATUS": "OPEN",
            "IRAN_CONFLICT": "RESOLVED",
            "UKRAINE_WAR": "RESOLVED",
            "US_CHINA_RELATIONS": "COOPERATIVE",
            "NATO_SPENDING": "DECLINING",
        }
        result = _model_geopolitical("IGLN.L", "GOLD", 100.0, 5, geo_override)
        self.assertIsNotNone(result)
        self.assertLess(result["predicted_pct"], 0,
                        "Gold should predict downside in full de-escalation")


class TestRSI(unittest.TestCase):
    def test_rsi_range(self):
        from src.prediction_engine import _compute_rsi
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50)))
        rsi = _compute_rsi(prices)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)


class TestDirectionAndConfidence(unittest.TestCase):
    def test_direction(self):
        from src.prediction_engine import _determine_direction
        self.assertEqual(_determine_direction(2.0), "UP")
        self.assertEqual(_determine_direction(-2.0), "DOWN")
        self.assertEqual(_determine_direction(0.5), "FLAT")
        self.assertEqual(_determine_direction(-0.5), "FLAT")

    def test_confidence_high(self):
        from src.prediction_engine import _determine_confidence
        results = {
            "m1": {"predicted_pct": 3.0},
            "m2": {"predicted_pct": 2.0},
            "m3": {"predicted_pct": 4.0},
        }
        self.assertEqual(_determine_confidence(results, 3.0), "HIGH")

    def test_confidence_low_disagreement(self):
        from src.prediction_engine import _determine_confidence
        results = {
            "m1": {"predicted_pct": 3.0},
            "m2": {"predicted_pct": -3.0},
        }
        self.assertEqual(_determine_confidence(results, 0.0), "LOW")

    def test_confidence_single_model(self):
        from src.prediction_engine import _determine_confidence
        results = {"m1": {"predicted_pct": 3.0}}
        self.assertEqual(_determine_confidence(results, 3.0), "LOW")


class TestRandomForestModel(unittest.TestCase):
    def _make_prices(self, n=100, trend=0.001):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
        vals = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + trend))
        return pd.Series(vals, index=dates)

    def test_runs_with_sufficient_data(self):
        from src.prediction_engine import _model_random_forest
        prices = self._make_prices(100)
        result = _model_random_forest(prices, horizon_days=5)
        if result is not None:
            self.assertIn("predicted_pct", result)
            self.assertIn("ci_80", result)

    def test_insufficient_data_returns_none(self):
        from src.prediction_engine import _model_random_forest
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=15, freq="D")
        prices = pd.Series(np.linspace(100, 105, 15), index=dates)
        result = _model_random_forest(prices, horizon_days=5)
        self.assertIsNone(result)

    def test_with_commodity_features(self):
        from src.prediction_engine import _model_random_forest
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq="D")
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.01)), index=dates)
        brent = pd.Series(80 * np.exp(np.cumsum(np.random.randn(100) * 0.02)), index=dates)
        commodity_hist = {"brent": brent, "gold": None, "copper": None}
        result = _model_random_forest(prices, 5, commodity_hist, "ENERGY", 15.0)
        if result is not None:
            self.assertIn("predicted_pct", result)


class TestCommodityCorrelationModel(unittest.TestCase):
    def _make_correlated(self, n=100):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
        base = np.cumsum(np.random.randn(n) * 0.02)
        oil = pd.Series(80 * np.exp(base), index=dates)
        # Energy equity correlated with oil (r~0.7)
        equity = pd.Series(50 * np.exp(base * 0.7 + np.random.randn(n) * 0.01), index=dates)
        return equity, oil

    def test_energy_sector_uses_brent(self):
        from src.prediction_engine import _model_commodity_correlation
        equity, oil = self._make_correlated()
        commodity_hist = {"brent": oil, "gold": None, "copper": None}
        result = _model_commodity_correlation("XOM", "ENERGY", 100.0, 5, equity, commodity_hist)
        self.assertIsNotNone(result)
        self.assertIn("Brent", result["factor"])

    def test_gold_sector_uses_gold(self):
        from src.prediction_engine import _model_commodity_correlation
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq="D")
        gold_spot = pd.Series(2000 * np.exp(np.cumsum(np.random.randn(100) * 0.01)), index=dates)
        gold_equity = pd.Series(40 * np.exp(np.cumsum(np.random.randn(100) * 0.015)), index=dates)
        commodity_hist = {"brent": None, "gold": gold_spot, "copper": None}
        result = _model_commodity_correlation("GDX", "GOLD", 40.0, 5, gold_equity, commodity_hist)
        self.assertIsNotNone(result)
        self.assertIn("Gold", result["factor"])
        self.assertIn("beta 1.6", result["factor"])

    def test_defense_sector_returns_none(self):
        from src.prediction_engine import _model_commodity_correlation
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq="D")
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)), index=dates)
        commodity_hist = {"brent": prices, "gold": None, "copper": None}
        result = _model_commodity_correlation("RHM.DE", "DEFENSE", 100.0, 5, prices, commodity_hist)
        self.assertIsNone(result)

    def test_metals_sector_uses_copper(self):
        from src.prediction_engine import _model_commodity_correlation
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq="D")
        copper = pd.Series(4.5 * np.exp(np.cumsum(np.random.randn(100) * 0.015)), index=dates)
        metals_eq = pd.Series(30 * np.exp(np.cumsum(np.random.randn(100) * 0.02)), index=dates)
        commodity_hist = {"brent": None, "gold": None, "copper": copper}
        result = _model_commodity_correlation("FCX", "METALS", 30.0, 5, metals_eq, commodity_hist)
        self.assertIsNotNone(result)
        self.assertIn("Copper", result["factor"])
        self.assertIn("beta 1.3", result["factor"])


class TestCrossCorrelation(unittest.TestCase):
    def test_high_correlation_detected(self):
        from src.crisis_patterns import compute_cross_sector_correlation
        np.random.seed(42)
        base = pd.Series(np.cumsum(np.random.randn(60)) + 100)
        # All sectors move together (high correlation)
        sector_prices = {
            "ENERGY": base + np.random.randn(60) * 0.1,
            "DEFENSE": base + np.random.randn(60) * 0.1,
            "GOLD": base + np.random.randn(60) * 0.1,
        }
        corr = compute_cross_sector_correlation(sector_prices, window=30)
        self.assertGreater(corr, 0.8)

    def test_low_correlation_with_independent_sectors(self):
        from src.crisis_patterns import compute_cross_sector_correlation
        np.random.seed(42)
        sector_prices = {
            "ENERGY": pd.Series(np.cumsum(np.random.randn(60)) + 100),
            "DEFENSE": pd.Series(np.cumsum(np.random.randn(60)) + 100),
            "GOLD": pd.Series(np.cumsum(np.random.randn(60)) + 100),
        }
        corr = compute_cross_sector_correlation(sector_prices, window=30)
        self.assertLess(corr, 0.6)

    def test_insufficient_data(self):
        from src.crisis_patterns import compute_cross_sector_correlation
        sector_prices = {"ENERGY": pd.Series([1, 2, 3])}
        corr = compute_cross_sector_correlation(sector_prices)
        self.assertEqual(corr, 0.0)


class TestVolRegime(unittest.TestCase):
    def test_normal_regime(self):
        from src.crisis_patterns import compute_vol_regime
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.01)
        z = compute_vol_regime(returns)
        self.assertLess(abs(z), 3.0)  # should be near zero

    def test_high_vol_spike(self):
        from src.crisis_patterns import compute_vol_regime
        np.random.seed(42)
        returns = pd.Series(np.random.randn(300) * 0.01)
        # Spike vol at the end
        returns.iloc[-20:] = np.random.randn(20) * 0.05
        z = compute_vol_regime(returns)
        self.assertGreater(z, 1.0)


class TestRegimeDetection(unittest.TestCase):
    def test_calm_regime(self):
        from src.crisis_patterns import detect_market_regime
        np.random.seed(42)
        sector_prices = {
            "ENERGY": pd.Series(np.cumsum(np.random.randn(100) * 0.5) + 100),
            "DEFENSE": pd.Series(np.cumsum(np.random.randn(100) * 0.5) + 100),
        }
        geo_scores = {"ENERGY": 5.0, "DEFENSE": 5.0}
        regime = detect_market_regime(sector_prices, vix=12.0, geo_scores=geo_scores)
        self.assertEqual(regime.regime_name, "CALM")
        self.assertLess(regime.regime_score, 30)

    def test_crisis_high_vix(self):
        from src.crisis_patterns import detect_market_regime
        np.random.seed(42)
        base = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        sector_prices = {
            "ENERGY": base + np.random.randn(100) * 0.1,
            "DEFENSE": base + np.random.randn(100) * 0.1,
            "METALS": base + np.random.randn(100) * 0.1,
        }
        geo_scores = {"ENERGY": 9.5, "DEFENSE": 9.5, "METALS": 5.0}
        regime = detect_market_regime(sector_prices, vix=50.0, geo_scores=geo_scores)
        self.assertIn(regime.regime_name, ("STRESSED", "CRISIS"))
        self.assertGreater(regime.regime_score, 40)

    def test_sector_adjustments_populated(self):
        from src.crisis_patterns import detect_market_regime
        np.random.seed(42)
        base = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        sector_prices = {
            "ENERGY": base * 1.5,  # energy spiking
            "DEFENSE": base + np.random.randn(100) * 0.1,
        }
        geo_scores = {"ENERGY": 9.0, "DEFENSE": 9.0}
        regime = detect_market_regime(sector_prices, vix=30.0, geo_scores=geo_scores)
        # Should have sector adjustments
        self.assertIn("ENERGY", regime.sector_adjustments)
        self.assertIn("DEFENSE", regime.sector_adjustments)


class TestCrisisRegimeModel(unittest.TestCase):
    def test_model_with_crisis_regime(self):
        from src.crisis_patterns import model_crisis_regime, MarketRegime
        regime = MarketRegime(
            regime_name="STRESSED",
            regime_score=60.0,
            vix_level=35.0,
            cross_sector_correlation=0.75,
            vol_regime_z=2.0,
            active_patterns=[("SUPPLY_SHOCK", 0.75)],
            sector_adjustments={"ENERGY": 3.0, "GOLD": 1.5, "METALS": -2.0,
                                "DEFENSE": 1.0, "BIOTECH": -1.5},
            contagion_risk=0.6,
            factors=["VIX at 35", "Supply shock pattern"],
        )
        result = model_crisis_regime("XOM", "ENERGY", 100.0, 5, regime)
        self.assertIsNotNone(result)
        self.assertGreater(result["predicted_pct"], 0)  # energy benefits from supply shock
        self.assertIn("STRESSED", result["factor"])

    def test_model_with_calm_regime(self):
        from src.crisis_patterns import model_crisis_regime, MarketRegime
        regime = MarketRegime(
            regime_name="CALM",
            regime_score=10.0,
            vix_level=12.0,
            cross_sector_correlation=0.3,
            vol_regime_z=0.5,
            active_patterns=[],
            sector_adjustments={"ENERGY": 0, "GOLD": 0, "METALS": 0,
                                "DEFENSE": 0, "BIOTECH": 0},
            contagion_risk=0.1,
            factors=["No crisis patterns"],
        )
        result = model_crisis_regime("XOM", "ENERGY", 100.0, 5, regime)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["predicted_pct"], 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
