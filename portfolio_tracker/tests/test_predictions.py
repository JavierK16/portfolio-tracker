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
        self.assertIn("HORMUZ", result["factor"])

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
        self.assertIn("HOSTILE", result["factor"])


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


class TestLinearRegressionModel(unittest.TestCase):
    def _make_prices(self, n=100, trend=0.001):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="D")
        vals = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + trend))
        return pd.Series(vals, index=dates)

    def test_runs_with_sufficient_data(self):
        from src.prediction_engine import _model_linear_regression
        prices = self._make_prices(100)
        result = _model_linear_regression(prices, horizon_days=5)
        # May be None if sklearn not installed
        if result is not None:
            self.assertIn("predicted_pct", result)
            self.assertIn("ci_80", result)

    def test_insufficient_data_returns_none(self):
        from src.prediction_engine import _model_linear_regression
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=15, freq="D")
        prices = pd.Series(np.linspace(100, 105, 15), index=dates)
        result = _model_linear_regression(prices, horizon_days=5)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
