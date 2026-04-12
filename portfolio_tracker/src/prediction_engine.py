"""
prediction_engine.py — 5-model ensemble prediction engine.
Models: EMA momentum, Ridge regression, Bollinger mean-reversion,
        geopolitical overlay, crisis pattern & cross-sector regime.
Produces position-, sector-, and portfolio-level forecasts at 3 horizons (24h, 1w, 1m).
"""

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    PORTFOLIO, SECTOR_CONFIG, PREDICTION_ENSEMBLE_WEIGHTS,
    PREDICTION_HORIZONS, PREDICTION_MIN_HISTORY_DAYS,
    PREDICTION_REFRESH_INTERVAL, GEO_VARIABLES_DEFAULT,
)
from src.database import (
    get_price_history, save_prediction, get_geo_states,
    get_sector_score_history, get_recent_news,
)
from src.crisis_patterns import (
    detect_market_regime, model_crisis_regime, MarketRegime,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class PricePrediction:
    ticker: str
    horizon: str
    current_price_eur: float
    predicted_price_eur: float
    predicted_change_pct: float
    direction: str
    confidence_level: str
    ci_80: Tuple[float, float]
    ci_95: Tuple[float, float]
    model_used: str
    factors: List[str]
    model_scores: Dict[str, float]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    warning: Optional[str] = None


@dataclass
class SectorPrediction:
    sector: str
    horizon: str
    current_value_eur: float
    predicted_value_eur: float
    predicted_change_pct: float
    direction: str
    confidence_level: str
    ci_80: Tuple[float, float]
    ci_95: Tuple[float, float]
    top_driver: str
    position_predictions: List[PricePrediction]


@dataclass
class PortfolioPrediction:
    horizon: str
    current_value_eur: float
    predicted_value_eur: float
    predicted_change_pct: float
    predicted_pnl_eur: float
    direction: str
    overall_confidence: str
    ci_80: Tuple[float, float]
    ci_95: Tuple[float, float]
    sector_predictions: List[SectorPrediction]
    risk_summary: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────
# INDIVIDUAL MODELS
# ─────────────────────────────────────────────────────────────

def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI from a price series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _model_ema_momentum(prices: pd.Series, horizon_days: int) -> Optional[dict]:
    """Model 1: EMA crossover momentum."""
    if len(prices) < 21:
        return None

    ema8 = prices.ewm(span=8, adjust=False).mean()
    ema21 = prices.ewm(span=21, adjust=False).mean()

    current = prices.iloc[-1]
    ema8_now = ema8.iloc[-1]
    ema21_now = ema21.iloc[-1]

    # Trend direction from EMA8 slope over last 10 days
    ema8_recent = ema8.iloc[-10:]
    if len(ema8_recent) < 2:
        return None

    x = np.arange(len(ema8_recent))
    slope = np.polyfit(x, ema8_recent.values, 1)[0]
    daily_rate = slope / current if current > 0 else 0

    # Determine signal
    cross_up = ema8_now > ema21_now
    gap_widening = abs(ema8_now - ema21_now) > abs(ema8.iloc[-5] - ema21.iloc[-5])

    if cross_up and gap_widening:
        factor_desc = "EMA8 crossed above EMA21 (bullish momentum)"
    elif cross_up and not gap_widening:
        factor_desc = "EMA8 above EMA21 but converging (momentum fading)"
        daily_rate *= 0.5
    elif not cross_up and gap_widening:
        factor_desc = "EMA8 below EMA21 and widening (bearish momentum)"
    else:
        factor_desc = "EMA8 below EMA21 but converging (potential reversal)"
        daily_rate *= 0.5

    predicted_pct = daily_rate * horizon_days * 100
    predicted_price = current * (1 + daily_rate * horizon_days)

    # CI from recent volatility
    daily_returns = prices.pct_change().dropna()
    vol = daily_returns.std() * math.sqrt(horizon_days) if len(daily_returns) > 5 else 0.05
    ci_80 = (current * (1 + predicted_pct / 100 - 1.28 * vol),
             current * (1 + predicted_pct / 100 + 1.28 * vol))
    ci_95 = (current * (1 + predicted_pct / 100 - 1.96 * vol),
             current * (1 + predicted_pct / 100 + 1.96 * vol))

    return {
        "predicted_pct": predicted_pct,
        "predicted_price": predicted_price,
        "ci_80": ci_80,
        "ci_95": ci_95,
        "factor": factor_desc,
    }


def _model_linear_regression(prices: pd.Series, horizon_days: int) -> Optional[dict]:
    """Model 2: Ridge regression with momentum features."""
    if len(prices) < 30:
        return None

    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        logger.warning("scikit-learn not installed — skipping linear regression model")
        return None

    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        return None

    # Build feature matrix
    df = pd.DataFrame(index=returns.index)
    df["ret_5d"] = returns.rolling(5).sum()
    df["ret_10d"] = returns.rolling(10).sum()
    df["ret_21d"] = returns.rolling(21).sum()
    df["vol_5d"] = returns.rolling(5).std()
    df["rsi"] = 0.0

    # RSI
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = rsi_series.reindex(df.index)

    # Price vs SMA50
    sma50 = prices.rolling(50).mean()
    df["price_vs_sma50"] = ((prices / sma50) - 1).reindex(df.index)

    # Target: forward N-day return
    df["target"] = returns.rolling(horizon_days).sum().shift(-horizon_days)

    df = df.dropna()
    if len(df) < 20:
        return None

    feature_cols = ["ret_5d", "ret_10d", "ret_21d", "vol_5d", "rsi", "price_vs_sma50"]
    X = df[feature_cols].values
    y = df["target"].values

    # Train on all but last row (last row is our prediction input)
    X_train, y_train = X[:-1], y[:-1]
    X_pred = X[-1:].copy()

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred_return = model.predict(X_pred)[0]

    # Residual standard error for CI
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    rse = np.std(residuals)

    current = prices.iloc[-1]
    predicted_pct = pred_return * 100
    predicted_price = current * (1 + pred_return)

    ci_80 = (current * (1 + pred_return - 1.28 * rse),
             current * (1 + pred_return + 1.28 * rse))
    ci_95 = (current * (1 + pred_return - 1.96 * rse),
             current * (1 + pred_return + 1.96 * rse))

    # Top feature contribution
    rsi_val = df["rsi"].iloc[-1]
    if rsi_val > 70:
        factor_desc = f"RSI at {rsi_val:.0f} (overbought, reversion risk)"
    elif rsi_val < 30:
        factor_desc = f"RSI at {rsi_val:.0f} (oversold, bounce potential)"
    else:
        sma_dev = df["price_vs_sma50"].iloc[-1] * 100
        factor_desc = f"Price {sma_dev:+.1f}% vs 50-day SMA (regression signal)"

    return {
        "predicted_pct": predicted_pct,
        "predicted_price": predicted_price,
        "ci_80": ci_80,
        "ci_95": ci_95,
        "factor": factor_desc,
    }


def _model_mean_reversion(prices: pd.Series, horizon_days: int) -> Optional[dict]:
    """Model 3: Bollinger Band mean reversion."""
    if len(prices) < 20:
        return None

    sma20 = prices.rolling(20).mean()
    std20 = prices.rolling(20).std()

    current = prices.iloc[-1]
    sma = sma20.iloc[-1]
    std = std20.iloc[-1]

    if std == 0 or np.isnan(sma):
        return None

    upper = sma + 2 * std
    lower = sma - 2 * std
    z_score = (current - sma) / std

    # Predict reversion toward SMA
    if current > upper:
        reversion_target = sma + 0.5 * std  # partial reversion
        factor_desc = f"Price {z_score:.1f} sigma above 20-day mean (mean reversion expected)"
    elif current < lower:
        reversion_target = sma - 0.5 * std
        factor_desc = f"Price {z_score:.1f} sigma below 20-day mean (mean reversion expected)"
    else:
        # Within bands — slight continuation toward nearest band
        if current > sma:
            reversion_target = current + 0.3 * std
        else:
            reversion_target = current - 0.3 * std
        factor_desc = f"Price within Bollinger Bands (z={z_score:.1f}), moderate continuation"

    # Scale by horizon — mean reversion stronger at short horizons
    horizon_factor = min(1.0, horizon_days / 5.0)
    predicted_price = current + (reversion_target - current) * horizon_factor
    predicted_pct = ((predicted_price / current) - 1) * 100

    # Reduce weight for 1-month horizon (spec says 10%)
    if horizon_days >= 21:
        predicted_pct *= 0.5  # dampen long-term mean reversion
        predicted_price = current * (1 + predicted_pct / 100)

    vol = std / current * math.sqrt(horizon_days)
    ci_80 = (current * (1 + predicted_pct / 100 - 1.28 * vol),
             current * (1 + predicted_pct / 100 + 1.28 * vol))
    ci_95 = (current * (1 + predicted_pct / 100 - 1.96 * vol),
             current * (1 + predicted_pct / 100 + 1.96 * vol))

    return {
        "predicted_pct": predicted_pct,
        "predicted_price": predicted_price,
        "ci_80": ci_80,
        "ci_95": ci_95,
        "factor": factor_desc,
    }


def _model_geopolitical(ticker: str, sector: str, current_price: float,
                        horizon_days: int,
                        geo_states_override: Optional[Dict] = None) -> Optional[dict]:
    """Model 4: Geopolitical sentiment overlay."""
    cfg = SECTOR_CONFIG.get(sector)
    if not cfg:
        return None

    # Get geo states
    if geo_states_override:
        geo_vars = geo_states_override
    else:
        db_states = get_geo_states()
        geo_vars = {k: v.current_value for k, v in db_states.items()}
        # Fill defaults for missing
        for k, v in GEO_VARIABLES_DEFAULT.items():
            if k not in geo_vars:
                geo_vars[k] = v

    # Get sector geo score and trend
    score_hist = get_sector_score_history(sector, days=7)
    if score_hist and len(score_hist) >= 2:
        current_score = score_hist[-1].geo_score or cfg["base_score"]
        old_score = score_hist[0].geo_score or cfg["base_score"]
        score_delta = current_score - old_score
        if score_delta > 0.3:
            trend = "rising"
        elif score_delta < -0.3:
            trend = "falling"
        else:
            trend = "flat"
    else:
        current_score = cfg["base_score"]
        score_delta = 0
        trend = "flat"

    current_score = max(0.0, min(10.0, current_score))

    # Base prediction from geo score + trend
    if current_score > 8 and trend == "rising":
        daily_pct = (current_score - 5) * 0.5
        factor_desc = f"Geo score {current_score:.1f} and rising (+{score_delta:.1f})"
    elif current_score > 8 and trend == "flat":
        daily_pct = 0.1
        factor_desc = f"Geo score {current_score:.1f} and stable (slight upside)"
    elif 5 <= current_score <= 8 and trend == "rising":
        daily_pct = (current_score - 5) * 0.3
        factor_desc = f"Geo score {current_score:.1f} rising (moderate upside)"
    elif 5 <= current_score <= 8 and trend == "falling":
        daily_pct = -(5 - (current_score - 5)) * 0.2
        factor_desc = f"Geo score {current_score:.1f} falling (moderate downside)"
    elif current_score < 5 and trend == "falling":
        daily_pct = -(5 - current_score) * 0.4
        factor_desc = f"Geo score {current_score:.1f} falling (bearish sentiment)"
    elif current_score < 5 and trend == "rising":
        daily_pct = 0.05
        factor_desc = f"Geo score {current_score:.1f} recovering (flat to slight up)"
    else:
        daily_pct = 0
        factor_desc = f"Geo score {current_score:.1f} neutral"

    predicted_pct = daily_pct * horizon_days / 100

    # CRITICAL OVERRIDES
    override_factor = None
    hormuz = geo_vars.get("HORMUZ_STATUS", "OPEN")
    ukraine = geo_vars.get("UKRAINE_WAR", "STALEMATE")
    nato = geo_vars.get("NATO_SPENDING", "INCREASING")
    us_china = geo_vars.get("US_CHINA_RELATIONS", "TENSE")

    if hormuz == "CLOSED":
        if sector == "ENERGY":
            predicted_pct = max(predicted_pct, 0.03 + 0.05 * (horizon_days / 5))
            override_factor = f"HORMUZ CLOSED: forcing +{predicted_pct*100:.1f}% energy premium"
        elif sector == "GOLD":
            predicted_pct = max(predicted_pct, 0.01 + 0.02 * (horizon_days / 5))
            override_factor = f"HORMUZ CLOSED: safe haven +{predicted_pct*100:.1f}%"

    if ukraine == "RESOLVED" and nato == "DECLINING":
        if sector == "DEFENSE":
            predicted_pct = min(predicted_pct, -0.02 - 0.03 * (horizon_days / 5))
            override_factor = f"Ukraine RESOLVED + NATO DECLINING: defense -{abs(predicted_pct)*100:.1f}%"

    if us_china == "HOSTILE":
        if sector == "METALS":
            predicted_pct = max(predicted_pct, 0.02 + 0.03 * (horizon_days / 5))
            override_factor = f"US-China HOSTILE: metals supply disruption +{predicted_pct*100:.1f}%"

    if override_factor:
        factor_desc = override_factor

    predicted_pct_display = predicted_pct * 100
    predicted_price = current_price * (1 + predicted_pct)

    # CI — wider when geo variables volatile
    volatile_count = sum(1 for v in [
        hormuz == "CLOSED", hormuz == "PARTIAL",
        geo_vars.get("IRAN_CONFLICT") == "ACTIVE",
        ukraine == "ESCALATING",
    ] if v)
    base_vol = 0.02 * math.sqrt(horizon_days)
    vol = base_vol * (1 + 0.3 * volatile_count)

    ci_80 = (current_price * (1 + predicted_pct - 1.28 * vol),
             current_price * (1 + predicted_pct + 1.28 * vol))
    ci_95 = (current_price * (1 + predicted_pct - 1.96 * vol),
             current_price * (1 + predicted_pct + 1.96 * vol))

    return {
        "predicted_pct": predicted_pct_display,
        "predicted_price": predicted_price,
        "ci_80": ci_80,
        "ci_95": ci_95,
        "factor": factor_desc,
    }


# ─────────────────────────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────────────────────────

def _determine_direction(pct: float) -> str:
    if pct > 1.0:
        return "UP"
    elif pct < -1.0:
        return "DOWN"
    return "FLAT"


def _determine_confidence(model_results: Dict[str, dict], final_pct: float) -> str:
    """HIGH if >=3 agree on direction and 80% CI doesn't cross zero,
    MEDIUM if >=2 agree, LOW otherwise."""
    if len(model_results) < 2:
        return "LOW"

    directions = []
    for m in model_results.values():
        d = _determine_direction(m["predicted_pct"])
        directions.append(d)

    final_dir = _determine_direction(final_pct)
    agree_count = sum(1 for d in directions if d == final_dir)

    if agree_count >= 3:
        return "HIGH"
    elif agree_count >= 2:
        return "MEDIUM"
    return "LOW"


def run_ensemble_for_position(ticker: str, sector: str, current_price_eur: float,
                              horizon_name: str, horizon_days: int,
                              geo_states_override: Optional[Dict] = None,
                              regime: Optional[MarketRegime] = None) -> Optional[PricePrediction]:
    """Run all 5 models and combine into ensemble prediction for one position/horizon."""
    if current_price_eur is None or current_price_eur <= 0:
        return None

    # Get price history — use full backfilled data (up to 730 days)
    hist = get_price_history(ticker, days=730)
    if hist:
        prices = pd.Series(
            [h.price_eur for h in hist if h.price_eur],
            index=pd.to_datetime([h.timestamp for h in hist if h.price_eur]),
        ).sort_index()
    else:
        prices = pd.Series(dtype=float)

    has_enough_history = len(prices) >= PREDICTION_MIN_HISTORY_DAYS
    warning = None if has_enough_history else f"Limited data: only {len(prices)} days"

    # Run models
    model_results = {}
    weights = dict(PREDICTION_ENSEMBLE_WEIGHTS)

    if has_enough_history:
        m1 = _model_ema_momentum(prices, horizon_days)
        if m1:
            model_results["ema_momentum"] = m1

        m2 = _model_linear_regression(prices, horizon_days)
        if m2:
            model_results["linear_regression"] = m2

        m3 = _model_mean_reversion(prices, horizon_days)
        if m3:
            model_results["mean_reversion"] = m3

    m4 = _model_geopolitical(ticker, sector, current_price_eur, horizon_days,
                              geo_states_override)
    if m4:
        model_results["geopolitical"] = m4

    # Model 5: Crisis pattern & cross-sector regime
    if regime is not None:
        m5 = model_crisis_regime(ticker, sector, current_price_eur, horizon_days, regime)
        if m5:
            model_results["crisis_regime"] = m5

    if not model_results:
        return None

    # Redistribute weights for missing models
    active_weights = {k: weights[k] for k in model_results}
    total_w = sum(active_weights.values())
    norm_weights = {k: v / total_w for k, v in active_weights.items()}

    # Weighted average
    final_pct = sum(norm_weights[k] * model_results[k]["predicted_pct"] for k in model_results)
    final_price = current_price_eur * (1 + final_pct / 100)

    # CI: 80% = weighted average, 95% = widest
    ci_80_lower = sum(norm_weights[k] * model_results[k]["ci_80"][0] for k in model_results)
    ci_80_upper = sum(norm_weights[k] * model_results[k]["ci_80"][1] for k in model_results)

    ci_95_lower = min(model_results[k]["ci_95"][0] for k in model_results)
    ci_95_upper = max(model_results[k]["ci_95"][1] for k in model_results)

    # Widen 95% CI if geo variables volatile
    geo_vars = geo_states_override or {}
    if not geo_vars:
        db_states = get_geo_states()
        geo_vars = {k: v.current_value for k, v in db_states.items()}
    volatile = any([
        geo_vars.get("HORMUZ_STATUS") == "CLOSED",
        geo_vars.get("IRAN_CONFLICT") == "ACTIVE",
        geo_vars.get("UKRAINE_WAR") == "ESCALATING",
    ])
    if volatile:
        mid_95 = (ci_95_lower + ci_95_upper) / 2
        half_range = (ci_95_upper - ci_95_lower) / 2 * 1.5
        ci_95_lower = mid_95 - half_range
        ci_95_upper = mid_95 + half_range

    direction = _determine_direction(final_pct)
    confidence = _determine_confidence(model_results, final_pct)
    if not has_enough_history:
        confidence = "LOW"

    # Top 3 factors
    factors = [model_results[k]["factor"] for k in model_results][:3]
    model_scores = {k: v["predicted_pct"] for k, v in model_results.items()}

    model_used = "ENSEMBLE" if len(model_results) > 1 else list(model_results.keys())[0]

    return PricePrediction(
        ticker=ticker,
        horizon=horizon_name,
        current_price_eur=current_price_eur,
        predicted_price_eur=round(final_price, 4),
        predicted_change_pct=round(final_pct, 2),
        direction=direction,
        confidence_level=confidence,
        ci_80=(round(ci_80_lower, 4), round(ci_80_upper, 4)),
        ci_95=(round(ci_95_lower, 4), round(ci_95_upper, 4)),
        model_used=model_used,
        factors=factors,
        model_scores=model_scores,
        warning=warning,
    )


# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE CLASS
# ─────────────────────────────────────────────────────────────

class PredictionEngine:
    """Thread-safe prediction engine with background refresh."""

    def __init__(self):
        self._lock = threading.Lock()
        self._position_predictions: Dict[str, Dict[str, PricePrediction]] = {}  # ticker -> horizon -> pred
        self._sector_predictions: Dict[str, Dict[str, SectorPrediction]] = {}
        self._portfolio_predictions: Dict[str, PortfolioPrediction] = {}
        self._current_regime: Optional[MarketRegime] = None
        self._last_refresh: Optional[datetime] = None
        self._running = False

    def get_current_regime(self) -> Optional[MarketRegime]:
        with self._lock:
            return self._current_regime

    def _compute_regime(self, positions, geo_states_override=None) -> Optional[MarketRegime]:
        """Build sector price series from DB and detect market regime."""
        from src.price_engine import get_price_engine

        try:
            pe = get_price_engine()

            # Build sector-level price series (value-weighted)
            sector_prices: Dict[str, pd.Series] = {}
            for sector in SECTOR_CONFIG:
                sector_positions = [p for p in positions if p.sector == sector]
                sector_series = []
                for pos in sector_positions:
                    hist = get_price_history(pos.ticker, days=365)
                    if hist and len(hist) > 30:
                        s = pd.Series(
                            [(h.price_eur or 0) * (pos.shares_units or 1) for h in hist],
                            index=pd.to_datetime([h.timestamp for h in hist]),
                        ).sort_index()
                        # Resample to daily
                        s = s.resample("1D").last().ffill().dropna()
                        if len(s) > 30:
                            sector_series.append(s)
                if sector_series:
                    combined = pd.concat(sector_series, axis=1).ffill().sum(axis=1)
                    if len(combined) > 30:
                        sector_prices[sector] = combined

            # Get VIX
            vix = pe.get_vix()

            # Get geo scores
            geo_scores = {}
            for sector in SECTOR_CONFIG:
                score_hist = get_sector_score_history(sector, days=7)
                if score_hist:
                    geo_scores[sector] = max(0.0, min(10.0, score_hist[-1].geo_score or 5.0))
                else:
                    geo_scores[sector] = SECTOR_CONFIG[sector]["base_score"]

            regime = detect_market_regime(sector_prices, vix, geo_scores)
            logger.info(
                "Market regime: %s (score %.0f/100, correlation %.2f, "
                "contagion %.0f%%, patterns: %s)",
                regime.regime_name, regime.regime_score,
                regime.cross_sector_correlation, regime.contagion_risk * 100,
                ", ".join(f"{p[0]}({p[1]:.0%})" for p in regime.active_patterns) or "none",
            )
            return regime

        except Exception as e:
            logger.error("Regime detection failed: %s", e)
            return None

    def refresh_all(self, geo_states_override: Optional[Dict] = None) -> None:
        """Generate predictions for all positions, sectors, portfolio."""
        from src.price_engine import get_price_engine

        pe = get_price_engine()
        positions = pe.get_all_positions()

        # Compute market regime ONCE for all positions (cross-sector context)
        regime = self._compute_regime(positions, geo_states_override)

        new_pos_preds: Dict[str, Dict[str, PricePrediction]] = {}
        new_sec_preds: Dict[str, Dict[str, SectorPrediction]] = {}
        new_port_preds: Dict[str, PortfolioPrediction] = {}

        # Position-level predictions
        for pos in positions:
            if pos.current_price_eur is None or pos.current_price_eur <= 0:
                continue
            ticker_preds = {}
            for h_name, h_days in PREDICTION_HORIZONS.items():
                try:
                    pred = run_ensemble_for_position(
                        pos.ticker, pos.sector, pos.current_price_eur,
                        h_name, h_days, geo_states_override, regime,
                    )
                    if pred:
                        ticker_preds[h_name] = pred
                        # Store to DB (skip for scenario overrides)
                        if geo_states_override is None:
                            target_date = datetime.now(timezone.utc) + timedelta(days=h_days)
                            save_prediction(
                                ticker=pos.ticker, sector=pos.sector, level="POSITION",
                                horizon=h_name, target_date=target_date,
                                current_price_eur=pred.current_price_eur,
                                predicted_price_eur=pred.predicted_price_eur,
                                predicted_change_pct=pred.predicted_change_pct,
                                direction=pred.direction,
                                confidence_level=pred.confidence_level,
                                ci_80=pred.ci_80, ci_95=pred.ci_95,
                                model_used=pred.model_used,
                                factors_json=json.dumps(pred.factors),
                            )
                        logger.info(
                            "Prediction for %s %s: %+.1f%% %s (%s)",
                            pos.ticker, h_name, pred.predicted_change_pct,
                            pred.direction, pred.confidence_level,
                        )
                except Exception as e:
                    logger.error("Prediction failed for %s %s: %s", pos.ticker, h_name, e)
            if ticker_preds:
                new_pos_preds[pos.ticker] = ticker_preds

        # Sector-level aggregation
        for sector, cfg in SECTOR_CONFIG.items():
            sector_positions = [p for p in positions if p.sector == sector]
            for h_name, h_days in PREDICTION_HORIZONS.items():
                try:
                    pos_preds = []
                    total_current = 0
                    total_predicted = 0

                    for sp in sector_positions:
                        pred = new_pos_preds.get(sp.ticker, {}).get(h_name)
                        if pred and sp.current_value_eur:
                            pos_preds.append(pred)
                            val = sp.current_value_eur
                            total_current += val
                            total_predicted += val * (1 + pred.predicted_change_pct / 100)

                    if not pos_preds or total_current == 0:
                        continue

                    sec_pct = ((total_predicted / total_current) - 1) * 100
                    sec_dir = _determine_direction(sec_pct)

                    # Aggregate CIs (weighted by position value)
                    ci80_l = sum(p.ci_80[0] for p in pos_preds) / len(pos_preds)
                    ci80_u = sum(p.ci_80[1] for p in pos_preds) / len(pos_preds)
                    ci95_l = min(p.ci_95[0] for p in pos_preds)
                    ci95_u = max(p.ci_95[1] for p in pos_preds)

                    confidences = [p.confidence_level for p in pos_preds]
                    if confidences.count("HIGH") > len(confidences) / 2:
                        sec_conf = "HIGH"
                    elif confidences.count("LOW") > len(confidences) / 2:
                        sec_conf = "LOW"
                    else:
                        sec_conf = "MEDIUM"

                    top_driver = pos_preds[0].factors[0] if pos_preds[0].factors else "N/A"

                    sp = SectorPrediction(
                        sector=sector, horizon=h_name,
                        current_value_eur=round(total_current, 2),
                        predicted_value_eur=round(total_predicted, 2),
                        predicted_change_pct=round(sec_pct, 2),
                        direction=sec_dir, confidence_level=sec_conf,
                        ci_80=(round(ci80_l, 2), round(ci80_u, 2)),
                        ci_95=(round(ci95_l, 2), round(ci95_u, 2)),
                        top_driver=top_driver,
                        position_predictions=pos_preds,
                    )
                    new_sec_preds.setdefault(sector, {})[h_name] = sp

                    if geo_states_override is None:
                        target_date = datetime.now(timezone.utc) + timedelta(days=h_days)
                        save_prediction(
                            ticker=None, sector=sector, level="SECTOR",
                            horizon=h_name, target_date=target_date,
                            current_price_eur=total_current,
                            predicted_price_eur=total_predicted,
                            predicted_change_pct=sec_pct,
                            direction=sec_dir, confidence_level=sec_conf,
                            ci_80=(ci80_l, ci80_u), ci_95=(ci95_l, ci95_u),
                            model_used="SECTOR_AGG",
                            factors_json=json.dumps([top_driver]),
                        )
                except Exception as e:
                    logger.error("Sector prediction failed for %s %s: %s", sector, h_name, e)

        # Portfolio-level aggregation
        for h_name, h_days in PREDICTION_HORIZONS.items():
            try:
                all_sec_preds = []
                total_current = 0
                total_predicted = 0

                for sector in SECTOR_CONFIG:
                    sp = new_sec_preds.get(sector, {}).get(h_name)
                    if sp:
                        all_sec_preds.append(sp)
                        total_current += sp.current_value_eur
                        total_predicted += sp.predicted_value_eur

                if not all_sec_preds or total_current == 0:
                    continue

                port_pct = ((total_predicted / total_current) - 1) * 100
                port_pnl = total_predicted - total_current
                port_dir = _determine_direction(port_pct)

                ci80_l = sum(s.ci_80[0] for s in all_sec_preds) / len(all_sec_preds)
                ci80_u = sum(s.ci_80[1] for s in all_sec_preds) / len(all_sec_preds)
                ci95_l = min(s.ci_95[0] for s in all_sec_preds)
                ci95_u = max(s.ci_95[1] for s in all_sec_preds)

                confs = [s.confidence_level for s in all_sec_preds]
                if confs.count("HIGH") > len(confs) / 2:
                    overall_conf = "HIGH"
                elif confs.count("LOW") > len(confs) / 2:
                    overall_conf = "LOW"
                else:
                    overall_conf = "MEDIUM"

                # Risk summary
                risk_parts = []
                for sp in all_sec_preds:
                    if abs(sp.predicted_change_pct) > 3:
                        risk_parts.append(f"{sp.sector} {sp.direction} {sp.predicted_change_pct:+.1f}%")
                risk_summary = "; ".join(risk_parts) if risk_parts else "No significant sector moves expected"

                pp = PortfolioPrediction(
                    horizon=h_name,
                    current_value_eur=round(total_current, 2),
                    predicted_value_eur=round(total_predicted, 2),
                    predicted_change_pct=round(port_pct, 2),
                    predicted_pnl_eur=round(port_pnl, 2),
                    direction=port_dir, overall_confidence=overall_conf,
                    ci_80=(round(ci80_l, 2), round(ci80_u, 2)),
                    ci_95=(round(ci95_l, 2), round(ci95_u, 2)),
                    sector_predictions=all_sec_preds,
                    risk_summary=risk_summary,
                )
                new_port_preds[h_name] = pp

                if geo_states_override is None:
                    target_date = datetime.now(timezone.utc) + timedelta(days=h_days)
                    save_prediction(
                        ticker=None, sector=None, level="PORTFOLIO",
                        horizon=h_name, target_date=target_date,
                        current_price_eur=total_current,
                        predicted_price_eur=total_predicted,
                        predicted_change_pct=port_pct,
                        direction=port_dir, confidence_level=overall_conf,
                        ci_80=(ci80_l, ci80_u), ci_95=(ci95_l, ci95_u),
                        model_used="PORTFOLIO_AGG",
                        factors_json=json.dumps([risk_summary]),
                    )
            except Exception as e:
                logger.error("Portfolio prediction failed for %s: %s", h_name, e)

        with self._lock:
            self._position_predictions = new_pos_preds
            self._sector_predictions = new_sec_preds
            self._portfolio_predictions = new_port_preds
            self._current_regime = regime
            self._last_refresh = datetime.now(timezone.utc)

        logger.info("Predictions refreshed: %d positions, %d sectors, %d horizons",
                     len(new_pos_preds), len(new_sec_preds), len(new_port_preds))

    def get_position_prediction(self, ticker: str, horizon: str) -> Optional[PricePrediction]:
        with self._lock:
            return self._position_predictions.get(ticker, {}).get(horizon)

    def get_all_position_predictions(self) -> Dict[str, Dict[str, PricePrediction]]:
        with self._lock:
            return dict(self._position_predictions)

    def get_sector_prediction(self, sector: str, horizon: str) -> Optional[SectorPrediction]:
        with self._lock:
            return self._sector_predictions.get(sector, {}).get(horizon)

    def get_all_sector_predictions(self) -> Dict[str, Dict[str, SectorPrediction]]:
        with self._lock:
            return dict(self._sector_predictions)

    def get_portfolio_prediction(self, horizon: str) -> Optional[PortfolioPrediction]:
        with self._lock:
            return self._portfolio_predictions.get(horizon)

    def get_all_portfolio_predictions(self) -> Dict[str, PortfolioPrediction]:
        with self._lock:
            return dict(self._portfolio_predictions)

    def last_refresh_time(self) -> Optional[datetime]:
        with self._lock:
            return self._last_refresh

    def run_scenario(self, geo_states: Dict[str, str]) -> Dict[str, PortfolioPrediction]:
        """Run predictions with overridden geo states (for what-if analysis)."""
        # Save current state
        old_pos = self._position_predictions
        old_sec = self._sector_predictions
        old_port = self._portfolio_predictions

        self.refresh_all(geo_states_override=geo_states)

        with self._lock:
            scenario_results = dict(self._portfolio_predictions)
            # Restore original
            self._position_predictions = old_pos
            self._sector_predictions = old_sec
            self._portfolio_predictions = old_port

        return scenario_results

    def start_background_refresh(self) -> None:
        if self._running:
            return
        self._running = True

        def _loop():
            time.sleep(10)  # Initial delay to let price engine populate
            while self._running:
                try:
                    self.refresh_all()
                except Exception as e:
                    logger.error("Prediction refresh error: %s", e)
                time.sleep(PREDICTION_REFRESH_INTERVAL)

        t = threading.Thread(target=_loop, daemon=True, name="prediction-refresh")
        t.start()
        logger.info("Prediction engine background refresh started (interval=%ds)",
                     PREDICTION_REFRESH_INTERVAL)


# ─────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────

_instance: Optional[PredictionEngine] = None
_instance_lock = threading.Lock()


def get_prediction_engine() -> PredictionEngine:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PredictionEngine()
    return _instance
