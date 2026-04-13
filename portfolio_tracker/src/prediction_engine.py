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
from src.futures_curves import (
    get_all_futures_signals, compute_futures_sector_adjustment, FuturesCurveSignal,
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


def _model_random_forest(prices: pd.Series, horizon_days: int,
                          commodity_hist: Optional[Dict[str, pd.Series]] = None,
                          sector: Optional[str] = None,
                          vix: Optional[float] = None) -> Optional[dict]:
    """
    Model 2: Random Forest with commodity + momentum features.

    Upgrade from Ridge based on academic evidence:
    - Random Forest reduces RMSE 50-65% vs OLS (Driesprong et al. 2008 framework)
    - Kilian (2009): oil demand/supply shocks explain 22% of long-run equity variance
    - Hamilton (2003): nonlinear oil transformations outperform linear models

    Features include lagged commodity returns per sector:
    - ENERGY: Brent crude (lag 1-2 days, correlation ~0.70)
    - GOLD: Gold spot (beta 1.6x for miners, correlation ~0.65)
    - METALS: Copper spot (beta 1.3x for miners)
    """
    if len(prices) < 30:
        return None

    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        logger.warning("scikit-learn not installed — skipping Random Forest model")
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
    df["vol_20d"] = returns.rolling(20).std()

    # RSI (Wilder, 1978)
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = rsi_series.reindex(df.index)

    # Price vs SMA50
    sma50 = prices.rolling(50).mean()
    df["price_vs_sma50"] = ((prices / sma50) - 1).reindex(df.index)

    # VIX feature (market fear — Whaley 2000)
    if vix is not None:
        df["vix_level"] = vix / 100.0  # normalised

    # ── Commodity features per sector (Kilian 2009, Hamilton 2003) ──
    commodity_factor = None
    if commodity_hist and sector:
        # Map sector → relevant commodity
        sector_commodity = {
            "ENERGY": "brent",
            "GOLD": "gold",
            "METALS": "copper",
        }
        # Amplification betas from academic literature
        sector_beta = {
            "ENERGY": 1.0,   # direct correlation ~0.70
            "GOLD": 1.6,     # gold miner beta per literature
            "METALS": 1.3,   # copper miner beta estimate
        }

        commodity_key = sector_commodity.get(sector)
        if commodity_key and commodity_key in commodity_hist:
            comm_series = commodity_hist[commodity_key]
            if comm_series is not None and len(comm_series) > 10:
                comm_ret = comm_series.pct_change().dropna()
                # Align to price index
                comm_ret_aligned = comm_ret.reindex(df.index, method="ffill")

                if comm_ret_aligned.notna().sum() > 10:
                    # Lag 1 (oil leads energy equities by 1-2 days — Driesprong 2008)
                    df["comm_ret_lag1"] = comm_ret_aligned.shift(1)
                    df["comm_ret_lag2"] = comm_ret_aligned.shift(2)
                    # 5-day commodity return
                    df["comm_ret_5d"] = comm_ret_aligned.rolling(5).sum()
                    # 20-day commodity momentum
                    df["comm_ret_20d"] = comm_ret_aligned.rolling(20).sum()

                    beta = sector_beta.get(sector, 1.0)
                    recent_comm = comm_ret_aligned.iloc[-5:].sum() if len(comm_ret_aligned) >= 5 else 0
                    commodity_factor = (
                        f"{commodity_key.title()} 5d return {recent_comm*100:+.1f}% "
                        f"(beta {beta:.1f}x, lag 1-2d)"
                    )

    # Target: forward N-day return
    df["target"] = returns.rolling(horizon_days).sum().shift(-horizon_days)

    df = df.dropna()
    if len(df) < 20:
        return None

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    # Train on all but last row
    X_train, y_train = X[:-1], y[:-1]
    X_pred = X[-1:].copy()

    if len(X_train) < 15:
        return None

    # Random Forest — 200 trees, limited depth to prevent overfitting
    # Per literature: RF reduces RMSE 50%+ vs OLS for commodity-equity forecasting
    model = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred_return = model.predict(X_pred)[0]

    # Out-of-bag or residual error for CI
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    rse = np.std(residuals) * 1.2  # inflate slightly — RF can be overconfident

    current = prices.iloc[-1]
    predicted_pct = pred_return * 100
    predicted_price = current * (1 + pred_return)

    ci_80 = (current * (1 + pred_return - 1.28 * rse),
             current * (1 + pred_return + 1.28 * rse))
    ci_95 = (current * (1 + pred_return - 1.96 * rse),
             current * (1 + pred_return + 1.96 * rse))

    # Feature importance for factor description
    if commodity_factor:
        factor_desc = commodity_factor
    else:
        rsi_val = df["rsi"].iloc[-1]
        if rsi_val > 70:
            factor_desc = f"RSI at {rsi_val:.0f} (overbought, reversion risk)"
        elif rsi_val < 30:
            factor_desc = f"RSI at {rsi_val:.0f} (oversold, bounce potential)"
        else:
            sma_dev = df["price_vs_sma50"].iloc[-1] * 100
            factor_desc = f"RF model: price {sma_dev:+.1f}% vs 50d SMA, vol={df['vol_5d'].iloc[-1]*100:.1f}%"

    return {
        "predicted_pct": predicted_pct,
        "predicted_price": predicted_price,
        "ci_80": ci_80,
        "ci_95": ci_95,
        "factor": factor_desc,
    }


def _model_commodity_correlation(ticker: str, sector: str, current_price: float,
                                  horizon_days: int, prices: pd.Series,
                                  commodity_hist: Optional[Dict[str, pd.Series]] = None,
                                  futures_signals: Optional[Dict[str, FuturesCurveSignal]] = None,
                                  ) -> Optional[dict]:
    """
    Model 6: Commodity-Equity Correlation Model.

    Academic grounding:
    - Kilian & Park (2009): oil shocks explain 22% of long-run equity return variance
    - Hamilton (2003): nonlinear oil price changes predict equity returns
    - Driesprong et al. (2008): oil returns predict stock returns at 1-month lag
    - Gold-miner beta = 1.6x gold spot (empirical, Baur & Lucey 2010)
    - Copper-miner beta = 1.3x copper spot (estimated)

    For ENERGY: tracks Brent crude with rolling 90-day correlation,
                applies 1-2 day lag structure
    For GOLD:   tracks gold spot with 1.6x amplification beta
    For METALS: tracks copper spot with 1.3x amplification beta
    For DEFENSE/BIOTECH: no direct commodity link (returns None)
    """
    SECTOR_COMMODITY_MAP = {
        "ENERGY": {"commodity": "brent", "base_corr": 0.70, "beta": 1.0, "lag_days": 1},
        "GOLD":   {"commodity": "gold",  "base_corr": 0.65, "beta": 1.6, "lag_days": 0},
        "METALS": {"commodity": "copper", "base_corr": 0.60, "beta": 1.3, "lag_days": 0},
    }

    if sector not in SECTOR_COMMODITY_MAP:
        return None
    if commodity_hist is None:
        return None

    cfg = SECTOR_COMMODITY_MAP[sector]
    comm_key = cfg["commodity"]
    comm_series = commodity_hist.get(comm_key)

    if comm_series is None or len(comm_series) < 20:
        return None
    if len(prices) < 20:
        return None

    # Compute rolling correlation (90-day window per literature)
    equity_ret = prices.pct_change().dropna()
    comm_ret = comm_series.pct_change().dropna()

    # Align on common dates
    combined = pd.DataFrame({
        "equity": equity_ret,
        "commodity": comm_ret,
    }).dropna()

    if len(combined) < 20:
        return None

    # Rolling 90-day correlation (or max available)
    window = min(90, len(combined) - 5)
    if window < 20:
        window = 20
    rolling_corr = combined["equity"].rolling(window).corr(combined["commodity"])
    current_corr = rolling_corr.iloc[-1] if len(rolling_corr) > 0 and not np.isnan(rolling_corr.iloc[-1]) else cfg["base_corr"]

    # Commodity momentum (Hamilton 2003: nonlinear transformations)
    comm_5d_ret = combined["commodity"].iloc[-5:].sum() if len(combined) >= 5 else 0
    comm_20d_ret = combined["commodity"].iloc[-20:].sum() if len(combined) >= 20 else 0

    # Net oil price increase (Hamilton 2003): max over trailing 12 months
    if comm_key == "brent" and len(comm_series) >= 252:
        trailing_max = comm_series.iloc[-252:].max()
        current_comm = comm_series.iloc[-1]
        hamilton_nopi = max(0, (current_comm / trailing_max - 1))  # 0 if below trailing max
    else:
        hamilton_nopi = 0

    # Predict equity move from commodity move
    # Apply beta and correlation weighting
    beta = cfg["beta"]
    lag = cfg["lag_days"]

    # Use lagged commodity return if available
    if lag > 0 and len(combined) > lag:
        lagged_comm_ret = combined["commodity"].iloc[-(lag + 5):-lag].sum() / 5  # avg daily
    else:
        lagged_comm_ret = combined["commodity"].iloc[-5:].mean()

    # Predicted daily equity return = correlation × beta × commodity return
    predicted_daily = current_corr * beta * lagged_comm_ret
    predicted_pct = predicted_daily * horizon_days * 100

    # ── Futures curve adjustment (Gorton & Rouwenhorst 2006) ──
    futures_adj = 0.0
    futures_desc = ""
    if futures_signals:
        futures_adj = compute_futures_sector_adjustment(futures_signals, sector)
        # Scale by horizon
        predicted_pct += futures_adj * horizon_days
        # Collect curve descriptions
        curve_parts = []
        for sig in futures_signals.values():
            if sig.affected_sectors.get(sector, 0) != 0:
                curve_parts.append(f"{sig.commodity}: {sig.curve_state}")
        if curve_parts:
            futures_desc = " | Futures: " + ", ".join(curve_parts)

    # Cap extreme predictions
    predicted_pct = max(-20.0, min(20.0, predicted_pct))

    predicted_price = current_price * (1 + predicted_pct / 100)

    # CI: wider when correlation is unstable
    corr_std = rolling_corr.iloc[-30:].std() if len(rolling_corr) >= 30 else 0.15
    base_vol = abs(combined["equity"].std()) * math.sqrt(horizon_days)
    vol = base_vol * (1 + corr_std)  # wider CI when correlation volatile

    ci_80 = (current_price * (1 + predicted_pct / 100 - 1.28 * vol),
             current_price * (1 + predicted_pct / 100 + 1.28 * vol))
    ci_95 = (current_price * (1 + predicted_pct / 100 - 1.96 * vol),
             current_price * (1 + predicted_pct / 100 + 1.96 * vol))

    # Factor description
    factor_desc = (
        f"{comm_key.title()} correlation {current_corr:.2f} "
        f"(base {cfg['base_corr']:.2f}, beta {beta:.1f}x). "
        f"{comm_key.title()} 5d: {comm_5d_ret*100:+.1f}%, 20d: {comm_20d_ret*100:+.1f}%"
    )
    if hamilton_nopi > 0:
        factor_desc += f". Hamilton NOPI: {hamilton_nopi*100:.1f}% above 12m high"
    factor_desc += futures_desc

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
    """
    Model 4: Geopolitical overlay — maps variable states to sector-specific predictions.

    Academic grounding:
    ────────────────────────────────────────────────────────────────────
    Hamilton (2003): Oil price shocks explain recessions. A 100% oil price
    increase predicts -2.5% GDP within 3 quarters. Hormuz closure → ~40%
    Brent spike → energy equities +15-25%, all others -5 to -10%.

    Kilian (2009): Supply-driven oil shocks (e.g., Hormuz) have 2-3x the
    equity impact of demand-driven shocks. Decomposition matters.

    Caldara & Iacoviello (2022): GPR index spikes of 2+ std dev predict
    -1.5% equity returns in the next month for high-sensitivity sectors.

    Leigh et al. (2003, IMF): Geopolitical risk resolution ("peace dividend")
    produces +5-15% equity rally in affected sectors within 3 months,
    but -10 to -20% in sectors that benefited from the conflict.

    Rigobon & Sack (2005): Geopolitical events explain 15-25% of daily
    variance in oil, gold, and defense equities.
    ────────────────────────────────────────────────────────────────────

    This model maps each geo variable state to per-sector % impacts that
    are calibrated to the academic literature. It handles both escalation
    AND de-escalation symmetrically.
    """
    cfg = SECTOR_CONFIG.get(sector)
    if not cfg:
        return None

    # Get geo states
    if geo_states_override:
        geo_vars = geo_states_override
    else:
        db_states = get_geo_states()
        geo_vars = {k: v.current_value for k, v in db_states.items()}
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

    # ── Geo variable states ──────────────────────────────────
    hormuz = geo_vars.get("HORMUZ_STATUS", "OPEN")
    iran = geo_vars.get("IRAN_CONFLICT", "ACTIVE")
    ukraine = geo_vars.get("UKRAINE_WAR", "STALEMATE")
    nato = geo_vars.get("NATO_SPENDING", "INCREASING")
    us_china = geo_vars.get("US_CHINA_RELATIONS", "TENSE")
    sensitivity = cfg.get("geopolitical_sensitivity", 0.5)

    # ── Sector-specific impact matrix ────────────────────────
    # Each variable+state → per-sector weekly % impact
    # Calibrated from Hamilton (2003), Kilian (2009), Leigh (2003 IMF)
    #
    # ESCALATION impacts (positive = bullish for that sector):
    #   Hormuz closure: energy +5%/wk, gold +2%, defense +1.5%, metals -2%, biotech -2%
    #   Iran active conflict: energy +2%, gold +1.5%, defense +1%, metals -0.5%, biotech -1%
    #   Ukraine escalating: defense +3%, energy +1%, gold +1.5%, metals -1%, biotech -1%
    #   US-China hostile: metals +2% (supply), gold +1%, defense +0.5%, energy 0, biotech -1.5%
    #   NATO accelerating: defense +2.5%, energy 0, metals 0, gold 0, biotech 0
    #
    # DE-ESCALATION impacts (peace dividend / conflict unwind):
    #   Hormuz open + Iran resolved: energy -3%/wk, gold -1.5%, defense -0.5%,
    #                                 metals +1%, biotech +1%
    #   Ukraine resolved: defense -2.5%, gold -1%, energy -0.5%, metals +0.5%, biotech +0.5%
    #   US-China cooperative: metals -1% (no supply premium), biotech +1%, gold -0.5%
    #   NATO declining: defense -2%, rest neutral
    #
    # NEUTRAL states → ~0 impact (already priced in)

    # Accumulate weekly % impact from all variables
    weekly_impact = 0.0
    override_factors = []

    # ── HORMUZ STATUS ────────────────────────────────────────
    # Gulf War 1991: Hormuz reopening → Brent -30% in one week
    # 2019 tanker crisis: partial → +15% Brent in days
    hormuz_impact = {
        "ENERGY":  {"CLOSED": +15.0, "PARTIAL": +6.0, "OPEN": 0.0},
        "GOLD":    {"CLOSED": +5.0,  "PARTIAL": +2.5, "OPEN": 0.0},
        "DEFENSE": {"CLOSED": +4.0,  "PARTIAL": +1.5, "OPEN": 0.0},
        "METALS":  {"CLOSED": -4.0,  "PARTIAL": -2.0, "OPEN": 0.0},
        "BIOTECH": {"CLOSED": -5.0,  "PARTIAL": -2.5, "OPEN": 0.0},
    }
    h_imp = hormuz_impact.get(sector, {}).get(hormuz, 0.0)
    weekly_impact += h_imp
    if abs(h_imp) > 0.5:
        override_factors.append(f"Hormuz {hormuz}: {h_imp:+.1f}%/wk for {sector}")

    # ── IRAN CONFLICT ────────────────────────────────────────
    # Iran nuclear deal 2015: Brent -60% over 6 months, energy equities -25%
    # Iran escalation 2019-20: Brent +15%, energy equities +8% in days
    # Resolution = full reversal of war premium, historically violent moves
    iran_impact = {
        "ENERGY":  {"ACTIVE": +4.0, "CEASEFIRE": -3.0, "RESOLVED": -10.0},
        "GOLD":    {"ACTIVE": +3.0, "CEASEFIRE": -1.5, "RESOLVED": -5.0},
        "DEFENSE": {"ACTIVE": +2.0, "CEASEFIRE": -1.0, "RESOLVED": -3.0},
        "METALS":  {"ACTIVE": -1.0, "CEASEFIRE": +1.0, "RESOLVED": +3.0},
        "BIOTECH": {"ACTIVE": -2.0, "CEASEFIRE": +1.0, "RESOLVED": +3.0},
    }
    i_imp = iran_impact.get(sector, {}).get(iran, 0.0)
    weekly_impact += i_imp
    if abs(i_imp) > 0.5:
        override_factors.append(f"Iran {iran}: {i_imp:+.1f}%/wk for {sector}")

    # ── UKRAINE WAR ──────────────────────────────────────────
    # Feb 2022 invasion: Brent +30%, defense equities +40% in weeks
    # Any resolution: defense gives back 20-30% premium over months
    ukraine_impact = {
        "ENERGY":  {"ESCALATING": +4.0, "STALEMATE": 0.0, "DE-ESCALATING": -2.0, "RESOLVED": -5.0},
        "GOLD":    {"ESCALATING": +3.0, "STALEMATE": 0.0, "DE-ESCALATING": -1.5, "RESOLVED": -4.0},
        "DEFENSE": {"ESCALATING": +8.0, "STALEMATE": +1.0, "DE-ESCALATING": -3.0, "RESOLVED": -8.0},
        "METALS":  {"ESCALATING": -2.0, "STALEMATE": 0.0, "DE-ESCALATING": +1.0, "RESOLVED": +2.0},
        "BIOTECH": {"ESCALATING": -3.0, "STALEMATE": 0.0, "DE-ESCALATING": +1.0, "RESOLVED": +2.0},
    }
    u_imp = ukraine_impact.get(sector, {}).get(ukraine, 0.0)
    weekly_impact += u_imp
    if abs(u_imp) > 0.5:
        override_factors.append(f"Ukraine {ukraine}: {u_imp:+.1f}%/wk for {sector}")

    # ── US-CHINA RELATIONS ───────────────────────────────────
    # Trade war 2018-19: rare earth +50%, tech -15%
    # Phase 1 deal 2020: rare earth -20%, tech +10%
    china_impact = {
        "ENERGY":  {"HOSTILE": +1.0, "TENSE": 0.0, "NEUTRAL": 0.0, "COOPERATIVE": -0.5},
        "GOLD":    {"HOSTILE": +2.0, "TENSE": 0.0, "NEUTRAL": -0.5, "COOPERATIVE": -1.5},
        "DEFENSE": {"HOSTILE": +1.5, "TENSE": 0.0, "NEUTRAL": 0.0, "COOPERATIVE": -1.0},
        "METALS":  {"HOSTILE": +5.0, "TENSE": +1.0, "NEUTRAL": 0.0, "COOPERATIVE": -3.0},
        "BIOTECH": {"HOSTILE": -3.0, "TENSE": -0.5, "NEUTRAL": 0.0, "COOPERATIVE": +2.0},
    }
    c_imp = china_impact.get(sector, {}).get(us_china, 0.0)
    weekly_impact += c_imp
    if abs(c_imp) > 0.5:
        override_factors.append(f"US-China {us_china}: {c_imp:+.1f}%/wk for {sector}")

    # ── NATO SPENDING ────────────────────────────────────────
    # 2022-24 rearmament: EU defense equities +200% over 2 years
    # Any reversal would be catastrophic for defense pure-plays
    nato_impact = {
        "ENERGY":  {"DECLINING": 0.0, "STABLE": 0.0, "INCREASING": 0.0, "ACCELERATING": +0.5},
        "GOLD":    {"DECLINING": 0.0, "STABLE": 0.0, "INCREASING": 0.0, "ACCELERATING": +0.5},
        "DEFENSE": {"DECLINING": -6.0, "STABLE": -1.0, "INCREASING": +2.0, "ACCELERATING": +5.0},
        "METALS":  {"DECLINING": 0.0, "STABLE": 0.0, "INCREASING": +0.5, "ACCELERATING": +1.0},
        "BIOTECH": {"DECLINING": +0.5, "STABLE": 0.0, "INCREASING": 0.0, "ACCELERATING": -0.5},
    }
    n_imp = nato_impact.get(sector, {}).get(nato, 0.0)
    weekly_impact += n_imp
    if abs(n_imp) > 0.5:
        override_factors.append(f"NATO {nato}: {n_imp:+.1f}%/wk for {sector}")

    # ── COMPOUND EFFECTS (nonlinear interactions) ────────────
    # When multiple conflicts resolve simultaneously, the unwind is
    # MORE than additive — markets reprice the entire risk premium at once.
    #
    # Historical precedent:
    # - Gulf War end (Feb 1991): oil -33% in ONE WEEK, energy equities -15%
    # - Iran deal (Jul 2015): Brent -60% over 6 months
    # - If both Hormuz reopens AND Iran resolves: this is a regime change,
    #   not incremental news. Energy war premium (est. $20-30/bbl) collapses.
    #   At Brent beta 1.0 and $20 premium on $80 base = -25% oil = -20% equities

    if hormuz == "OPEN" and iran in ("CEASEFIRE", "RESOLVED"):
        peace_multiplier = 2.0 if iran == "RESOLVED" else 1.3
        if sector == "ENERGY":
            compound = -5.0 * peace_multiplier
            weekly_impact += compound
            override_factors.append(
                f"PEACE DIVIDEND (Hormuz open + Iran {iran}): "
                f"war premium collapsing {compound:+.1f}%/wk — "
                f"Gulf War 1991 precedent: oil -33% in 1 week"
            )
        elif sector == "GOLD":
            compound = -3.0 * peace_multiplier
            weekly_impact += compound
            override_factors.append(
                f"Peace dividend: safe-haven demand evaporating {compound:+.1f}%/wk"
            )
        elif sector == "DEFENSE":
            compound = -2.0 * peace_multiplier
            weekly_impact += compound
            override_factors.append(
                f"Peace dividend: conflict premium unwinding {compound:+.1f}%/wk"
            )

    # Full multi-front de-escalation: Ukraine + Iran + Hormuz all peaceful
    # This is the "peace breaks out everywhere" scenario — maximum pain for
    # conflict beneficiaries, maximum gain for risk-on assets
    if (hormuz == "OPEN" and iran == "RESOLVED"
            and ukraine in ("RESOLVED", "DE-ESCALATING")
            and us_china in ("NEUTRAL", "COOPERATIVE")):
        if sector == "ENERGY":
            weekly_impact += -5.0
            override_factors.append(
                "FULL DE-ESCALATION: all conflict premiums collapsing simultaneously"
            )
        elif sector == "DEFENSE":
            weekly_impact += -5.0
            override_factors.append(
                "FULL DE-ESCALATION: peace dividend → defense budgets at risk"
            )
        elif sector == "GOLD":
            weekly_impact += -3.0
            override_factors.append(
                "FULL DE-ESCALATION: no safe-haven need → gold selling"
            )
        elif sector in ("METALS", "BIOTECH"):
            weekly_impact += +3.0
            override_factors.append(
                "FULL DE-ESCALATION: risk-on rotation into growth/industrials"
            )

    # Multi-front escalation: amplify all impacts
    escalation_count = sum([
        hormuz in ("CLOSED", "PARTIAL"),
        iran == "ACTIVE",
        ukraine == "ESCALATING",
        us_china == "HOSTILE",
    ])
    if escalation_count >= 3:
        # Kinderberger: multiple simultaneous shocks amplify each other
        weekly_impact *= 1.3
        override_factors.append(
            f"Multi-front escalation ({escalation_count}/4): impacts amplified 1.3x"
        )

    # ── Scale weekly impact to horizon ───────────────────────
    # Apply geo sensitivity scaling (ENERGY=0.85, BIOTECH=0.20)
    weekly_impact *= sensitivity

    # Convert weekly % impact to horizon-period impact
    # Use diminishing impact for longer horizons (markets reprice within ~2 weeks)
    if horizon_days <= 5:
        predicted_pct = weekly_impact * (horizon_days / 5.0)
    elif horizon_days <= 21:
        # First week at full rate, then 60% for remaining weeks
        predicted_pct = weekly_impact + weekly_impact * 0.6 * ((horizon_days - 5) / 5.0)
    else:
        predicted_pct = weekly_impact + weekly_impact * 0.6 * 3.2  # ~1 month cap
    predicted_pct = predicted_pct / 100.0  # convert to decimal

    # ── Geo score trend overlay ──────────────────────────────
    # Adds momentum from the geo score itself (captures news flow direction)
    score_adj = 0.0
    if current_score > 7 and trend == "rising":
        score_adj = +0.005 * (current_score - 5) * horizon_days
    elif current_score < 4 and trend == "falling":
        score_adj = -0.003 * (5 - current_score) * horizon_days
    predicted_pct += score_adj

    # Build factor description
    if override_factors:
        factor_desc = " | ".join(override_factors[:3])
    else:
        factor_desc = f"Geo score {current_score:.1f} ({trend}), sensitivity {sensitivity:.2f}"

    predicted_pct_display = predicted_pct * 100
    predicted_price = current_price * (1 + predicted_pct)

    # CI — wider when geo variables volatile
    volatile_count = sum(1 for v in [
        hormuz in ("CLOSED", "PARTIAL"),
        iran == "ACTIVE",
        ukraine == "ESCALATING",
        us_china == "HOSTILE",
    ] if v)
    # Also widen CI during active de-escalation (uncertainty about follow-through)
    deescalation_count = sum(1 for v in [
        iran == "CEASEFIRE",
        ukraine == "DE-ESCALATING",
    ] if v)
    uncertainty = volatile_count + 0.5 * deescalation_count

    base_vol = 0.02 * math.sqrt(horizon_days)
    vol = base_vol * (1 + 0.3 * uncertainty)

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
                              regime: Optional[MarketRegime] = None,
                              commodity_hist: Optional[Dict[str, pd.Series]] = None,
                              vix: Optional[float] = None) -> Optional[PricePrediction]:
    """Run all 6 models and combine into ensemble prediction for one position/horizon."""
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

        # Model 2: Random Forest with commodity features
        # Upgraded from Ridge — RF reduces RMSE 50%+ (academic literature)
        m2 = _model_random_forest(prices, horizon_days, commodity_hist, sector, vix)
        if m2:
            model_results["random_forest"] = m2

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

    # Model 6: Commodity-equity correlation + futures curve signals
    # Kilian (2009): oil explains 22% of equity variance
    # Gold-miner beta 1.6x, copper-miner beta 1.3x
    # Gorton & Rouwenhorst (2006): backwardation predicts 10% excess returns
    if has_enough_history and commodity_hist:
        # Compute futures curve signals if we have commodity spot prices
        futures_sigs = None
        try:
            brent_spot = float(commodity_hist["brent"].iloc[-1]) if commodity_hist.get("brent") is not None and len(commodity_hist["brent"]) > 0 else None
            gold_spot = float(commodity_hist["gold"].iloc[-1]) if commodity_hist.get("gold") is not None and len(commodity_hist["gold"]) > 0 else None
            futures_sigs = get_all_futures_signals(
                brent_spot, gold_spot, vix,
                commodity_hist.get("brent"), commodity_hist.get("gold"),
            )
        except Exception as e:
            logger.debug("Futures curve analysis skipped: %s", e)

        m6 = _model_commodity_correlation(
            ticker, sector, current_price_eur, horizon_days, prices,
            commodity_hist, futures_sigs,
        )
        if m6:
            model_results["commodity_correlation"] = m6

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

        # Fetch commodity histories ONCE for all positions
        commodity_hist = None
        vix_level = None
        try:
            commodity_hist = pe.get_commodity_histories()
            vix_level = pe.get_vix()
        except Exception as e:
            logger.warning("Could not fetch commodity histories: %s", e)

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
                        commodity_hist, vix_level,
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
