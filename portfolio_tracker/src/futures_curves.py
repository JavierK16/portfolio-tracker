"""
futures_curves.py — Futures curve analysis for commodity and volatility markets.

Extracts forward-looking signals from the shape of futures curves:

Academic grounding:
- Gorton & Rouwenhorst (2006): backwardated commodities earn ~10% annualized
  excess returns vs ~1% for contango. Curve shape is a strong predictor.
- Samuelson (1965) hypothesis: front-month futures are more volatile and
  informative than deferred contracts — the curve shape encodes information.
- VIX term structure inversion (spot > 3-month) predicts negative S&P
  returns ~65% of the time over the next month.

Signals produced:
- Oil curve: backwardation/contango spread → ENERGY equity signal
- Gold curve: steepness vs cost-of-carry → inflation/fear signal
- VIX curve: term structure slope → regime indicator for all sectors
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FuturesCurveSignal:
    """Encodes a signal from futures curve shape."""
    commodity: str          # "oil", "gold", "vix"
    curve_state: str        # "BACKWARDATION", "CONTANGO", "FLAT", "INVERTED", "NORMAL"
    spread_pct: float       # front vs back spread as % of front price
    signal_strength: float  # -1.0 (very bearish) to +1.0 (very bullish)
    affected_sectors: Dict[str, float]  # sector -> adjustment multiplier
    description: str


def analyse_oil_curve(brent_spot: Optional[float],
                       brent_history: Optional["pd.Series"] = None) -> Optional[FuturesCurveSignal]:
    """
    Analyse oil futures curve shape.

    Backwardation (spot > futures): tight supply, bullish energy equities.
    Gorton & Rouwenhorst (2006): backwardated commodities earn ~10% excess.
    Contango (spot < futures): oversupply, bearish energy equities.

    Since we can't easily get the full futures curve from yfinance,
    we approximate using the price trend:
    - If spot is above 20-day SMA → backwardation proxy (supply tightness)
    - If spot is below 20-day SMA → contango proxy (oversupply)
    - The spread magnitude indicates strength.
    """
    if brent_spot is None or brent_spot <= 0:
        return None

    import pandas as pd
    if brent_history is not None and len(brent_history) >= 20:
        sma_20 = float(brent_history.iloc[-20:].mean())
        sma_60 = float(brent_history.iloc[-60:].mean()) if len(brent_history) >= 60 else sma_20
    else:
        return None

    if sma_20 <= 0:
        return None

    # Spread: how far spot is from recent average (proxy for curve shape)
    spread_pct = (brent_spot - sma_20) / sma_20 * 100
    trend_pct = (sma_20 - sma_60) / sma_60 * 100 if sma_60 > 0 else 0

    if spread_pct > 3.0:  # spot well above recent average → backwardation proxy
        curve_state = "BACKWARDATION"
        # Gorton & Rouwenhorst: backwardation = bullish for commodity equities
        signal_strength = min(1.0, spread_pct / 10.0)
        desc = (f"Oil in backwardation (spot {spread_pct:+.1f}% vs 20d avg). "
                f"Gorton-Rouwenhorst: bullish for energy equities")
        sector_adj = {
            "ENERGY": +0.5 * signal_strength,
            "GOLD": +0.1 * signal_strength,   # inflation hedge
            "METALS": -0.1 * signal_strength,  # demand destruction fear
            "DEFENSE": 0.0,
            "BIOTECH": -0.15 * signal_strength,
        }
    elif spread_pct < -3.0:  # spot below average → contango proxy
        curve_state = "CONTANGO"
        signal_strength = max(-1.0, spread_pct / 10.0)
        desc = (f"Oil in contango (spot {spread_pct:+.1f}% vs 20d avg). "
                f"Oversupply signal — bearish energy")
        sector_adj = {
            "ENERGY": +0.5 * signal_strength,  # negative
            "GOLD": -0.05,
            "METALS": +0.1,   # lower input costs
            "DEFENSE": 0.0,
            "BIOTECH": +0.05,
        }
    else:
        curve_state = "FLAT"
        signal_strength = 0.0
        desc = f"Oil curve flat (spot {spread_pct:+.1f}% vs 20d avg). Neutral"
        sector_adj = {s: 0.0 for s in ["ENERGY", "GOLD", "METALS", "DEFENSE", "BIOTECH"]}

    return FuturesCurveSignal(
        commodity="oil",
        curve_state=curve_state,
        spread_pct=round(spread_pct, 2),
        signal_strength=round(signal_strength, 3),
        affected_sectors=sector_adj,
        description=desc,
    )


def analyse_gold_curve(gold_spot: Optional[float],
                        gold_history: Optional["pd.Series"] = None) -> Optional[FuturesCurveSignal]:
    """
    Analyse gold futures curve.

    Normal gold contango = ~0.3-0.5%/month (cost of carry).
    Steeper than normal = institutional hedging demand (fear).
    Flattening/backwardation = extremely bullish (physical shortage).
    """
    if gold_spot is None or gold_spot <= 0:
        return None

    import pandas as pd
    if gold_history is not None and len(gold_history) >= 20:
        sma_20 = float(gold_history.iloc[-20:].mean())
    else:
        return None

    if sma_20 <= 0:
        return None

    spread_pct = (gold_spot - sma_20) / sma_20 * 100

    if spread_pct > 2.0:
        curve_state = "BACKWARDATION"
        signal_strength = min(1.0, spread_pct / 8.0)
        desc = f"Gold backwardation proxy ({spread_pct:+.1f}%). Physical demand surge — very bullish"
        sector_adj = {"GOLD": +0.6 * signal_strength, "ENERGY": +0.1, "METALS": +0.1,
                      "DEFENSE": 0.0, "BIOTECH": -0.1}
    elif spread_pct < -2.0:
        curve_state = "CONTANGO"
        signal_strength = max(-1.0, spread_pct / 8.0)
        desc = f"Gold contango ({spread_pct:+.1f}%). Risk-on environment — gold selling"
        sector_adj = {"GOLD": +0.6 * signal_strength, "ENERGY": 0.0, "METALS": 0.0,
                      "DEFENSE": 0.0, "BIOTECH": +0.1}
    else:
        curve_state = "NORMAL"
        signal_strength = 0.0
        desc = f"Gold normal carry ({spread_pct:+.1f}%). Neutral"
        sector_adj = {s: 0.0 for s in ["ENERGY", "GOLD", "METALS", "DEFENSE", "BIOTECH"]}

    return FuturesCurveSignal(
        commodity="gold", curve_state=curve_state,
        spread_pct=round(spread_pct, 2),
        signal_strength=round(signal_strength, 3),
        affected_sectors=sector_adj, description=desc,
    )


def analyse_vix_regime(vix_spot: Optional[float]) -> Optional[FuturesCurveSignal]:
    """
    VIX regime analysis.

    VIX term structure (spot vs typical 3-month):
    - Normal: spot < 20, upward sloping. Markets calm.
    - Elevated: spot 20-30. Uncertainty but not panic.
    - Inverted/Crisis: spot > 30. Predicts negative equity returns
      ~65% of the time over next month.

    We approximate 3-month VIX as the long-run mean (~19).
    """
    if vix_spot is None:
        return None

    vix_mean = 19.0  # long-run average
    spread = vix_spot - vix_mean
    spread_pct = (vix_spot / vix_mean - 1) * 100

    if vix_spot > 35:
        curve_state = "INVERTED"
        signal_strength = -min(1.0, (vix_spot - 35) / 20.0)
        desc = (f"VIX at {vix_spot:.0f} — crisis regime. "
                f"Inverted term structure predicts negative returns 65% of the time")
        sector_adj = {
            "ENERGY": -0.3, "GOLD": +0.3, "METALS": -0.4,
            "DEFENSE": -0.1, "BIOTECH": -0.5,
        }
    elif vix_spot > 25:
        curve_state = "ELEVATED"
        signal_strength = -0.3
        desc = f"VIX at {vix_spot:.0f} — elevated fear. Risk premiums expanding"
        sector_adj = {
            "ENERGY": -0.1, "GOLD": +0.2, "METALS": -0.2,
            "DEFENSE": 0.0, "BIOTECH": -0.3,
        }
    elif vix_spot < 14:
        curve_state = "COMPLACENT"
        signal_strength = 0.2
        desc = f"VIX at {vix_spot:.0f} — complacent. Low vol = risk-on"
        sector_adj = {
            "ENERGY": +0.05, "GOLD": -0.15, "METALS": +0.1,
            "DEFENSE": 0.0, "BIOTECH": +0.2,
        }
    else:
        curve_state = "NORMAL"
        signal_strength = 0.0
        desc = f"VIX at {vix_spot:.0f} — normal range. No curve signal"
        sector_adj = {s: 0.0 for s in ["ENERGY", "GOLD", "METALS", "DEFENSE", "BIOTECH"]}

    return FuturesCurveSignal(
        commodity="vix", curve_state=curve_state,
        spread_pct=round(spread_pct, 2),
        signal_strength=round(signal_strength, 3),
        affected_sectors=sector_adj, description=desc,
    )


def get_all_futures_signals(brent_spot, gold_spot, vix_spot,
                             brent_hist=None, gold_hist=None) -> Dict[str, FuturesCurveSignal]:
    """Run all futures curve analyses and return signals keyed by commodity."""
    signals = {}

    oil_sig = analyse_oil_curve(brent_spot, brent_hist)
    if oil_sig:
        signals["oil"] = oil_sig

    gold_sig = analyse_gold_curve(gold_spot, gold_hist)
    if gold_sig:
        signals["gold"] = gold_sig

    vix_sig = analyse_vix_regime(vix_spot)
    if vix_sig:
        signals["vix"] = vix_sig

    return signals


def compute_futures_sector_adjustment(futures_signals: Dict[str, FuturesCurveSignal],
                                       sector: str) -> float:
    """
    Aggregate all futures curve signals into a single sector adjustment (%).
    This feeds into the prediction ensemble as an additional factor.
    """
    total_adj = 0.0
    for sig in futures_signals.values():
        total_adj += sig.affected_sectors.get(sector, 0.0)
    return total_adj
