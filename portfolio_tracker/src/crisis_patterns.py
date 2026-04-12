"""
crisis_patterns.py — Historical crisis pattern recognition and cross-sector regime detection.

Encodes academic research on the structural causes and early-warning indicators
of major market crises, then scans current conditions for analogous patterns.

Academic sources informing this model:
────────────────────────────────────────────────────────────────────────────────
1929 Great Crash
  - Galbraith, J.K. (1954) "The Great Crash, 1929"
  - Rappoport & White (1993) "Was the Crash of 1929 Expected?" AER
  Key pattern: Speculative mania → margin leverage spike → monetary tightening
  → liquidity withdrawal → correlated asset collapse

1973 Oil Crisis
  - Hamilton, J.D. (1983) "Oil and the Macroeconomy since WWII" JPE
  - Barsky & Kilian (2004) "Oil and the Macroeconomy Since the 1970s" JEP
  Key pattern: Geopolitical supply shock → energy price spike → stagflation
  → broad equity drawdown (except energy/gold)

2000 Dotcom Bubble
  - Shiller, R.J. (2000) "Irrational Exuberance" (Princeton)
  - Ofek & Richardson (2003) "DotCom Mania" JF
  Key pattern: Extreme valuations (P/E expansion) → rate hike cycle
  → speculative sector collapse → rotation to value/commodities

2008 Financial Crisis
  - Brunnermeier (2009) "Deciphering the Liquidity and Credit Crunch" JEP
  - Gorton & Metrick (2012) "Securitized Banking and the Run on Repo" JFE
  - Reinhart & Rogoff (2009) "This Time Is Different" (Princeton)
  Key pattern: Credit spread widening → yield curve inversion → VIX spike
  → correlated selloff across all risk assets → gold/treasuries outperform

2020 COVID Crash
  - Baker et al. (2020) "The Unprecedented Stock Market Reaction to COVID-19" RFS
  - Ramelli & Wagner (2020) "Feverish Stock Price Reactions to COVID-19" RFS
  Key pattern: External shock → VIX spike >80 → liquidity crisis
  → correlation-1 selloff → rapid policy response → V-recovery
  → supply chain disruption → commodity rally (delayed)

Cross-crisis commonalities (Kindleberger & Aliber, "Manias, Panics and Crashes"):
  1. Cross-asset correlations spike toward 1.0 during crises (diversification fails)
  2. Volatility regimes are persistent (high-vol clusters via GARCH effects)
  3. Safe havens (gold) decouple from risk assets in crisis but with a lag
  4. Energy supply shocks propagate to all sectors within 2-6 months
  5. Geopolitical escalation amplifies financial contagion
────────────────────────────────────────────────────────────────────────────────
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CRISIS PATTERN DEFINITIONS
# ─────────────────────────────────────────────────────────────

@dataclass
class CrisisPattern:
    """Encodes the signature of a historical crisis type."""
    name: str
    year: str
    indicators: Dict[str, Tuple[str, float]]  # indicator_name -> (condition, threshold)
    sector_impacts: Dict[str, float]  # sector -> expected % impact per week
    description: str
    contagion_speed_days: int  # how fast it spreads across sectors


# Each pattern defines what indicators to watch and how sectors were affected
CRISIS_PATTERNS = [
    CrisisPattern(
        name="SUPPLY_SHOCK",
        year="1973/2022",
        indicators={
            "energy_spike_pct": ("above", 30.0),      # energy sector up >30% in 60 days
            "cross_correlation": ("above", 0.70),       # sectors moving together
            "vix_level": ("above", 25.0),               # elevated fear
            "geo_energy_score": ("above", 8.5),          # geopolitical energy stress
        },
        sector_impacts={
            "ENERGY": +4.0,   # major beneficiary
            "GOLD": +2.0,     # safe haven bid
            "DEFENSE": +1.5,  # conflict premium
            "METALS": -1.5,   # demand destruction fear
            "BIOTECH": -2.0,  # risk-off rotation
        },
        description="Energy supply disruption (1973 oil embargo, 2022 Russia). "
                    "Energy surges while demand-sensitive sectors suffer.",
        contagion_speed_days=14,
    ),
    CrisisPattern(
        name="CREDIT_CRISIS",
        year="2008",
        indicators={
            "vix_level": ("above", 35.0),               # high fear
            "cross_correlation": ("above", 0.80),        # everything correlated
            "max_drawdown_pct": ("below", -20.0),        # broad drawdown
            "vol_regime": ("above", 2.5),                # vol way above normal
        },
        sector_impacts={
            "ENERGY": -3.0,   # demand collapse
            "GOLD": +2.5,     # flight to safety
            "DEFENSE": -1.0,  # budget uncertainty
            "METALS": -4.0,   # industrial demand collapse
            "BIOTECH": -3.5,  # risk-off, funding dries up
        },
        description="Credit/liquidity crisis (2008 GFC). Correlation spikes to 1, "
                    "all risk assets fall, only gold survives.",
        contagion_speed_days=5,
    ),
    CrisisPattern(
        name="EXTERNAL_SHOCK",
        year="2020",
        indicators={
            "vix_level": ("above", 45.0),               # extreme fear
            "cross_correlation": ("above", 0.85),        # everything sells
            "drawdown_speed": ("above", 3.0),            # fast decline (%/day)
        },
        sector_impacts={
            "ENERGY": -5.0,   # demand evaporates
            "GOLD": +1.0,     # safe haven (but initially sold for liquidity)
            "DEFENSE": -2.0,  # uncertainty
            "METALS": -3.0,   # China demand uncertainty
            "BIOTECH": +1.0,  # pharma demand (but volatile)
        },
        description="Exogenous shock (COVID, pandemic, black swan). "
                    "Correlation-1 selloff, then rapid V-recovery possible.",
        contagion_speed_days=3,
    ),
    CrisisPattern(
        name="GEOPOLITICAL_ESCALATION",
        year="Ongoing",
        indicators={
            "geo_energy_score": ("above", 9.0),
            "geo_defense_score": ("above", 9.0),
            "vix_level": ("above", 25.0),
            "energy_spike_pct": ("above", 15.0),
        },
        sector_impacts={
            "ENERGY": +3.0,
            "GOLD": +2.5,
            "DEFENSE": +4.0,
            "METALS": +1.0,   # supply disruption premium
            "BIOTECH": -1.5,  # risk-off
        },
        description="Multi-front geopolitical escalation. Defense and energy benefit, "
                    "risk-sensitive sectors rotate out.",
        contagion_speed_days=7,
    ),
    CrisisPattern(
        name="STAGFLATION",
        year="1970s/2022",
        indicators={
            "energy_spike_pct": ("above", 20.0),
            "gold_momentum_pct": ("above", 10.0),       # gold rising (inflation hedge)
            "cross_correlation": ("above", 0.50),
            "vol_regime": ("above", 1.5),
        },
        sector_impacts={
            "ENERGY": +2.0,
            "GOLD": +3.0,    # inflation hedge
            "DEFENSE": +0.5,
            "METALS": +1.0,  # real asset bid
            "BIOTECH": -2.5, # growth/duration assets suffer
        },
        description="Stagflation regime: rising prices + slowing growth. "
                    "Real assets outperform, growth sectors underperform.",
        contagion_speed_days=30,
    ),
]


# ─────────────────────────────────────────────────────────────
# MARKET REGIME DETECTION
# ─────────────────────────────────────────────────────────────

@dataclass
class MarketRegime:
    """Current market regime assessment."""
    regime_name: str          # CALM / ELEVATED / STRESSED / CRISIS
    regime_score: float       # 0-100 (0=calm, 100=crisis)
    vix_level: float
    cross_sector_correlation: float
    vol_regime_z: float       # current vol vs 1-year mean in std devs
    active_patterns: List[Tuple[str, float]]  # (pattern_name, match_score 0-1)
    sector_adjustments: Dict[str, float]      # sector -> % adjustment from patterns
    contagion_risk: float     # 0-1 probability of cross-sector contagion
    factors: List[str]        # human-readable explanations


def compute_cross_sector_correlation(sector_prices: Dict[str, pd.Series],
                                      window: int = 30) -> float:
    """
    Compute average pairwise correlation between sector return series.
    High correlation (>0.7) indicates regime where diversification fails.
    Kindleberger (2005): correlations spike toward 1.0 in crises.
    """
    if len(sector_prices) < 2:
        return 0.0

    returns = {}
    for sector, prices in sector_prices.items():
        if len(prices) >= window:
            ret = prices.pct_change().dropna().iloc[-window:]
            if len(ret) >= 10:
                returns[sector] = ret

    if len(returns) < 2:
        return 0.0

    df = pd.DataFrame(returns)
    corr_matrix = df.corr()

    # Average of upper triangle (excluding diagonal)
    n = len(corr_matrix)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if not np.isnan(val):
                pairs.append(val)

    return float(np.mean(pairs)) if pairs else 0.0


def compute_vol_regime(returns: pd.Series, lookback: int = 252) -> float:
    """
    Current volatility relative to 1-year mean, in standard deviations.
    Based on GARCH-clustering literature (Engle, 1982; Bollerslev, 1986):
    vol regimes are persistent — high vol predicts more high vol.

    Returns z-score: >2 = stressed, >3 = crisis.
    """
    if len(returns) < lookback:
        lookback = len(returns)
    if lookback < 30:
        return 0.0

    rolling_vol = returns.rolling(20).std()
    recent_vol = rolling_vol.iloc[-1]
    mean_vol = rolling_vol.iloc[-lookback:].mean()
    std_vol = rolling_vol.iloc[-lookback:].std()

    if std_vol == 0 or np.isnan(std_vol):
        return 0.0

    return float((recent_vol - mean_vol) / std_vol)


def compute_drawdown_speed(prices: pd.Series, window: int = 10) -> float:
    """Average daily drawdown rate over recent window. Used to detect crash velocity."""
    if len(prices) < window + 1:
        return 0.0
    recent = prices.iloc[-(window + 1):]
    total_change = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
    return abs(total_change / window) if total_change < 0 else 0.0


def detect_market_regime(
    sector_prices: Dict[str, pd.Series],
    vix: Optional[float],
    geo_scores: Dict[str, float],
) -> MarketRegime:
    """
    Assess current market regime by scanning for crisis pattern matches.

    This implements the Kindleberger-Minsky framework:
    Displacement → Boom → Euphoria → Profit-taking → Panic

    And adds Reinhart-Rogoff (2009) early warning indicators:
    - Cross-correlation spike
    - Volatility regime shift
    - Sector drawdown clustering
    """
    vix_level = vix or 15.0  # default calm

    # Compute cross-sector correlation
    cross_corr = compute_cross_sector_correlation(sector_prices, window=30)

    # Compute portfolio-level returns for vol regime
    all_returns = []
    for sector, prices in sector_prices.items():
        if len(prices) > 30:
            all_returns.append(prices.pct_change().dropna())

    if all_returns:
        combined_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        vol_z = compute_vol_regime(combined_returns)
    else:
        combined_returns = pd.Series(dtype=float)
        vol_z = 0.0

    # Compute sector-specific metrics
    sector_60d_returns = {}
    sector_drawdown_speeds = {}
    for sector, prices in sector_prices.items():
        if len(prices) >= 60:
            ret_60d = (prices.iloc[-1] / prices.iloc[-60] - 1) * 100
            sector_60d_returns[sector] = ret_60d
        if len(prices) >= 11:
            sector_drawdown_speeds[sector] = compute_drawdown_speed(prices)

    energy_spike = sector_60d_returns.get("ENERGY", 0)
    gold_momentum = sector_60d_returns.get("GOLD", 0)
    max_drawdown = min(sector_60d_returns.values()) if sector_60d_returns else 0
    max_drawdown_speed = max(sector_drawdown_speeds.values()) if sector_drawdown_speeds else 0

    # Build indicator snapshot
    indicators = {
        "vix_level": vix_level,
        "cross_correlation": cross_corr,
        "vol_regime": vol_z,
        "energy_spike_pct": energy_spike,
        "gold_momentum_pct": gold_momentum,
        "max_drawdown_pct": max_drawdown,
        "drawdown_speed": max_drawdown_speed,
        "geo_energy_score": geo_scores.get("ENERGY", 5.0),
        "geo_defense_score": geo_scores.get("DEFENSE", 5.0),
    }

    # Score each crisis pattern
    active_patterns = []
    for pattern in CRISIS_PATTERNS:
        match_count = 0
        total_indicators = len(pattern.indicators)

        for ind_name, (condition, threshold) in pattern.indicators.items():
            current_val = indicators.get(ind_name, 0)
            if condition == "above" and current_val >= threshold:
                match_count += 1
            elif condition == "below" and current_val <= threshold:
                match_count += 1

        match_score = match_count / total_indicators if total_indicators > 0 else 0
        if match_score >= 0.4:  # At least 40% of indicators match
            active_patterns.append((pattern.name, match_score))

    active_patterns.sort(key=lambda x: x[1], reverse=True)

    # Compute sector adjustments from matched patterns
    # Weight by match score — stronger matches dominate
    sector_adjustments: Dict[str, float] = {s: 0.0 for s in
                                             ["ENERGY", "DEFENSE", "METALS", "GOLD", "BIOTECH"]}
    for pattern_name, match_score in active_patterns:
        pattern = next(p for p in CRISIS_PATTERNS if p.name == pattern_name)
        for sector, impact in pattern.sector_impacts.items():
            # Scale impact by match strength and apply diminishing returns for multiple patterns
            sector_adjustments[sector] += impact * match_score * 0.7

    # Contagion risk: Brunnermeier (2009) — contagion probability rises
    # with correlation and volatility
    contagion_risk = min(1.0, max(0.0,
        0.3 * (cross_corr / 0.8) +
        0.3 * min(1.0, vix_level / 40.0) +
        0.2 * min(1.0, vol_z / 3.0) +
        0.2 * (len(active_patterns) / 3.0)
    ))

    # Overall regime score (0-100)
    regime_score = min(100.0, max(0.0,
        25.0 * min(1.0, vix_level / 30.0) +
        25.0 * min(1.0, cross_corr / 0.7) +
        25.0 * min(1.0, vol_z / 2.5) +
        25.0 * min(1.0, len(active_patterns) / 2.0)
    ))

    # Name the regime
    if regime_score >= 75:
        regime_name = "CRISIS"
    elif regime_score >= 50:
        regime_name = "STRESSED"
    elif regime_score >= 25:
        regime_name = "ELEVATED"
    else:
        regime_name = "CALM"

    # Build human-readable factors
    factors = []
    if vix_level > 25:
        factors.append(f"VIX at {vix_level:.0f} (elevated fear)")
    if cross_corr > 0.6:
        factors.append(f"Cross-sector correlation {cross_corr:.2f} "
                       f"(diversification weakening)")
    if vol_z > 1.5:
        factors.append(f"Volatility {vol_z:.1f} sigma above normal "
                       f"(GARCH clustering)")
    for pname, pscore in active_patterns[:2]:
        pattern = next(p for p in CRISIS_PATTERNS if p.name == pname)
        factors.append(f"Pattern match: {pname} ({pscore:.0%}) — {pattern.description[:80]}")

    if not factors:
        factors.append("No crisis patterns detected — normal market conditions")

    return MarketRegime(
        regime_name=regime_name,
        regime_score=regime_score,
        vix_level=vix_level,
        cross_sector_correlation=cross_corr,
        vol_regime_z=vol_z,
        active_patterns=active_patterns,
        sector_adjustments=sector_adjustments,
        contagion_risk=contagion_risk,
        factors=factors,
    )


# ─────────────────────────────────────────────────────────────
# MODEL 5: CRISIS PATTERN & CROSS-SECTOR MODEL
# ─────────────────────────────────────────────────────────────

def model_crisis_regime(
    ticker: str,
    sector: str,
    current_price: float,
    horizon_days: int,
    regime: MarketRegime,
) -> Optional[dict]:
    """
    Model 5: Adjusts predictions based on detected crisis patterns
    and cross-sector regime state.

    Key insight from Reinhart & Rogoff (2009):
    "The aftermath of severe financial crises share three characteristics:
     1. Asset market collapses are deep and prolonged
     2. The aftermath is associated with profound declines in output
     3. The real value of government debt tends to explode"

    And from Kindleberger: crises propagate across sectors with predictable
    lag structures — energy shocks hit industrials within 2-6 months,
    credit crises hit everything within weeks.
    """
    if current_price <= 0:
        return None

    # Base adjustment from pattern matching
    sector_adj_weekly = regime.sector_adjustments.get(sector, 0.0)
    # Scale to horizon
    predicted_pct = sector_adj_weekly * (horizon_days / 5.0)

    # Contagion adjustment: in crisis, cross-sector effects dominate
    # individual sector fundamentals (Brunnermeier 2009)
    if regime.contagion_risk > 0.5:
        # In high-contagion environments, all risk assets move together
        avg_adj = sum(regime.sector_adjustments.values()) / len(regime.sector_adjustments)
        blend = regime.contagion_risk  # higher contagion = more herding
        predicted_pct = predicted_pct * (1 - blend * 0.5) + avg_adj * (horizon_days / 5.0) * blend * 0.5

    # Volatility scaling: widen predictions in high-vol regimes
    vol_multiplier = max(1.0, 1.0 + regime.vol_regime_z * 0.2)

    # CI from regime volatility
    base_vol = 0.015 * math.sqrt(horizon_days) * vol_multiplier
    # Crisis regimes have fatter tails (Mandelbrot, 1963)
    if regime.regime_name in ("CRISIS", "STRESSED"):
        base_vol *= 1.5  # fat-tail adjustment

    predicted_price = current_price * (1 + predicted_pct / 100)

    ci_80 = (current_price * (1 + predicted_pct / 100 - 1.28 * base_vol),
             current_price * (1 + predicted_pct / 100 + 1.28 * base_vol))
    ci_95 = (current_price * (1 + predicted_pct / 100 - 1.96 * base_vol),
             current_price * (1 + predicted_pct / 100 + 1.96 * base_vol))

    # Build factor description
    if regime.active_patterns:
        top_pattern = regime.active_patterns[0]
        factor_desc = (
            f"Regime: {regime.regime_name} (score {regime.regime_score:.0f}/100). "
            f"Pattern: {top_pattern[0]} ({top_pattern[1]:.0%} match). "
            f"Contagion risk: {regime.contagion_risk:.0%}"
        )
    else:
        factor_desc = (
            f"Regime: {regime.regime_name} (score {regime.regime_score:.0f}/100). "
            f"Cross-corr: {regime.cross_sector_correlation:.2f}. "
            f"No crisis patterns active"
        )

    return {
        "predicted_pct": predicted_pct,
        "predicted_price": predicted_price,
        "ci_80": ci_80,
        "ci_95": ci_95,
        "factor": factor_desc,
    }
