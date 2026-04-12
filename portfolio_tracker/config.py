"""
config.py — Single source of truth for all constants, portfolio, and sector configuration.
Edit this file to customize positions, thresholds, and investor profile.
"""

from datetime import date

# ─────────────────────────────────────────────────────────────
# SYSTEM START DATE — used for tranche deployment countdown
# ─────────────────────────────────────────────────────────────
SYSTEM_START_DATE = date.today()  # Will be overridden by DB on subsequent runs

# ─────────────────────────────────────────────────────────────
# INVESTOR PROFILE
# ─────────────────────────────────────────────────────────────
INVESTOR_PROFILE = {
    "base_currency": "EUR",
    "total_capital": 100_000,
    "risk_profile": "HIGH",
    "horizon_years": 5,
    "domicile": "Luxembourg",
    "tax_regime": "No capital gains tax on securities held > 6 months",
    "rebalance_threshold_pct": 5.0,
    "stop_loss_pct": -40.0,
    "max_single_position_pct": 20.0,
}

# ─────────────────────────────────────────────────────────────
# PORTFOLIO POSITIONS
# ─────────────────────────────────────────────────────────────
PORTFOLIO = [
    # ── ENERGY / OIL & GAS (28%) ─────────────────────────────
    {
        "ticker": "SPOG.L",
        "name": "iShares Oil & Gas E&P UCITS ETF",
        "exchange": "LSE",
        "type": "UCITS_ETF",
        "sector": "ENERGY",
        "isin": "IE00B6R51Z18",
        "allocation_eur": 14_000,
        "target_pct": 14.0,
        "currency": "USD",
        "fx_strategy": "ACCEPT",
        "tranche": 1,
        "ucits_compliant": True,
    },
    {
        "ticker": "EQNR",
        "name": "Equinor ASA",
        "exchange": "NYSE",
        "type": "STOCK",
        "sector": "ENERGY",
        "isin": "NO0010096985",
        "allocation_eur": 8_000,
        "target_pct": 8.0,
        "currency": "NOK",
        "fx_strategy": "NATURAL",
        "tranche": 1,
        "ucits_compliant": False,
    },
    {
        "ticker": "XOM",
        "name": "ExxonMobil Corp",
        "exchange": "NYSE",
        "type": "STOCK",
        "sector": "ENERGY",
        "isin": "US30231G1022",
        "allocation_eur": 6_000,
        "target_pct": 6.0,
        "currency": "USD",
        "fx_strategy": "ACCEPT",
        "tranche": 1,
        "ucits_compliant": False,
    },
    # ── EUROPEAN DEFENSE (27%) ────────────────────────────────
    {
        "ticker": "DFNS.L",
        "name": "VanEck Defense UCITS ETF",
        "exchange": "LSE",
        "type": "UCITS_ETF",
        "sector": "DEFENSE",
        "isin": "IE00BF0ZS284",
        "allocation_eur": 9_000,
        "target_pct": 9.0,
        "currency": "USD",
        "fx_strategy": "EUR_LISTED",
        "tranche": 1,
        "ucits_compliant": True,
    },
    {
        "ticker": "NATO.L",
        "name": "HANetf Future of Defence UCITS ETF",
        "exchange": "LSE",
        "type": "UCITS_ETF",
        "sector": "DEFENSE",
        "isin": "IE0002FJ2JP4",
        "allocation_eur": 8_000,
        "target_pct": 8.0,
        "currency": "EUR",
        "fx_strategy": "EUR_LISTED",
        "tranche": 1,
        "ucits_compliant": True,
    },
    {
        "ticker": "RHM.DE",
        "name": "Rheinmetall AG",
        "exchange": "XETRA",
        "type": "STOCK",
        "sector": "DEFENSE",
        "isin": "DE0007030009",
        "allocation_eur": 7_000,
        "target_pct": 7.0,
        "currency": "EUR",
        "fx_strategy": "EUR_NATIVE",
        "tranche": 2,
        "ucits_compliant": False,
    },
    {
        "ticker": "BA.L",
        "name": "BAE Systems plc",
        "exchange": "LSE",
        "type": "STOCK",
        "sector": "DEFENSE",
        "isin": "GB0002634946",
        "allocation_eur": 3_000,
        "target_pct": 3.0,
        "currency": "GBP",
        "fx_strategy": "MINOR_FX",
        "tranche": 2,
        "ucits_compliant": False,
    },
    # ── METALS & MINERALS (22%) ───────────────────────────────
    {
        "ticker": "SETM.L",
        "name": "Sprott Energy Transition Materials UCITS ETF",
        "exchange": "LSE",
        "type": "UCITS_ETF",
        "sector": "METALS",
        "isin": "IE000H9FHVZ8",
        "allocation_eur": 10_000,
        "target_pct": 10.0,
        "currency": "USD",
        "fx_strategy": "DO_NOT_HEDGE",
        "tranche": 1,
        "ucits_compliant": True,
    },
    {
        "ticker": "VVMX.DE",
        "name": "VanEck Rare Earth Strategic Metals UCITS ETF",
        "exchange": "XETRA",
        "type": "UCITS_ETF",
        "sector": "METALS",
        "isin": "IE0002PG6CA6",
        "allocation_eur": 7_000,
        "target_pct": 7.0,
        "currency": "USD",
        "fx_strategy": "DO_NOT_HEDGE",
        "tranche": 1,
        "ucits_compliant": True,
    },
    {
        "ticker": "FCX",
        "name": "Freeport-McMoRan Inc",
        "exchange": "NYSE",
        "type": "STOCK",
        "sector": "METALS",
        "isin": "US35671D8570",
        "allocation_eur": 5_000,
        "target_pct": 5.0,
        "currency": "USD",
        "fx_strategy": "ACCEPT",
        "tranche": 2,
        "ucits_compliant": False,
    },
    # ── GOLD / PRECIOUS METALS (15%) ──────────────────────────
    {
        "ticker": "IGLN.L",
        "name": "iShares Physical Gold ETC",
        "exchange": "LSE",
        "type": "ETC",
        "sector": "GOLD",
        "isin": "IE00B4ND3602",
        "allocation_eur": 10_000,
        "target_pct": 10.0,
        "currency": "USD",
        "fx_strategy": "ACCEPT",
        "tranche": 1,
        "ucits_compliant": True,
    },
    {
        "ticker": "GDX",
        "name": "VanEck Gold Miners UCITS ETF",
        "exchange": "NYSE",
        "type": "UCITS_ETF",
        "sector": "GOLD",
        "isin": "IE00BQQP9F84",
        "allocation_eur": 5_000,
        "target_pct": 5.0,
        "currency": "USD",
        "fx_strategy": "ACCEPT",
        "tranche": 1,
        "ucits_compliant": True,
    },
    # ── BIOTECH (8%) ──────────────────────────────────────────
    {
        "ticker": "BTEC.AS",
        "name": "iShares Nasdaq US Biotech UCITS ETF",
        "exchange": "EURONEXT_AMSTERDAM",
        "type": "UCITS_ETF",
        "sector": "BIOTECH",
        "isin": "IE00BD3VGB10",
        "allocation_eur": 5_000,
        "target_pct": 5.0,
        "currency": "EUR",
        "fx_strategy": "EUR_CLASS",
        "tranche": 3,
        "ucits_compliant": True,
    },
    {
        "ticker": "VRTX",
        "name": "Vertex Pharmaceuticals Inc",
        "exchange": "NASDAQ",
        "type": "STOCK",
        "sector": "BIOTECH",
        "isin": "US92532F1003",
        "allocation_eur": 3_000,
        "target_pct": 3.0,
        "currency": "USD",
        "fx_strategy": "ACCEPT",
        "tranche": 3,
        "ucits_compliant": False,
    },
]

# ─────────────────────────────────────────────────────────────
# YAHOO FINANCE TICKER OVERRIDES
# Some portfolio tickers don't exist on Yahoo Finance under their
# primary exchange symbol. Map them to the correct YF symbol here.
# The portfolio ISIN and original ticker are preserved for display;
# only price fetching uses the override.
#
# Verified 2026-04: checked with yfinance fast_info.currency
# ─────────────────────────────────────────────────────────────
YF_TICKER_MAP = {
    "SETM.L":  "SETM",    # Not available on LSE via YF — use NYSE/ARCA listing (USD)
    "BTEC.AS": "BTEC.L",  # Not available on AEX via YF — use LSE listing (USD)
}

# Effective currency to use for EUR conversion, keyed by PORTFOLIO ticker.
# Overrides pos.currency when Yahoo Finance prices in a different currency
# than the portfolio config states.
#   SPOG.L  — YF returns GBp (pence); after ÷100 → GBP
#   BA.L    — YF returns GBp (pence); after ÷100 → GBP
#   NATO.L  — YF quotes in USD (fund NAV is USD-based on LSE)
#   SETM.L  — remapped to SETM (NYSE) which quotes USD
#   BTEC.AS — remapped to BTEC.L (LSE) which quotes USD
YF_CURRENCY_OVERRIDE = {
    # LSE tickers priced in pence → after ÷100 use GBP
    "SPOG.L":  "GBP",   # GBp ÷100 → GBP → EUR
    "BA.L":    "GBP",   # GBp ÷100 → GBP → EUR
    # LSE ETFs whose Yahoo quote currency differs from portfolio config
    "NATO.L":  "USD",   # LSE listing quotes USD (not EUR)
    # Remapped tickers
    "SETM.L":  "USD",   # NYSE SETM quotes USD
    "BTEC.AS": "USD",   # LSE BTEC.L quotes USD (not EUR)
    # Portfolio states NOK (home market) but Yahoo fetches NYSE listing in USD
    "EQNR":    "USD",
    # Portfolio states USD (underlying) but XETRA listing quotes EUR
    "VVMX.DE": "EUR",
}

# Portfolio tickers whose Yahoo Finance price arrives in pence (GBp) → divide by 100
# Only tickers where fast_info.currency == "GBp"
YF_PENCE_TICKERS = {"SPOG.L", "BA.L"}

# ─────────────────────────────────────────────────────────────
# SECTOR CONFIGURATION
# ─────────────────────────────────────────────────────────────
SECTOR_CONFIG = {
    "ENERGY": {
        "base_score": 9.1,
        "weight_in_portfolio": 0.28,
        "geopolitical_sensitivity": "EXTREME",
        "primary_triggers": [
            "hormuz", "iran", "oil", "brent", "crude", "opec",
            "sanctions", "strait", "gulf", "tanker", "lng",
        ],
        "negative_triggers": [
            "ceasefire", "hormuz reopened", "iran deal",
            "global recession", "demand destruction",
        ],
        "bull_scenario": "Hormuz closure extends, Brent $150+",
        "bear_scenario": "Ceasefire + full strait reopening → oil normalises $70",
    },
    "DEFENSE": {
        "base_score": 9.0,
        "weight_in_portfolio": 0.27,
        "geopolitical_sensitivity": "VERY_HIGH",
        "primary_triggers": [
            "nato", "defense spending", "rearmament", "ukraine",
            "rheinmetall", "bae", "military", "war", "conflict",
            "eu defense", "european army",
        ],
        "negative_triggers": [
            "peace deal", "ukraine ceasefire", "nato spending cut",
            "trump nato withdrawal", "disarmament",
        ],
        "bull_scenario": "EU €800B plan accelerates + NATO 5% GDP",
        "bear_scenario": "Ukraine peace + US re-engages NATO → European defense cut",
    },
    "METALS": {
        "base_score": 6.8,
        "weight_in_portfolio": 0.22,
        "geopolitical_sensitivity": "HIGH",
        "primary_triggers": [
            "copper", "lithium", "rare earth", "critical minerals",
            "china export", "ev demand", "battery", "setm", "mining",
            "supply deficit",
        ],
        "negative_triggers": [
            "china rare earth deal", "recession", "ev slowdown",
            "oversupply", "demand crash", "commodity bust",
        ],
        "bull_scenario": "China restricts again + EV demand surges + AI data centers",
        "bear_scenario": "China-US détente removes rare earth premium + recession",
    },
    "GOLD": {
        "base_score": 8.5,
        "weight_in_portfolio": 0.15,
        "geopolitical_sensitivity": "HIGH",
        "primary_triggers": [
            "inflation", "safe haven", "gold", "central bank buying",
            "dollar weakness", "geopolitical risk", "war escalation",
        ],
        "negative_triggers": [
            "peace", "inflation falls", "rate cuts confirmed",
            "dollar strengthens", "risk on",
        ],
        "bull_scenario": "Inflation stays high + central bank buying + conflict escalates",
        "bear_scenario": "Peace + deflation + strong dollar",
    },
    "BIOTECH": {
        "base_score": 5.0,
        "weight_in_portfolio": 0.08,
        "geopolitical_sensitivity": "LOW",
        "primary_triggers": [
            "fda approval", "m&a biotech", "acquisition", "drug approval",
            "clinical trial", "rate cut", "pharma buyout",
        ],
        "negative_triggers": [
            "rate hike", "fda rejection", "clinical failure",
            "drug pricing", "stagflation", "capital rotation",
        ],
        "bull_scenario": "M&A wave + FDA approvals + rate cuts resume",
        "bear_scenario": "Stagflation + hikes + capital rotation to energy",
    },
}

# ─────────────────────────────────────────────────────────────
# SIGNAL ENGINE WEIGHTS
# ─────────────────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "sector_geo_score": 0.35,
    "price_momentum": 0.20,
    "position_drift": 0.15,
    "drawdown_risk": 0.15,
    "volatility_regime": 0.15,
}

SIGNAL_THRESHOLDS = {
    "BUY":    75,
    "ADD":    60,
    "HOLD":   45,
    "REDUCE": 30,
    # Below 30 → SELL
}

# ─────────────────────────────────────────────────────────────
# RSS FEEDS
# ─────────────────────────────────────────────────────────────
RSS_FEEDS = [
    {"name": "Reuters World",    "url": "https://feeds.reuters.com/reuters/worldNews"},
    {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "BBC World",        "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "Financial Times",  "url": "https://www.ft.com/rss/home"},
    {"name": "Al Jazeera",       "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"name": "Oilprice.com",     "url": "https://oilprice.com/rss/main"},
    {"name": "Defense News",     "url": "https://www.defensenews.com/arc/outboundfeeds/rss/"},
]

# ─────────────────────────────────────────────────────────────
# GEOPOLITICAL VARIABLES — default states
# ─────────────────────────────────────────────────────────────
GEO_VARIABLES_DEFAULT = {
    "HORMUZ_STATUS":       "OPEN",          # OPEN / PARTIAL / CLOSED
    "IRAN_CONFLICT":       "ACTIVE",        # ACTIVE / CEASEFIRE / RESOLVED
    "UKRAINE_WAR":         "STALEMATE",     # ESCALATING / STALEMATE / DE-ESCALATING / RESOLVED
    "US_CHINA_RELATIONS":  "TENSE",         # HOSTILE / TENSE / NEUTRAL / COOPERATIVE
    "NATO_SPENDING":       "INCREASING",    # DECLINING / STABLE / INCREASING / ACCELERATING
}

# ─────────────────────────────────────────────────────────────
# ECB FX FEED
# ─────────────────────────────────────────────────────────────
ECB_FX_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"

# ─────────────────────────────────────────────────────────────
# MARKET HOURS (CET) for price refresh
# ─────────────────────────────────────────────────────────────
MARKET_HOURS_START = 8    # 08:00 CET
MARKET_HOURS_END   = 22   # 22:00 CET

# ─────────────────────────────────────────────────────────────
# REFRESH INTERVALS (seconds)
# ─────────────────────────────────────────────────────────────
PRICE_REFRESH_INTERVAL  = 300    # 5 minutes
GEO_REFRESH_INTERVAL    = 900    # 15 minutes
SIGNAL_REFRESH_INTERVAL = 900    # 15 minutes

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
DB_PATH = "data/portfolio.db"

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
LOG_DIR  = "logs"
APP_LOG  = "logs/app.log"
ALERT_LOG = "logs/alerts.log"

# ─────────────────────────────────────────────────────────────
# ALERT THRESHOLDS
# ─────────────────────────────────────────────────────────────
ALERT_DAY_MOVE_MEDIUM   = 5.0    # % single-day move → MEDIUM alert
ALERT_DAY_MOVE_HIGH     = 10.0   # % single-day move → HIGH alert
ALERT_PORTFOLIO_DAY_PCT = 3.0    # % portfolio day change → HIGH alert
ALERT_DRIFT_THRESHOLD   = 5.0    # % drift from target → MEDIUM alert
ALERT_DRAWDOWN_MEDIUM   = -25.0  # % drawdown → MEDIUM alert
ALERT_DRAWDOWN_CRITICAL = -40.0  # % drawdown → CRITICAL alert

# ─────────────────────────────────────────────────────────────
# TRANCHE DEPLOYMENT DAYS
# ─────────────────────────────────────────────────────────────
TRANCHE_2_DAYS = 90
TRANCHE_3_DAYS = 180

# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
PREDICTION_REFRESH_INTERVAL = 1800   # 30 minutes
PREDICTION_MIN_HISTORY_DAYS = 30     # minimum for technical models (1-3)
PREDICTION_ENSEMBLE_WEIGHTS = {
    "ema_momentum":      0.25,
    "linear_regression": 0.25,
    "mean_reversion":    0.20,
    "geopolitical":      0.30,
}
PREDICTION_HORIZONS = {
    "24h": 1,    # trading days
    "1w":  5,
    "1m":  21,
}
