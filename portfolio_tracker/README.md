# Geopolitical Investment Tracker

A real-time geopolitical investment tracking and advisory system for a €100,000 portfolio
with a HIGH risk profile and 5-year horizon, managed from Luxembourg.

---

## Setup

```bash
cd portfolio_tracker
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

## Running the Dashboard

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Running the Terminal Dashboard

```bash
python run_terminal.py
```

With backtest mode (replays last 30 days of stored price history):

```bash
python run_terminal.py --backtest
```

Custom refresh interval (seconds):

```bash
python run_terminal.py --interval 30
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

All portfolio positions, sector weights, signal thresholds, and investor profile
are defined in **`config.py`** — the single source of truth.

### Customising the Portfolio

Edit the `PORTFOLIO` list in `config.py`:

```python
PORTFOLIO = [
    {
        "ticker":         "XOM",
        "name":           "ExxonMobil Corp",
        "exchange":       "NYSE",
        "type":           "STOCK",
        "sector":         "ENERGY",
        "isin":           "US30231G1022",
        "allocation_eur": 6000,
        "target_pct":     6.0,
        "currency":       "USD",
        "fx_strategy":    "ACCEPT",
        "tranche":        1,
        "ucits_compliant": False,
    },
    # Add more positions here...
]
```

### Customising Sector Scoring

Edit `SECTOR_CONFIG` in `config.py` to adjust:
- `base_score` (0-10) — starting geopolitical conviction score
- `primary_triggers` — keywords that increase the score
- `negative_triggers` — keywords that decrease the score
- `geopolitical_sensitivity` — informational only

### Signal Thresholds

Edit `SIGNAL_THRESHOLDS` in `config.py`:

```python
SIGNAL_THRESHOLDS = {
    "BUY":    75,   # composite >= 75
    "ADD":    60,   # composite >= 60
    "HOLD":   45,   # composite >= 45
    "REDUCE": 30,   # composite >= 30
    # Below 30 → SELL
}
```

### Alert Thresholds

```python
ALERT_DAY_MOVE_MEDIUM   = 5.0    # % single-day move → MEDIUM
ALERT_DAY_MOVE_HIGH     = 10.0   # % single-day move → HIGH
ALERT_DRAWDOWN_CRITICAL = -40.0  # % drawdown → CRITICAL
```

---

## Adding a NewsAPI Key (Optional)

1. Copy `.env.example` to `.env`
2. Set your key: `NEWSAPI_KEY=your_key`
3. The system will automatically use it on next restart

Without a key, the system works entirely on free RSS feeds from Reuters, BBC,
Financial Times, Al Jazeera, Oilprice.com, and Defense News.

---

## Architecture

### `config.py`
Single source of truth. All portfolio positions, sector configurations, signal
weights, thresholds, and investor profile are defined here. Nothing is hardcoded
elsewhere.

### `src/fx_engine.py`
Fetches live EUR FX rates from the ECB XML daily feed every 4 hours. Provides
EUR conversion for USD, GBP, NOK, and CHF positions. Falls back to hardcoded
rates if the ECB feed is unavailable.

### `src/price_engine.py`
Uses yfinance to fetch prices for all 14 portfolio positions plus VIX and Brent
crude. Calculates P&L, momentum (1W/1M), drift from target, and current weight.
Stores all price snapshots in SQLite. Refreshes every 5 minutes during market
hours (08:00-22:00 CET, Mon-Fri). Records entry prices on first run and never
overwrites them.

### `src/geo_scorer.py`
Fetches 7 RSS feeds every 15 minutes (Reuters, BBC, FT, Al Jazeera, Oilprice,
DefenseNews). Scores each article via keyword matching against sector trigger
lists and negative trigger lists. Maintains 5 geopolitical state variables
(Hormuz, Iran, Ukraine, US-China, NATO). Computes sector geo scores (0-10)
as base_score ± news sentiment drift (max ±2.5). No paid APIs required.

### `src/signal_engine.py`
Generates BUY/ADD/HOLD/REDUCE/SELL signals every 15 minutes using a 5-factor
weighted composite score: Sector Geo Score (35%), Price Momentum (20%),
Position Drift (15%), Drawdown Risk (15%), Volatility Regime (15%). Applies
hard override rules for geopolitical emergencies and broken investment theses.
Calibrated for HIGH risk profile with 5-year horizon.

### `src/alert_manager.py`
Monitors all positions and geo variables for alert conditions: price moves >5%
or >10% in a day, signal changes, geo variable state changes (CRITICAL for
Hormuz), drawdown breaches at -25% and -40%, and rebalancing drift >5%.
Sends desktop notifications via `plyer` for HIGH/CRITICAL alerts. Writes to a
rotating `logs/alerts.log` file.

### `src/database.py`
SQLAlchemy ORM layer over SQLite at `data/portfolio.db`. Manages price history
(2-year retention), entry prices, alerts, geo state, signal history, sector
score history, and news cache. Uses WAL mode for safe concurrent access.

### `app.py`
Streamlit web dashboard with 7 sections: header bar, geopolitical situation board,
sector overview with sparklines, position table with signal colouring, signal
rationale panel (expandable per position), alerts feed, portfolio charts (5 tabs),
and tranche deployment tracker. Auto-refreshes every 60 seconds.

### `run_terminal.py`
Rich terminal dashboard using `Live()` context manager. Shows portfolio summary,
geo status, full position table with signals, and last 5 alerts. Supports
`--backtest` mode to replay the last 30 days of stored price history.

---

## Data Storage

- `data/portfolio.db` — SQLite database (auto-created on first run)
- `logs/app.log` — Application log (daily rotation, 30 days retention)
- `logs/alerts.log` — Alert log (daily rotation, 30 days retention)

---

## First Run Behaviour

On first run, the system:
1. Initialises the SQLite database
2. Fetches current prices from yfinance
3. Records entry prices for all positions at current market prices
4. Entry prices are **never overwritten** on subsequent runs
5. Stores the system start date for tranche deployment countdown

---

## Tranche Deployment

The portfolio is deployed in 3 tranches:
- **Tranche 1** (immediate): SPOG.L, EQNR, XOM, DFNS.L, NATO.L, SETM.L, VVMX.DE, IGLN.L, GDX
- **Tranche 2** (day 90): RHM.DE, BA.L, FCX
- **Tranche 3** (day 180): BTEC.AS, VRTX

The dashboard shows countdown timers and generates deployment reminder alerts.
