"""
database.py — SQLite ORM helpers using SQLAlchemy.
Handles price history, alerts, entry prices, geo state, signals, and system config.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    create_engine, event, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    from config import DB_PATH
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return DB_PATH


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────
# ORM MODELS
# ─────────────────────────────────────────────────────────────

class PriceHistory(Base):
    __tablename__ = "price_history"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    ticker      = Column(String(20), nullable=False, index=True)
    timestamp   = Column(DateTime, nullable=False, index=True)
    price_local = Column(Float, nullable=True)
    price_eur   = Column(Float, nullable=True)
    volume      = Column(Float, nullable=True)


class EntryPrice(Base):
    __tablename__ = "entry_prices"
    ticker          = Column(String(20), primary_key=True)
    entry_price_local = Column(Float, nullable=False)
    entry_price_eur   = Column(Float, nullable=False)
    entry_timestamp   = Column(DateTime, nullable=False)
    shares_units      = Column(Float, nullable=True)   # derived from allocation / entry_price_eur


class Alert(Base):
    __tablename__ = "alerts"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(DateTime, nullable=False, index=True)
    severity     = Column(String(20), nullable=False)   # CRITICAL / HIGH / MEDIUM / REMINDER
    ticker       = Column(String(20), nullable=True)
    sector       = Column(String(20), nullable=True)
    alert_type   = Column(String(50), nullable=False)
    message      = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False, nullable=False)


class GeoState(Base):
    __tablename__ = "geo_state"
    variable      = Column(String(50), primary_key=True)
    current_value = Column(String(50), nullable=False)
    previous_value = Column(String(50), nullable=True)
    last_changed  = Column(DateTime, nullable=True)
    triggering_headline = Column(Text, nullable=True)
    impact_summary = Column(Text, nullable=True)


class SignalHistory(Base):
    __tablename__ = "signal_history"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    ticker          = Column(String(20), nullable=False, index=True)
    timestamp       = Column(DateTime, nullable=False)
    signal          = Column(String(10), nullable=False)
    composite_score = Column(Float, nullable=True)
    geo_score       = Column(Float, nullable=True)
    momentum_score  = Column(Float, nullable=True)
    drift_score     = Column(Float, nullable=True)
    drawdown_score  = Column(Float, nullable=True)
    volatility_score = Column(Float, nullable=True)
    flags           = Column(Text, nullable=True)   # JSON list of override flags


class SectorScoreHistory(Base):
    __tablename__ = "sector_score_history"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    sector    = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    geo_score = Column(Float, nullable=True)


class SystemConfig(Base):
    __tablename__ = "system_config"
    key   = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)


class NewsCache(Base):
    __tablename__ = "news_cache"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    url_hash   = Column(String(64), nullable=False, unique=True, index=True)
    title      = Column(Text, nullable=False)
    summary    = Column(Text, nullable=True)
    source     = Column(String(100), nullable=True)
    published  = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, nullable=False)
    sectors_matched = Column(Text, nullable=True)   # JSON list
    sentiment_delta = Column(Float, nullable=True)


# ─────────────────────────────────────────────────────────────
# ENGINE / SESSION FACTORY
# ─────────────────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        db_path = _get_db_path()
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        # Enable WAL mode for concurrent readers
        @event.listens_for(_engine, "connect")
        def set_wal(dbapi_conn, connection_record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        Base.metadata.create_all(_engine)
        logger.info("Database initialised at %s", db_path)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    with get_session() as s:
        row = s.get(SystemConfig, key)
        return row.value if row else default


def set_config_value(key: str, value: str) -> None:
    with get_session() as s:
        row = s.get(SystemConfig, key)
        if row:
            row.value = value
        else:
            s.add(SystemConfig(key=key, value=value))
        s.commit()


def save_entry_price(ticker: str, price_local: float, price_eur: float,
                     shares: float, timestamp: Optional[datetime] = None) -> None:
    """Store entry price — only on first run (never overwrite)."""
    with get_session() as s:
        existing = s.get(EntryPrice, ticker)
        if existing:
            return  # Never overwrite entry prices
        s.add(EntryPrice(
            ticker=ticker,
            entry_price_local=price_local,
            entry_price_eur=price_eur,
            entry_timestamp=timestamp or datetime.now(timezone.utc),
            shares_units=shares,
        ))
        s.commit()
        logger.info("Entry price stored for %s: %.4f EUR", ticker, price_eur)


def get_entry_price(ticker: str) -> Optional[EntryPrice]:
    with get_session() as s:
        return s.get(EntryPrice, ticker)


def save_price_snapshot(ticker: str, price_local: float, price_eur: float,
                        volume: Optional[float] = None) -> None:
    with get_session() as s:
        s.add(PriceHistory(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            price_local=price_local,
            price_eur=price_eur,
            volume=volume,
        ))
        s.commit()


def get_price_history(ticker: str, days: int = 30):
    """Return price_eur history for the last N days."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with get_session() as s:
        rows = (
            s.query(PriceHistory)
            .filter(PriceHistory.ticker == ticker, PriceHistory.timestamp >= cutoff)
            .order_by(PriceHistory.timestamp.asc())
            .all()
        )
        return rows


def purge_old_prices(keep_days: int = 730) -> None:
    """Keep only 2 years of price history."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    with get_session() as s:
        deleted = s.query(PriceHistory).filter(PriceHistory.timestamp < cutoff).delete()
        s.commit()
        if deleted:
            logger.info("Purged %d old price records", deleted)


def save_alert(severity: str, alert_type: str, message: str,
               ticker: Optional[str] = None, sector: Optional[str] = None) -> int:
    with get_session() as s:
        a = Alert(
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            ticker=ticker,
            sector=sector,
            alert_type=alert_type,
            message=message,
            acknowledged=False,
        )
        s.add(a)
        s.commit()
        return a.id


def get_unacknowledged_alerts(limit: int = 200):
    with get_session() as s:
        return (
            s.query(Alert)
            .filter(Alert.acknowledged == False)
            .order_by(Alert.timestamp.desc())
            .limit(limit)
            .all()
        )


def acknowledge_alert(alert_id: int) -> None:
    with get_session() as s:
        a = s.get(Alert, alert_id)
        if a:
            a.acknowledged = True
            s.commit()


def save_geo_state(variable: str, value: str, headline: Optional[str] = None,
                   impact: Optional[str] = None) -> bool:
    """Returns True if the value changed."""
    with get_session() as s:
        row = s.get(GeoState, variable)
        changed = False
        if row:
            if row.current_value != value:
                row.previous_value = row.current_value
                row.current_value  = value
                row.last_changed   = datetime.now(timezone.utc)
                row.triggering_headline = headline
                row.impact_summary = impact
                changed = True
        else:
            s.add(GeoState(
                variable=variable,
                current_value=value,
                previous_value=None,
                last_changed=datetime.now(timezone.utc),
                triggering_headline=headline,
                impact_summary=impact,
            ))
            changed = True
        s.commit()
        return changed


def get_geo_states() -> dict:
    with get_session() as s:
        rows = s.query(GeoState).all()
        return {r.variable: r for r in rows}


def save_signal(ticker: str, signal: str, composite: float, factor_scores: dict,
                flags: Optional[str] = None) -> None:
    with get_session() as s:
        s.add(SignalHistory(
            ticker=ticker,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            composite_score=composite,
            geo_score=factor_scores.get("geo"),
            momentum_score=factor_scores.get("momentum"),
            drift_score=factor_scores.get("drift"),
            drawdown_score=factor_scores.get("drawdown"),
            volatility_score=factor_scores.get("volatility"),
            flags=flags,
        ))
        s.commit()


def get_signal_history(ticker: str, limit: int = 50):
    with get_session() as s:
        return (
            s.query(SignalHistory)
            .filter(SignalHistory.ticker == ticker)
            .order_by(SignalHistory.timestamp.desc())
            .limit(limit)
            .all()
        )


def save_sector_score(sector: str, score: float) -> None:
    with get_session() as s:
        s.add(SectorScoreHistory(
            sector=sector,
            timestamp=datetime.now(timezone.utc),
            geo_score=score,
        ))
        s.commit()


def get_sector_score_history(sector: str, days: int = 30):
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with get_session() as s:
        return (
            s.query(SectorScoreHistory)
            .filter(
                SectorScoreHistory.sector == sector,
                SectorScoreHistory.timestamp >= cutoff,
            )
            .order_by(SectorScoreHistory.timestamp.asc())
            .all()
        )


def news_already_cached(url_hash: str) -> bool:
    with get_session() as s:
        return s.query(NewsCache).filter(NewsCache.url_hash == url_hash).count() > 0


def save_news_cache(url_hash: str, title: str, summary: str, source: str,
                    published: Optional[datetime], sectors_matched: str,
                    sentiment_delta: float) -> None:
    with get_session() as s:
        if not news_already_cached(url_hash):
            s.add(NewsCache(
                url_hash=url_hash,
                title=title,
                summary=summary or "",
                source=source or "",
                published=published,
                fetched_at=datetime.now(timezone.utc),
                sectors_matched=sectors_matched,
                sentiment_delta=sentiment_delta,
            ))
            s.commit()


def get_recent_news(hours: int = 48, limit: int = 100):
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with get_session() as s:
        return (
            s.query(NewsCache)
            .filter(NewsCache.fetched_at >= cutoff)
            .order_by(NewsCache.fetched_at.desc())
            .limit(limit)
            .all()
        )


def init_db() -> None:
    """Ensure DB is initialised and tables exist."""
    get_engine()
    logger.info("Database ready")
