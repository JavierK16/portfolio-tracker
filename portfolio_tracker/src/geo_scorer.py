"""
geo_scorer.py — Geopolitical news scoring engine.
Fetches RSS feeds, matches keywords, updates geo variables, and computes sector scores.
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import feedparser
import requests

from src.database import (
    get_geo_states, save_geo_state, news_already_cached, save_news_cache,
    get_recent_news, save_sector_score,
)

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    published: Optional[datetime]
    url_hash: str
    sectors_matched: List[str]
    sentiment_delta: float  # -2 to +2


@dataclass
class GeoContext:
    variables: Dict[str, dict]            # variable -> {value, prev, changed_hours_ago, headline}
    sector_scores: Dict[str, float]        # sector -> current geo score
    sector_score_changes: Dict[str, float] # sector -> delta vs yesterday
    recent_news: List[NewsItem]
    last_updated: Optional[datetime]


# Keyword → geopolitical variable state inference
_GEO_VARIABLE_SIGNALS = {
    "HORMUZ_STATUS": {
        "CLOSED":  ["hormuz closed", "strait closed", "hormuz blockade", "hormuz blocked",
                    "iran closes", "mining strait"],
        "PARTIAL": ["hormuz tension", "hormuz threat", "hormuz warning", "tanker attacked",
                    "strait threatened", "iran threatens strait"],
        "OPEN":    ["hormuz reopened", "strait clear", "iran deal signed",
                    "hormuz open", "ceasefire hormuz"],
    },
    "IRAN_CONFLICT": {
        "ACTIVE":    ["iran attack", "iran missile", "iran strike", "iran military",
                      "iran nuclear", "iran sanctions", "iran conflict"],
        "CEASEFIRE": ["iran ceasefire", "iran talks", "iran deal", "iran agreement",
                      "iran negotiations"],
        "RESOLVED":  ["iran deal signed", "iran sanctions lifted", "iran nuclear deal",
                      "iran agreement signed"],
    },
    "UKRAINE_WAR": {
        "ESCALATING":    ["ukraine escalat", "russia expand", "nuclear threat russia",
                          "ukraine offensive", "russia advance", "nato conflict"],
        "STALEMATE":     ["ukraine front stalemate", "ukraine deadlock", "frozen conflict"],
        "DE-ESCALATING": ["ukraine ceasefire", "peace talks ukraine", "ukraine negotiation",
                          "russia withdraw"],
        "RESOLVED":      ["ukraine peace deal", "ukraine war ends", "russia ukraine agreement"],
    },
    "US_CHINA_RELATIONS": {
        "HOSTILE":     ["china us war", "taiwan invasion", "china blockade taiwan",
                        "us china sanctions", "china chip ban"],
        "TENSE":       ["china us tariff", "china us trade war", "us china tension",
                        "rare earth ban china", "china export restriction"],
        "NEUTRAL":     ["china us talks", "china us meeting", "china us dialogue"],
        "COOPERATIVE": ["china us deal", "china us agreement", "china us trade deal"],
    },
    "NATO_SPENDING": {
        "DECLINING":     ["nato budget cut", "nato spending cut", "nato funding reduced",
                          "trump nato withdrawal"],
        "STABLE":        ["nato spending stable", "nato budget unchanged"],
        "INCREASING":    ["nato spending increase", "nato rearmament", "eu defense spending",
                          "nato gdp target", "defense budget increase"],
        "ACCELERATING":  ["nato accelerate", "eu 800 billion defense", "nato emergency",
                          "massive rearmament", "eu defense fund", "nato surge"],
    },
}


class GeopoliticalScorer:
    """
    Fetches RSS feeds, scores news articles, updates geo variables,
    and computes sector geo scores.
    """

    def __init__(self):
        self._sector_scores: Dict[str, float] = {}
        self._sector_scores_yesterday: Dict[str, float] = {}
        self._recent_news: List[NewsItem] = []
        self._lock = threading.RLock()
        self._last_refresh: Optional[datetime] = None
        self._newsapi_key: Optional[str] = None
        self._load_initial_state()

    def _load_initial_state(self) -> None:
        """Seed sector scores from base_score config."""
        from config import SECTOR_CONFIG, GEO_VARIABLES_DEFAULT
        for sector, cfg in SECTOR_CONFIG.items():
            self._sector_scores[sector] = cfg["base_score"]
            self._sector_scores_yesterday[sector] = cfg["base_score"]

        # Seed geo states if not in DB yet
        existing = get_geo_states()
        for var, default_val in GEO_VARIABLES_DEFAULT.items():
            if var not in existing:
                save_geo_state(var, default_val, "Initial system state", "Default value")

        # Check for NewsAPI key
        import os
        self._newsapi_key = os.getenv("NEWSAPI_KEY")

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Fetch all RSS feeds, score articles, update states and scores."""
        logger.info("GeoScorer: starting refresh")
        items = self._fetch_all_feeds()
        items += self._fetch_newsapi()

        with self._lock:
            # Save yesterday's scores
            self._sector_scores_yesterday = dict(self._sector_scores)

            # Process new articles
            processed = []
            for item in items:
                if not news_already_cached(item["hash"]):
                    news_item = self._score_article(item)
                    if news_item:
                        processed.append(news_item)
                        save_news_cache(
                            url_hash=news_item.url_hash,
                            title=news_item.title,
                            summary=news_item.summary,
                            source=news_item.source,
                            published=news_item.published,
                            sectors_matched=json.dumps(news_item.sectors_matched),
                            sentiment_delta=news_item.sentiment_delta,
                        )
                        # Check for geo variable updates
                        self._infer_geo_variable(news_item)

            # Recompute sector scores from last 48h of news
            self._recompute_sector_scores()

            # Load recent news for display
            self._recent_news = self._load_recent_news_items()
            self._last_refresh = datetime.now(timezone.utc)

        logger.info("GeoScorer refresh complete — %d new articles processed", len(processed))

    def get_geo_context(self) -> GeoContext:
        """Return full geopolitical context for dashboard."""
        with self._lock:
            geo_db = get_geo_states()
            variables = {}
            now = datetime.now(timezone.utc)
            for var, row in geo_db.items():
                changed_hours = None
                if row.last_changed:
                    lc = row.last_changed
                    if lc.tzinfo is None:
                        lc = lc.replace(tzinfo=timezone.utc)
                    changed_hours = (now - lc).total_seconds() / 3600
                variables[var] = {
                    "value":              row.current_value,
                    "previous":          row.previous_value,
                    "changed_hours_ago": changed_hours,
                    "headline":          row.triggering_headline,
                    "impact":            row.impact_summary,
                }

            score_changes = {
                sector: self._sector_scores.get(sector, 0) -
                         self._sector_scores_yesterday.get(sector, 0)
                for sector in self._sector_scores
            }

            return GeoContext(
                variables=variables,
                sector_scores=dict(self._sector_scores),
                sector_score_changes=score_changes,
                recent_news=list(self._recent_news),
                last_updated=self._last_refresh,
            )

    def get_sector_score(self, sector: str) -> float:
        with self._lock:
            from config import SECTOR_CONFIG
            return self._sector_scores.get(sector, SECTOR_CONFIG.get(sector, {}).get("base_score", 5.0))

    def last_refresh_time(self) -> Optional[datetime]:
        with self._lock:
            return self._last_refresh

    # ─────────────────────────────────────────────────────────
    # RSS FETCHING
    # ─────────────────────────────────────────────────────────

    def _fetch_all_feeds(self) -> List[dict]:
        from config import RSS_FEEDS
        items = []
        for feed_cfg in RSS_FEEDS:
            try:
                feed_items = self._fetch_feed(feed_cfg["url"], feed_cfg["name"])
                items.extend(feed_items)
            except Exception as e:
                logger.warning("RSS fetch failed for %s: %s", feed_cfg["name"], e)
        return items

    def _fetch_feed(self, url: str, source: str) -> List[dict]:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except Exception:
            # Let feedparser try directly
            feed = feedparser.parse(url)

        items = []
        for entry in feed.entries[:30]:  # Max 30 per feed
            title   = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            link    = getattr(entry, "link", "") or ""
            pub     = getattr(entry, "published_parsed", None)

            published = None
            if pub:
                try:
                    import calendar
                    published = datetime.fromtimestamp(
                        calendar.timegm(pub), tz=timezone.utc
                    )
                except Exception:
                    pass

            url_hash = hashlib.sha256(
                (title + link).encode("utf-8", errors="ignore")
            ).hexdigest()[:32]

            items.append({
                "title": title[:500],
                "summary": summary[:1000],
                "source": source,
                "published": published,
                "hash": url_hash,
            })
        return items

    def _fetch_newsapi(self) -> List[dict]:
        """Optional NewsAPI.org enrichment (requires NEWSAPI_KEY env var)."""
        if not self._newsapi_key:
            return []
        try:
            url = (
                "https://newsapi.org/v2/top-headlines"
                "?category=general&pageSize=50"
                f"&apiKey={self._newsapi_key}"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = []
            for art in data.get("articles", []):
                title   = art.get("title", "") or ""
                summary = art.get("description", "") or ""
                url_hash = hashlib.sha256(
                    title.encode("utf-8", errors="ignore")
                ).hexdigest()[:32]
                pub = art.get("publishedAt")
                published = None
                if pub:
                    try:
                        published = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    except Exception:
                        pass
                items.append({
                    "title": title[:500],
                    "summary": summary[:1000],
                    "source": art.get("source", {}).get("name", "NewsAPI"),
                    "published": published,
                    "hash": url_hash,
                })
            logger.info("NewsAPI: fetched %d articles", len(items))
            return items
        except Exception as e:
            logger.warning("NewsAPI fetch failed: %s", e)
            return []

    # ─────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────

    def _score_article(self, item: dict) -> Optional[NewsItem]:
        """Score article against all sector trigger keywords."""
        from config import SECTOR_CONFIG
        text = (item["title"] + " " + item["summary"]).lower()
        sectors_matched = []
        total_delta = 0.0

        for sector, cfg in SECTOR_CONFIG.items():
            pos_hits = sum(1 for kw in cfg["primary_triggers"]  if kw in text)
            neg_hits = sum(1 for kw in cfg["negative_triggers"] if kw in text)

            if pos_hits == 0 and neg_hits == 0:
                continue

            # Sentiment delta: -2 to +2
            delta = min(pos_hits, 3) * 0.5 - min(neg_hits, 3) * 0.6
            delta = max(-2.0, min(2.0, delta))
            total_delta += delta
            sectors_matched.append(sector)

        if not sectors_matched:
            return None

        return NewsItem(
            title=item["title"],
            summary=item["summary"],
            source=item["source"],
            published=item["published"],
            url_hash=item["hash"],
            sectors_matched=sectors_matched,
            sentiment_delta=round(total_delta, 3),
        )

    def _infer_geo_variable(self, item: NewsItem) -> None:
        """Try to update geo variable states from news article."""
        text = (item.title + " " + item.summary).lower()
        for var, state_map in _GEO_VARIABLE_SIGNALS.items():
            for state, keywords in state_map.items():
                hits = sum(1 for kw in keywords if kw in text)
                if hits >= 2:  # Need at least 2 keyword hits for confidence
                    changed = save_geo_state(
                        variable=var,
                        value=state,
                        headline=item.title[:200],
                        impact=self._describe_impact(var, state),
                    )
                    if changed:
                        logger.info(
                            "Geo variable %s changed to %s via: %s",
                            var, state, item.title[:80]
                        )
                    break  # First matching state wins

    @staticmethod
    def _describe_impact(variable: str, value: str) -> str:
        impacts = {
            ("HORMUZ_STATUS",      "CLOSED"):       "ENERGY +++ | GOLD ++ | DEFENSE +",
            ("HORMUZ_STATUS",      "PARTIAL"):      "ENERGY ++ | GOLD +",
            ("HORMUZ_STATUS",      "OPEN"):         "ENERGY -- | GOLD -",
            ("IRAN_CONFLICT",      "ACTIVE"):       "ENERGY ++ | GOLD ++ | DEFENSE +",
            ("IRAN_CONFLICT",      "CEASEFIRE"):    "ENERGY - | GOLD -",
            ("IRAN_CONFLICT",      "RESOLVED"):     "ENERGY --- | GOLD --",
            ("UKRAINE_WAR",        "ESCALATING"):   "DEFENSE +++ | GOLD ++ | ENERGY +",
            ("UKRAINE_WAR",        "STALEMATE"):    "DEFENSE + | GOLD +",
            ("UKRAINE_WAR",        "DE-ESCALATING"):"DEFENSE -- | GOLD -",
            ("UKRAINE_WAR",        "RESOLVED"):     "DEFENSE --- | GOLD --",
            ("US_CHINA_RELATIONS", "HOSTILE"):      "METALS ++ | GOLD ++ | DEFENSE +",
            ("US_CHINA_RELATIONS", "TENSE"):        "METALS + | GOLD +",
            ("US_CHINA_RELATIONS", "NEUTRAL"):      "METALS -",
            ("US_CHINA_RELATIONS", "COOPERATIVE"):  "METALS --",
            ("NATO_SPENDING",      "ACCELERATING"): "DEFENSE +++",
            ("NATO_SPENDING",      "INCREASING"):   "DEFENSE ++",
            ("NATO_SPENDING",      "STABLE"):       "DEFENSE 0",
            ("NATO_SPENDING",      "DECLINING"):    "DEFENSE ---",
        }
        return impacts.get((variable, value), "Impact unknown")

    def _recompute_sector_scores(self) -> None:
        """
        Aggregate last 48h of news into sector geo scores.
        Maximum drift from base_score: ±2.5 points.
        """
        from config import SECTOR_CONFIG
        recent = get_recent_news(hours=48, limit=500)

        sector_deltas: Dict[str, List[float]] = {s: [] for s in SECTOR_CONFIG}

        for row in recent:
            try:
                sectors = json.loads(row.sectors_matched or "[]")
                for sector in sectors:
                    if sector in sector_deltas:
                        sector_deltas[sector].append(row.sentiment_delta or 0)
            except Exception:
                pass

        # Apply geo variable overrides on top of news sentiment
        geo_db = get_geo_states()

        for sector, cfg in SECTOR_CONFIG.items():
            base = cfg["base_score"]
            deltas = sector_deltas.get(sector, [])

            # News-driven delta (average, scaled)
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                # Scale: avg_delta of ±2 maps to ±2.5 score drift
                news_drift = avg_delta * 1.25
            else:
                news_drift = 0.0

            # Geo variable overrides
            var_drift = self._compute_var_drift(sector, geo_db)

            total_drift = news_drift + var_drift
            total_drift = max(-2.5, min(2.5, total_drift))

            new_score = round(max(0.0, min(10.0, base + total_drift)), 2)
            self._sector_scores[sector] = new_score
            save_sector_score(sector, new_score)

    def _compute_var_drift(self, sector: str, geo_db: dict) -> float:
        """Apply geo variable state rules to sector score drift."""
        drift = 0.0

        hormuz   = geo_db.get("HORMUZ_STATUS",      None)
        iran     = geo_db.get("IRAN_CONFLICT",       None)
        ukraine  = geo_db.get("UKRAINE_WAR",         None)
        us_china = geo_db.get("US_CHINA_RELATIONS",  None)
        nato     = geo_db.get("NATO_SPENDING",        None)

        def val(row): return row.current_value if row else None

        if sector == "ENERGY":
            v = val(hormuz)
            if v == "CLOSED":   drift += 2.0
            elif v == "PARTIAL": drift += 1.0
            elif v == "OPEN":    drift -= 1.0
            v = val(iran)
            if v == "ACTIVE":   drift += 0.5
            elif v == "RESOLVED": drift -= 1.5

        elif sector == "DEFENSE":
            v = val(ukraine)
            if v == "ESCALATING":    drift += 1.5
            elif v == "DE-ESCALATING": drift -= 1.0
            elif v == "RESOLVED":      drift -= 2.0
            v = val(nato)
            if v == "ACCELERATING": drift += 1.5
            elif v == "INCREASING":  drift += 0.5
            elif v == "DECLINING":   drift -= 2.0

        elif sector == "METALS":
            v = val(us_china)
            if v == "HOSTILE":     drift += 1.5
            elif v == "TENSE":      drift += 0.5
            elif v == "NEUTRAL":    drift -= 0.5
            elif v == "COOPERATIVE": drift -= 1.5

        elif sector == "GOLD":
            v = val(hormuz)
            if v == "CLOSED":   drift += 1.0
            elif v == "OPEN":    drift -= 0.5
            v = val(ukraine)
            if v == "ESCALATING": drift += 0.5
            elif v == "RESOLVED":  drift -= 1.0

        return drift

    def _load_recent_news_items(self) -> List[NewsItem]:
        rows = get_recent_news(hours=48, limit=20)
        result = []
        for row in rows:
            try:
                sectors = json.loads(row.sectors_matched or "[]")
            except Exception:
                sectors = []
            pub = row.published
            if pub and pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            result.append(NewsItem(
                title=row.title,
                summary=row.summary or "",
                source=row.source or "",
                published=pub,
                url_hash=row.url_hash,
                sectors_matched=sectors,
                sentiment_delta=row.sentiment_delta or 0.0,
            ))
        return result

    # ─────────────────────────────────────────────────────────
    # BACKGROUND REFRESH
    # ─────────────────────────────────────────────────────────

    def start_background_refresh(self) -> None:
        from config import GEO_REFRESH_INTERVAL
        self.refresh()   # Immediate first run

        def _loop():
            while True:
                time.sleep(GEO_REFRESH_INTERVAL)
                try:
                    self.refresh()
                except Exception as e:
                    logger.error("GeoScorer background refresh error: %s", e)

        t = threading.Thread(target=_loop, daemon=True, name="geo-refresh")
        t.start()
        logger.info("GeoScorer background refresh started")


# Module-level singleton
_geo_scorer: Optional[GeopoliticalScorer] = None


def get_geo_scorer() -> GeopoliticalScorer:
    global _geo_scorer
    if _geo_scorer is None:
        _geo_scorer = GeopoliticalScorer()
    return _geo_scorer
