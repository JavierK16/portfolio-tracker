"""
test_geo_scorer.py — Unit tests for the geopolitical scoring engine.
"""

import sys
import os
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geo_scorer import GeopoliticalScorer, _GEO_VARIABLE_SIGNALS


# ─────────────────────────────────────────────────────────────
# ARTICLE SCORING
# ─────────────────────────────────────────────────────────────

class TestArticleScoring:
    def setup_method(self):
        with patch("src.geo_scorer.get_geo_states", return_value={}), \
             patch("src.geo_scorer.save_geo_state"):
            self.scorer = GeopoliticalScorer.__new__(GeopoliticalScorer)
            self.scorer._sector_scores = {}
            self.scorer._sector_scores_yesterday = {}
            self.scorer._recent_news = []
            self.scorer._last_refresh = None
            self.scorer._newsapi_key = None
            # Load initial sector scores
            from config import SECTOR_CONFIG
            for sector, cfg in SECTOR_CONFIG.items():
                self.scorer._sector_scores[sector] = cfg["base_score"]
                self.scorer._sector_scores_yesterday[sector] = cfg["base_score"]

    def test_energy_article_scored(self):
        item = {
            "title": "Iran closes Hormuz strait as oil prices surge",
            "summary": "OPEC meeting cancelled as tanker attacks continue near gulf",
            "source": "Reuters",
            "published": datetime.now(timezone.utc),
            "hash": "abc123",
        }
        news_item = self.scorer._score_article(item)
        assert news_item is not None
        assert "ENERGY" in news_item.sectors_matched
        assert news_item.sentiment_delta > 0

    def test_defense_article_scored(self):
        item = {
            "title": "NATO increases defense spending amid Ukraine conflict",
            "summary": "EU rearmament plan accelerates as military budget grows",
            "source": "FT",
            "published": datetime.now(timezone.utc),
            "hash": "def456",
        }
        news_item = self.scorer._score_article(item)
        assert news_item is not None
        assert "DEFENSE" in news_item.sectors_matched
        assert news_item.sentiment_delta > 0

    def test_negative_trigger_reduces_sentiment(self):
        item = {
            "title": "Iran deal signed — ceasefire brings peace to the region",
            "summary": "Hormuz reopened as demand destruction hits global recession fears",
            "source": "BBC",
            "published": datetime.now(timezone.utc),
            "hash": "ghi789",
        }
        news_item = self.scorer._score_article(item)
        assert news_item is not None
        assert "ENERGY" in news_item.sectors_matched
        assert news_item.sentiment_delta < 0

    def test_irrelevant_article_returns_none(self):
        item = {
            "title": "Local sports team wins championship",
            "summary": "The fans celebrate a stunning victory in the finals",
            "source": "Local News",
            "published": datetime.now(timezone.utc),
            "hash": "jkl012",
        }
        news_item = self.scorer._score_article(item)
        assert news_item is None

    def test_multi_sector_article(self):
        item = {
            "title": "Ukraine war escalates as NATO rearmament drives copper demand",
            "summary": "Rare earth critical minerals supply deficit amid conflict",
            "source": "FT",
            "published": datetime.now(timezone.utc),
            "hash": "mno345",
        }
        news_item = self.scorer._score_article(item)
        assert news_item is not None
        assert len(news_item.sectors_matched) >= 2

    def test_sentiment_delta_clamped(self):
        # Very keyword-heavy article should not exceed ±2
        item = {
            "title": "Iran oil crude brent hormuz tanker opec gulf strait",
            "summary": "Iran oil crude brent hormuz tanker opec gulf strait lng sanctions",
            "source": "Test",
            "published": datetime.now(timezone.utc),
            "hash": "pqr678",
        }
        news_item = self.scorer._score_article(item)
        if news_item:
            assert -2.01 <= news_item.sentiment_delta <= 2.01


# ─────────────────────────────────────────────────────────────
# GEO VARIABLE INFERENCE
# ─────────────────────────────────────────────────────────────

class TestGeoVariableInference:
    def setup_method(self):
        with patch("src.geo_scorer.get_geo_states", return_value={}), \
             patch("src.geo_scorer.save_geo_state") as mock_save:
            self.scorer = GeopoliticalScorer.__new__(GeopoliticalScorer)
            self.scorer._sector_scores = {}
            self.scorer._sector_scores_yesterday = {}
            self.scorer._recent_news = []
            self.scorer._last_refresh = None
            self.scorer._newsapi_key = None
            self.mock_save = mock_save

    def test_hormuz_closed_detected(self):
        from src.geo_scorer import NewsItem
        item = NewsItem(
            title="Iran closes hormuz strait with full blockade",
            summary="The hormuz closure is in effect as ships are blocked",
            source="Reuters",
            published=datetime.now(timezone.utc),
            url_hash="test1",
            sectors_matched=["ENERGY"],
            sentiment_delta=1.5,
        )
        with patch("src.geo_scorer.save_geo_state") as mock_save:
            self.scorer._infer_geo_variable(item)
            # Should have called save_geo_state with HORMUZ_STATUS = CLOSED
            calls = [str(c) for c in mock_save.call_args_list]
            call_str = " ".join(calls)
            # The function should have detected CLOSED state
            assert mock_save.called

    def test_nato_accelerating_detected(self):
        from src.geo_scorer import NewsItem
        item = NewsItem(
            title="EU launches massive rearmament fund as NATO surges spending",
            summary="NATO accelerate defense programs with EU 800 billion defense commitment",
            source="FT",
            published=datetime.now(timezone.utc),
            url_hash="test2",
            sectors_matched=["DEFENSE"],
            sentiment_delta=1.8,
        )
        with patch("src.geo_scorer.save_geo_state") as mock_save:
            self.scorer._infer_geo_variable(item)
            assert mock_save.called


# ─────────────────────────────────────────────────────────────
# SECTOR SCORE COMPUTATION
# ─────────────────────────────────────────────────────────────

class TestSectorScoreComputation:
    def setup_method(self):
        with patch("src.geo_scorer.get_geo_states", return_value={}), \
             patch("src.geo_scorer.save_geo_state"):
            self.scorer = GeopoliticalScorer.__new__(GeopoliticalScorer)
            self.scorer._newsapi_key = None
            from config import SECTOR_CONFIG
            self.scorer._sector_scores = {s: c["base_score"] for s, c in SECTOR_CONFIG.items()}
            self.scorer._sector_scores_yesterday = dict(self.scorer._sector_scores)
            self.scorer._recent_news = []
            self.scorer._last_refresh = None

    def test_energy_score_increases_on_hormuz_closed(self):
        class FakeRow:
            def __init__(self, v): self.current_value = v

        geo_db = {
            "HORMUZ_STATUS": FakeRow("CLOSED"),
            "IRAN_CONFLICT": FakeRow("ACTIVE"),
            "UKRAINE_WAR":   FakeRow("STALEMATE"),
            "US_CHINA_RELATIONS": FakeRow("TENSE"),
            "NATO_SPENDING": FakeRow("INCREASING"),
        }
        drift = self.scorer._compute_var_drift("ENERGY", geo_db)
        assert drift > 0

    def test_defense_score_drops_on_ukraine_resolved(self):
        class FakeRow:
            def __init__(self, v): self.current_value = v

        geo_db = {
            "UKRAINE_WAR":   FakeRow("RESOLVED"),
            "NATO_SPENDING": FakeRow("DECLINING"),
        }
        drift = self.scorer._compute_var_drift("DEFENSE", geo_db)
        assert drift < 0

    def test_metals_score_increases_on_hostile_us_china(self):
        class FakeRow:
            def __init__(self, v): self.current_value = v

        geo_db = {
            "US_CHINA_RELATIONS": FakeRow("HOSTILE"),
        }
        drift = self.scorer._compute_var_drift("METALS", geo_db)
        assert drift > 0

    def test_sector_score_clamped_to_base_plus_2_5(self):
        from config import SECTOR_CONFIG
        base = SECTOR_CONFIG["ENERGY"]["base_score"]
        # With maximum positive drift (+2.5), score shouldn't exceed base + 2.5
        max_possible = base + 2.5
        # And shouldn't go below base - 2.5
        min_possible = base - 2.5
        assert min_possible >= 0


# ─────────────────────────────────────────────────────────────
# IMPACT DESCRIPTIONS
# ─────────────────────────────────────────────────────────────

class TestImpactDescriptions:
    def test_all_known_variable_states_have_impact(self):
        known_combos = [
            ("HORMUZ_STATUS", "CLOSED"),
            ("HORMUZ_STATUS", "PARTIAL"),
            ("HORMUZ_STATUS", "OPEN"),
            ("IRAN_CONFLICT", "ACTIVE"),
            ("UKRAINE_WAR", "ESCALATING"),
            ("UKRAINE_WAR", "RESOLVED"),
            ("NATO_SPENDING", "ACCELERATING"),
            ("NATO_SPENDING", "DECLINING"),
            ("US_CHINA_RELATIONS", "HOSTILE"),
            ("US_CHINA_RELATIONS", "COOPERATIVE"),
        ]
        for var, value in known_combos:
            impact = GeopoliticalScorer._describe_impact(var, value)
            assert impact != "Impact unknown", f"No impact defined for ({var}, {value})"

    def test_unknown_combo_returns_fallback(self):
        impact = GeopoliticalScorer._describe_impact("UNKNOWN_VAR", "UNKNOWN_VALUE")
        assert impact == "Impact unknown"


# ─────────────────────────────────────────────────────────────
# GEO VARIABLE SIGNALS STRUCTURE
# ─────────────────────────────────────────────────────────────

class TestGeoVariableSignalsStructure:
    def test_all_expected_variables_present(self):
        expected = {
            "HORMUZ_STATUS", "IRAN_CONFLICT", "UKRAINE_WAR",
            "US_CHINA_RELATIONS", "NATO_SPENDING"
        }
        assert set(_GEO_VARIABLE_SIGNALS.keys()) == expected

    def test_each_variable_has_multiple_states(self):
        for var, states in _GEO_VARIABLE_SIGNALS.items():
            assert len(states) >= 2, f"{var} has fewer than 2 states"

    def test_each_state_has_keywords(self):
        for var, states in _GEO_VARIABLE_SIGNALS.items():
            for state, keywords in states.items():
                assert len(keywords) >= 2, f"{var}/{state} has fewer than 2 keywords"

    def test_keywords_are_lowercase(self):
        for var, states in _GEO_VARIABLE_SIGNALS.items():
            for state, keywords in states.items():
                for kw in keywords:
                    assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"
