"""
logging_setup.py — Configure application-wide logging with daily rotation.
Call setup_logging() once at application startup.
"""

import logging
import logging.handlers
import os


def setup_logging(level: int = logging.INFO) -> None:
    from config import APP_LOG, LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))

    # File handler with daily rotation
    fh = logging.handlers.TimedRotatingFileHandler(
        APP_LOG, when="midnight", backupCount=30, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    ))

    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)

    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("feedparser").setLevel(logging.WARNING)
