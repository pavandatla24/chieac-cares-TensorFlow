import os
import csv
import time
from typing import Dict, Any


LOG_DIR = os.path.join("data", "logs")
SESSION_LOG = os.path.join(LOG_DIR, "sessions.csv")


def _ensure_dirs() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _ensure_headers(path: str, fieldnames: list[str]) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def log_session_event(session: Dict[str, Any], event_type: str, data: Dict[str, Any] | None = None) -> None:
    """Append a single event row to the session CSV log.

    Fields: timestamp, event, flow_id, step_idx, mood_pre, mood_post, change, details
    """
    try:
        _ensure_dirs()
        fieldnames = [
            "timestamp",
            "event",
            "flow_id",
            "step_idx",
            "mood_pre",
            "mood_post",
            "mood_change",
            "details",
        ]
        _ensure_headers(SESSION_LOG, fieldnames)

        details = data or {}
        row = {
            "timestamp": int(time.time()),
            "event": event_type,
            "flow_id": (session or {}).get("flow_id"),
            "step_idx": (session or {}).get("step_idx", 0),
            "mood_pre": (session or {}).get("mood_pre"),
            "mood_post": (session or {}).get("mood_post"),
            "mood_change": _compute_mood_change(session),
            "details": _compact_details(details),
        }
        with open(SESSION_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
    except Exception:
        # Logging must never break the chat
        pass


def _compute_mood_change(session: Dict[str, Any] | None) -> int | None:
    if not session:
        return None
    pre = session.get("mood_pre")
    post = session.get("mood_post")
    if isinstance(pre, int) and isinstance(post, int):
        return post - pre
    return None


def _compact_details(details: Dict[str, Any]) -> str:
    try:
        # Keep it simple and CSV-friendly
        pairs = []
        for k, v in (details or {}).items():
            pairs.append(f"{k}={v}")
        return ";".join(pairs)
    except Exception:
        return ""


