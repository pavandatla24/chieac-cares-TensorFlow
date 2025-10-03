import csv

CRISIS = []
with open("nlu/data/crisis_keywords.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        CRISIS.append((row["keyword"].strip().lower(), row["lang"].strip()))

def check_crisis(text, lang="en"):
    """Return crisis message if text matches crisis patterns.

    Uses simple keyword matching with basic severity tiers.
    """
    t = text.lower()

    # Severity tiers (simple heuristics)
    high_risk = [
        "kill myself", "suicide", "end my life", "hurt myself badly",
        "overdose", "jump off", "hang myself",
    ]
    medium_risk = [
        "self harm", "hurt myself", "cut myself", "want to die",
        "can't go on", "cant go on", "die right now",
    ]

    # Direct high-risk phrase detection
    for phrase in high_risk:
        if phrase in t:
            return _crisis_msg()

    # Keyword list from CSV (lang-aware)
    for kw, l in CRISIS:
        if l == lang and kw in t:
            return _crisis_msg()

    # Medium risk fallthrough
    for phrase in medium_risk:
        if phrase in t:
            return _crisis_msg()

    return None


def _crisis_msg() -> str:
    return (
        "If you're in danger or might harm yourself, call/text 988 (U.S.) or your local emergency number."
    )
