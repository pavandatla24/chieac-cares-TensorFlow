import csv
import re

CRISIS_KEYWORDS = []
with open("nlu/data/crisis_keywords.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        CRISIS_KEYWORDS.append(row["keyword"].strip().lower())

# Enhanced crisis detection patterns
HIGH_RISK_PHRASES = [
    "kill myself", "end my life", "overdose", "jump off", "hang myself", "suicide",
    "take my life", "end it all", "not worth living", "better off dead",
    "want to die", "ready to die", "planning to die", "going to kill myself"
]

MEDIUM_RISK_PHRASES = [
    "self harm", "cut myself", "hurt myself", "can't go on", "cant go on", 
    "want to hurt", "thinking of hurting", "harm myself", "self-injury",
    "don't want to live", "dont want to live", "tired of living",
    "thinking of cutting", "cutting myself"  # Added missing patterns
]

# Contextual patterns that might indicate crisis
CRISIS_CONTEXT_PATTERNS = [
    r"i (really )?want to (die|end it|kill myself)",
    r"(thinking|planning) (about )?(suicide|killing myself|ending it)",
    r"life (isn't|is not) worth (living|it)",
    r"(can't|cant) (take|handle) (this|it) (anymore|any more)",
    r"(everyone|everybody) (would be|will be) better (off|without me)"
]

def check_crisis(text, lang="en"):
    """Enhanced crisis detection with multiple layers of analysis"""
    t = text.lower().strip()
    
    # Skip if text is too short or just numbers
    if len(t) < 3 or t.isdigit():
        return None
    
    # Check for false positives first (common phrases that contain crisis words)
    false_positive_patterns = [
        r"die laughing", r"dying to", r"kill time", r"kill the mood",
        r"suicide mission", r"suicide squad", r"die hard", r"die trying"
    ]
    
    for pattern in false_positive_patterns:
        if re.search(pattern, t):
            return None  # Skip crisis detection for false positives
    
    # Check for high-risk phrases first (most urgent)
    for phrase in HIGH_RISK_PHRASES:
        if phrase in t:
            return _crisis_msg()
    
    # Check for medium-risk phrases
    for phrase in MEDIUM_RISK_PHRASES:
        if phrase in t:
            return _crisis_msg()
    
    # Check contextual patterns using regex
    for pattern in CRISIS_CONTEXT_PATTERNS:
        if re.search(pattern, t):
            return _crisis_msg()
    
    # Check individual crisis keywords from CSV
    for kw in CRISIS_KEYWORDS:
        if kw in t:
            return _crisis_msg()
    
    return None

def _crisis_msg() -> str:
    return "If you're in danger or might harm yourself, call/text 988 (U.S.) or your local emergency number."
