import re

KEYS = {
    "breathing_exercise": [r"\bbreath|respira", r"calm", r"panic|anx"],
    "grounding_exercise": [r"\bground|arraig|present|dissoci"],
    "affirmation_request": [r"affirm|positive|reassur|ánim|frase positiva"],
    "journal_prompt":     [r"journal|prompt|write|escribir|consigna"],
    "info_help":          [r"resource|support|¿qué recursos|recursos"],
    "self_assess":        [r"why am i|what's happening|esto es normal|por qué me siento|cerebro"],
    "language_switch":    [r"\bspanish\b|\bespañol\b|\binglés\b|\benglish\b|cambiar|switch"],
    "thanks_goodbye":     [r"\bthanks\b|\bgracias\b|goodnight|adiós|bye"],
    "check_in_mood":      [r"anxious|sad|angry|nerv|triste|enojad|numb|overwhelm|desconect"],
    "crisis_emergency":   [r"i want to die|kill myself|me quiero morir|matarme|lastimarme|hurt myself|overdose|sobredosis"],
    "greeting_start":     [r"\bhi\b|\bhello\b|\bhola\b|can we talk|podemos hablar|necesito chatear"],
}

INTENT_TO_FLOW = {
    "breathing_exercise": "panic",
    "check_in_mood": "panic",
    "grounding_exercise": "detachment",
    "affirmation_request": "sadness",
    "journal_prompt": "sadness",
    "greeting_start": "panic",
    # others -> None (we’ll handle gracefully)
}

def guess_intent(text: str) -> str:
    t = text.lower()
    for intent, pats in KEYS.items():
        for pat in pats:
            if re.search(pat, t):
                return intent
    return "fallback_clarify"

def intent_to_flow(intent: str):
    return INTENT_TO_FLOW.get(intent)
