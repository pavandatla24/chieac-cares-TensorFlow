import csv

CRISIS = []
with open("nlu/data/crisis_keywords.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        CRISIS.append((row["keyword"].strip().lower(), row["lang"].strip()))

def check_crisis(text, lang):
    t = text.lower()
    for kw, l in CRISIS:
        if l == lang and kw in t:
            return {
              "en":"If you’re in danger or might harm yourself, call/text 988 (U.S.) or your local emergency number.",
              "es":"Si estás en peligro o podrías lastimarte, llama/mensaja al 988 (EE. UU.) o a tu número local de emergencias."
            }[lang]
    return None
