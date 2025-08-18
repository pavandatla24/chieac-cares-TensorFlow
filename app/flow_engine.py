import yaml, os

def load_flow(name):
    with open(os.path.join("content","flows",f"{name}.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def start_session():
    return {"lang":"en", "state":None, "mood_pre":None, "mood_post":None}

def route_message(text, session, lang):
    # Week 4–6: replace with real NLU + state machine.
    # For Week 1 sanity check, echo a helper:
    return {"en":"(demo) try typing: 'breathing', 'grounding', or 'sad'", 
            "es":"(demo) escribe: 'respiración', 'arraigo' o 'triste'"}[lang if lang in ("en","es") else "en"]
