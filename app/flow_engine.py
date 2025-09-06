# import yaml, os

# def load_flow(name):
#     with open(os.path.join("content","flows",f"{name}.yaml"), "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)

# def start_session():
#     return {"lang":"en", "state":None, "mood_pre":None, "mood_post":None}

# def route_message(text, session, lang):
#     # Week 4–6: replace with real NLU + state machine.
#     # For Week 1 sanity check, echo a helper:
#     return {"en":"(demo) try typing: 'breathing', 'grounding', or 'sad'", 
#             "es":"(demo) escribe: 'respiración', 'arraigo' o 'triste'"}[lang if lang in ("en","es") else "en"]

import os, yaml
from app.router import guess_intent, intent_to_flow

FLOWS_DIR = os.path.join("content","flows")

def load_flow(name):
    with open(os.path.join(FLOWS_DIR, f"{name}.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def start_session():
    return {
        "lang": "en",
        "flow_id": None,
        "step_idx": 0,
        "mood_pre": None,
        "mood_post": None,
        "_flow_cache": {},
    }

def _render_step(step, lang):
    if step.get("type") == "sequence":
        return "\n".join(step.get(lang, []))
    return step.get(lang, "")

def _ensure_flow_loaded(session):
    fid = session.get("flow_id")
    cache = session.get("_flow_cache") or {}
    if fid and cache.get("id") == fid:
        return cache
    if not fid:
        return None
    flow = load_flow(fid)
    flow["id"] = fid
    session["_flow_cache"] = flow
    return flow

def route_message(text, session, lang):
    session["lang"] = lang if lang in ("en","es") else "en"

    # pick a flow if none active
    if not session.get("flow_id"):
        intent = guess_intent(text)
        target = intent_to_flow(intent)
        if target is None:
            return {
                "en":"Tell me if you want **breathing**, **grounding**, an **affirmation**, or a **journal** prompt.",
                "es":"Dime si quieres **respiración**, **arraigo**, una **afirmación** o un **prompt** de **escribir**."
            }[session["lang"]]
        session["flow_id"] = target
        session["step_idx"] = 0

    flow = _ensure_flow_loaded(session)
    if not flow:
        return {"en":"(no flow) try: breathing / grounding / affirmation / journal",
                "es":"(sin flujo) prueba: respiración / arraigo / afirmación / escribir"}[session["lang"]]

    steps = flow.get("steps", [])
    if session["step_idx"] >= len(steps):
        msg = {"en":"We finished this flow. Type breathing / grounding / affirmation / journal to try another.",
               "es":"Terminamos este flujo. Escribe respiración / arraigo / afirmación / escribir para otro."}[session["lang"]]
        session["flow_id"] = None
        session["step_idx"] = 0
        session["_flow_cache"] = {}
        return msg

    step = steps[session["step_idx"]]
    session["step_idx"] += 1
    out = _render_step(step, session["lang"])
    return out or {"en":"(empty step)","es":"(paso vacío)"}[session["lang"]]
