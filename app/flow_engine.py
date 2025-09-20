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
    """Load and validate a flow YAML file"""
    try:
        flow_path = os.path.join(FLOWS_DIR, f"{name}.yaml")
        if not os.path.exists(flow_path):
            print(f"[WARNING] Flow file not found: {flow_path}")
            return None
            
        with open(flow_path, "r", encoding="utf-8") as f:
            flow = yaml.safe_load(f)
        
        # Validate flow structure
        if not isinstance(flow, dict):
            print(f"[ERROR] Invalid flow format in {name}.yaml")
            return None
            
        required_fields = ["id", "title", "steps"]
        for field in required_fields:
            if field not in flow:
                print(f"[ERROR] Missing required field '{field}' in {name}.yaml")
                return None
        
        # Validate steps
        if not isinstance(flow["steps"], list) or len(flow["steps"]) == 0:
            print(f"[ERROR] No valid steps found in {name}.yaml")
            return None
            
        return flow
        
    except Exception as e:
        print(f"[ERROR] Failed to load flow {name}: {e}")
        return None

def start_session():
    return {
        "lang": "en",  # English only
        "flow_id": None,
        "step_idx": 0,
        "mood_pre": None,
        "mood_post": None,
        "_flow_cache": {},
        "conversation_history": [],
        "current_flow_data": None,
        "session_start_time": None,
        "last_activity": None,
    }

def _render_step(step, lang):
    if step.get("type") == "sequence":
        return "\n".join(step.get(lang, []))
    return step.get(lang, "")

def _handle_mood_response(text, session):
    """Handle user response to mood check-in prompts"""
    try:
        # Try to extract a number from the response
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            mood_score = int(numbers[0])
            if 0 <= mood_score <= 10:
                session["mood_post"] = mood_score
                session["awaiting_mood_response"] = False
                
                # Provide feedback based on mood change
                pre_mood = session.get("mood_pre", 5)  # Default to 5 if no pre-mood
                change = mood_score - pre_mood
                
                if change > 0:
                    return f"Great! You went from {pre_mood} to {mood_score}. That's improvement! 🌟"
                elif change == 0:
                    return f"You're holding steady at {mood_score}. That's okay - sometimes staying the same is progress too."
                else:
                    return f"You're at {mood_score} now. It's okay to have ups and downs. How can I help you further?"
        
        # If no valid number found, ask again
        return "Please give me a number from 0-10 for how you're feeling now."
        
    except Exception as e:
        print(f"[ERROR] Failed to handle mood response: {e}")
        return "I didn't catch that. Please give me a number from 0-10."

def _reset_flow(session):
    """Reset the current flow and clear flow-related session data"""
    session["flow_id"] = None
    session["step_idx"] = 0
    session["_flow_cache"] = {}
    session["current_flow_data"] = None
    session["awaiting_mood_response"] = False

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

def route_message(text, session, lang="en"):
    import time
    
    # Update session state - English only
    session["lang"] = "en"
    session["last_activity"] = time.time()
    
    # Record conversation history
    session["conversation_history"].append({
        "timestamp": time.time(),
        "user_input": text,
        "language": "en"
    })

    # Handle mood responses (if user is responding to a mood prompt)
    if session.get("awaiting_mood_response"):
        return _handle_mood_response(text, session)

    # Pick a flow if none active
    if not session.get("flow_id"):
        intent = guess_intent(text)
        target = intent_to_flow(intent)
        
        if target is None:
            return "I understand you need support. Tell me if you want **breathing exercises**, **grounding techniques**, an **affirmation**, or a **journal prompt**. You can also just say how you're feeling."
        
        session["flow_id"] = target
        session["step_idx"] = 0
        session["current_flow_data"] = None

    # Load and validate flow
    flow = _ensure_flow_loaded(session)
    if not flow:
        return "(no flow) try: breathing / grounding / affirmation / journal"

    # Check if flow is complete
    steps = flow.get("steps", [])
    if session["step_idx"] >= len(steps):
        msg = "We finished this flow. Type breathing / grounding / affirmation / journal to try another."
        _reset_flow(session)
        return msg

    # Get current step
    step = steps[session["step_idx"]]
    session["step_idx"] += 1
    
    # Handle different step types
    if step.get("type") == "prompt" and "mood" in step.get("id", ""):
        session["awaiting_mood_response"] = True
        return _render_step(step, "en")
    else:
        session["awaiting_mood_response"] = False
        out = _render_step(step, "en")
        return out or "(empty step)"

def get_available_flows():
    """Get list of available flow files"""
    flows = []
    if os.path.exists(FLOWS_DIR):
        for file in os.listdir(FLOWS_DIR):
            if file.endswith('.yaml'):
                flow_name = file[:-5]  # Remove .yaml extension
                flows.append(flow_name)
    return flows

def validate_flow_exists(flow_id):
    """Check if a flow file exists and is valid"""
    if not flow_id:
        return False
    flow = load_flow(flow_id)
    return flow is not None
