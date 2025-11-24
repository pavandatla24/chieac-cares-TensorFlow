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
from app.storage_local import log_session_event

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
        "is_first_message": True,  # Track if this is the first message
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
                session["awaiting_mood_response"] = False

                # If no pre-mood recorded yet, treat this as baseline
                if session.get("mood_pre") is None:
                    session["mood_pre"] = mood_score
                    from app.storage_local import log_session_event
                    log_session_event(session, "mood_baseline")
                    
                    # Add mood-based recommendation for low scores
                    if mood_score <= 3:
                        return f"Thanks, noted your mood at {mood_score}/10. I can see you're having a really tough time. Would you like to try some breathing exercises or grounding techniques? You might also want to consider reaching out to a friend or family member for support."
                    elif mood_score <= 5:
                        return f"Thanks, noted your mood at {mood_score}/10. I'm here to help. Would you like to try some calming exercises, or is there something specific you'd like to work on?"
                    else:
                        return f"Thanks, noted your mood at {mood_score}/10. How can I help you today?"

                # Otherwise this is a post-mood update
                session["mood_post"] = mood_score

                # Provide feedback based on mood change
                pre_mood = session.get("mood_pre")
                try:
                    change = mood_score - int(pre_mood)
                except Exception:
                    change = None
                
                from app.storage_local import log_session_event
                log_session_event(session, "mood_update")

                if change is None:
                    return f"Got it — you're at {mood_score}/10 now. Do you need help with anything else?"
                elif change > 0:
                    return f"Great! You went from {pre_mood} to {mood_score}. That's improvement! 🌟 Do you need help with anything else?"
                elif change == 0:
                    if mood_score <= 4:
                        return f"You're holding steady at {mood_score}. Since your mood is still quite low, it might help to reach out to a friend, family member, or someone you trust. Do you need help with anything else?"
                    else:
                        return f"You're holding steady at {mood_score}. That's okay — sometimes staying the same is progress too. Do you need help with anything else?"
                else:
                    if mood_score <= 4:
                        return f"You're at {mood_score} now. Since your mood is still quite low, it might help to reach out to a friend, family member, or someone you trust. Do you need help with anything else?"
                    else:
                        return f"You're at {mood_score} now. It's okay to have ups and downs. Do you need help with anything else?"
        
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

def _should_advance_step(text, session):
    """
    Determine if user input should advance the current flow step.
    Returns: (should_advance: bool, reason: str, new_intent: str or None)
    """
    text_lower = text.lower().strip()
    
    # 1. Check for explicit continuation signals
    continuation_keywords = [
        "done", "okay", "ok", "yes", "next", "continue", "finished", 
        "ready", "got it", "all done", "i'm done", "completed", "let's continue",
        "i finished", "i completed", "sure", "yep", "yeah", "alright", "all right"
    ]
    if any(keyword in text_lower for keyword in continuation_keywords):
        return True, "continue", None
    
    # 2. Check for interruption signals
    interruption_keywords = ["stop", "cancel", "wait", "actually", "never mind", "change my mind", "skip"]
    if any(keyword in text_lower for keyword in interruption_keywords):
        return False, "interrupt", None
    
    # 3. Check if user is starting a NEW emotional state/intent
    new_intent = guess_intent(text)
    # Strong emotional intents that should interrupt current flow
    strong_intents = ["breathing_exercise", "grounding_exercise", "anger_management", 
                      "affirmation_request", "journal_prompt", "check_in_mood"]
    if new_intent in strong_intents:
        return False, "new_flow", new_intent
    
    # 4. Check for thanks/goodbye - should end current flow
    if new_intent == "thanks_goodbye":
        return False, "goodbye", None
    
    # 5. Invalid input - doesn't match continuation, interruption, or new intent
    # Note: We don't auto-advance on short gibberish - user must explicitly continue
    return False, "invalid", None

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

    # Mark that we've received the first message (welcome already shown by chat interface)
    if session.get("is_first_message", True):
        session["is_first_message"] = False

    # Handle control commands
    if isinstance(text, str) and text.strip().lower() in ("reset", "cancel", "restart"):
        try:
            log_session_event(session, "flow_reset")
        except Exception:
            pass
        _reset_flow(session)
        return "Reset. You can type breathing / grounding / affirmation / journal to start a new flow."

    # Handle mood responses (if user is responding to a mood prompt)
    if session.get("awaiting_mood_response"):
        response = _handle_mood_response(text, session)
        # After mood response, reset the flow so user can start a new one
        _reset_flow(session)
        return response

    # Pick a flow if none active
    flow_just_started = False
    if not session.get("flow_id"):
        intent = guess_intent(text)
        target = intent_to_flow(intent)
        
        if target is None:
            if intent == "thanks_goodbye":
                return "I'm glad I could help! Take care and have a good day. Type 'exit' when you're ready to close the chat."
            else:
                # Check if input looks like gibberish/invalid (short random characters, no meaningful words)
                text_lower = text.lower().strip()
                # Common emotional/support words that indicate valid input
                meaningful_words = ["i", "feel", "am", "need", "want", "help", "anxious", "sad", "angry", "breathing", "grounding", "affirmation", "journal", "panic", "overwhelmed", "numb", "disconnected", "irritated", "frustrated"]
                has_meaningful_content = any(word in text_lower for word in meaningful_words) or len(text_lower.split()) > 2
                
                # If it's short and has no meaningful words, it's likely gibberish
                if len(text_lower) <= 15 and not has_meaningful_content:
                    return "I didn't quite understand that. You can tell me how you're feeling, or ask for breathing exercises, grounding techniques, an affirmation, or a journal prompt."
                else:
                    return "I understand you need support. Tell me if you want **breathing exercises**, **grounding techniques**, an **affirmation**, or a **journal prompt**. You can also just say how you're feeling."
        
        session["flow_id"] = target
        session["step_idx"] = 0
        session["current_flow_data"] = None
        log_session_event(session, "flow_start", {"intent": intent})
        flow_just_started = True

    # Load and validate flow
    flow = _ensure_flow_loaded(session)
    if not flow:
        return "(no flow) try: breathing / grounding / affirmation / journal"

    # Check if flow is complete
    steps = flow.get("steps", [])
    if session["step_idx"] >= len(steps):
        # Build completion message with optional mood summary
        summary = ""
        pre = session.get("mood_pre")
        post = session.get("mood_post")
        if isinstance(pre, int) and isinstance(post, int):
            delta = post - pre
            arrow = "+" if delta > 0 else ("±" if delta == 0 else "-")
            summary = f" Mood change: {pre} -> {post} ({arrow}{abs(delta)})."
        msg = "We finished this flow. Type breathing / grounding / affirmation / journal to try another." + summary
        log_session_event(session, "flow_complete")
        _reset_flow(session)
        return msg

    # If flow just started, show first step immediately without validation
    if flow_just_started:
        first_step = steps[0]
        session["step_idx"] = 1  # Advance to next step index (will be shown after user responds)
        if first_step.get("type") == "prompt" and "mood" in first_step.get("id", ""):
            session["awaiting_mood_response"] = True
        else:
            session["awaiting_mood_response"] = False
        out = _render_step(first_step, "en")
        log_session_event(session, "flow_step", {"step_id": first_step.get("id")})
        return out or "(empty step)"

    # Get the step the user is responding to (previous step, since step_idx was advanced after showing it)
    # step_idx points to the NEXT step, so we need to go back one to get the current step
    responding_to_step_idx = max(0, session["step_idx"] - 1)
    step_user_is_responding_to = steps[responding_to_step_idx]
    
    # Validate user input before advancing
    should_advance, reason, new_intent = _should_advance_step(text, session)
    
    # Handle different validation outcomes
    if not should_advance:
        if reason == "new_flow":
            # User wants to start a different flow - interrupt current and start new one
            target = intent_to_flow(new_intent)
            if target:
                _reset_flow(session)
                session["flow_id"] = target
                session["step_idx"] = 0
                session["current_flow_data"] = None
                log_session_event(session, "flow_start", {"intent": new_intent, "interrupted_previous": True})
                # Load new flow and return first step
                new_flow = _ensure_flow_loaded(session)
                if new_flow:
                    first_step = new_flow.get("steps", [])[0]
                    session["step_idx"] = 1
                    if first_step.get("type") == "prompt" and "mood" in first_step.get("id", ""):
                        session["awaiting_mood_response"] = True
                    else:
                        session["awaiting_mood_response"] = False
                    return _render_step(first_step, "en")
                return "Starting a new flow. How can I help you?"
            else:
                # Invalid intent, stay on current step
                current_step_msg = _render_step(step_user_is_responding_to, "en")
                return f"I understand you want to change direction. When you're ready, you can type 'done' or 'okay' to continue with the current exercise, or tell me what you'd like to do instead.\n\n{current_step_msg}"
        
        elif reason == "interrupt":
            # User wants to stop/cancel
            _reset_flow(session)
            return "I understand. What would you like to do instead? You can try breathing exercises, grounding techniques, affirmations, or journal prompts. Or just tell me how you're feeling."
        
        elif reason == "goodbye":
            # User is saying goodbye
            _reset_flow(session)
            return "I'm glad I could help! Take care and have a good day. Type 'exit' when you're ready to close the chat."
        
        else:  # reason == "invalid"
            # Invalid input - don't advance, provide helpful feedback and repeat the step user was responding to
            current_step_msg = _render_step(step_user_is_responding_to, "en")
            # Try to provide context-aware help based on step type
            if step_user_is_responding_to.get("type") == "instruction":
                return f"I'm not sure what you meant. To continue, you can type 'done', 'okay', or 'next' when you're ready.\n\n{current_step_msg}"
            else:
                return f"I didn't quite understand that. You can type 'done' or 'okay' to continue, or tell me if you want to try something else.\n\n{current_step_msg}"
    
    # Input is valid - show current step, then advance for next time
    current_step = steps[session["step_idx"]]
    
    # Check if this is the last step
    if session["step_idx"] >= len(steps) - 1:
        # This is the last step, show it and then mark flow as complete
        if current_step.get("type") == "prompt" and "mood" in current_step.get("id", ""):
            session["awaiting_mood_response"] = True
        else:
            session["awaiting_mood_response"] = False
        out = _render_step(current_step, "en")
        log_session_event(session, "flow_step", {"step_id": current_step.get("id")})
        
        # Advance past the last step
        session["step_idx"] += 1
        return out or "(empty step)"
    
    # Not the last step - show current step, then advance
    if current_step.get("type") == "prompt" and "mood" in current_step.get("id", ""):
        session["awaiting_mood_response"] = True
    else:
        session["awaiting_mood_response"] = False
    out = _render_step(current_step, "en")
    log_session_event(session, "flow_step", {"step_id": current_step.get("id")})
    
    # Advance to next step for next interaction
    session["step_idx"] += 1
    
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
