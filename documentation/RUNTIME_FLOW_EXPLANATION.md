# Complete Runtime Flow: From User Input to Bot Response

## 📊 Visual Flow Diagram

```
python -m app.chat
         ↓
    [app/chat.py]
         ↓
    start_session()
    Show Welcome Message
         ↓
    User Input: "I'm feeling anxious"
         ↓
    [app/safety.py]
         ↓
    check_crisis()
    (No crisis detected)
         ↓
    [app/flow_engine.py]
         ↓
    route_message()
         ↓
    [app/router.py]
         ↓
    guess_intent()
         ├── Try TensorFlow Model
         │   └── [nlu/export_en/english_nlu/inference.py]
         │       └── Load model weights
         │       └── Predict intent
         └── Fallback to Keywords
         ↓
    intent_to_flow()
    "check_in_mood" → "panic"
         ↓
    [app/flow_engine.py]
         ↓
    Load panic.yaml
    Show first step
         ↓
    User Input: "okay"
         ↓
    Validate Input
    Advance Step
    Show next step
         ↓
    [app/storage_local.py]
         ↓
    log_session_event()
    Save to CSV
         ↓
    Return Response to User
```

---

## 🔍 Step-by-Step Detailed Explanation

### **STEP 1: Starting the Chatbot**

**Command:** `python -m app.chat`

**File:** `app/chat.py`  
**Function:** `run()` (lines 5-30)

**What happens:**

1. **Initialize session:**
   ```python
   session = start_session()  # From flow_engine.py
   ```
   Creates session dictionary:
   ```python
   {
       "lang": "en",
       "flow_id": None,
       "step_idx": 0,
       "mood_pre": None,
       "mood_post": None,
       "awaiting_mood_response": False,
       "conversation_history": [],
       ...
   }
   ```

2. **Show welcome message:**
   ```python
   print("ChiEAC CARES (English). Type 'help' for tips...")
   print("bot: I'm here with you. Before we start...")
   ```

3. **Enter main loop:**
   ```python
   while True:
       user = input("you: ")
   ```

---

### **STEP 2: User Input Handling**

**File:** `app/chat.py`  
**Lines:** 11-20

**What happens:**

1. **Check for exit:**
   ```python
   if user.strip().lower() == "exit":
       print("bye. take care.")
       break
   ```

2. **Check for empty input:**
   ```python
   if not user.strip():
       print("bot: I'm here. You can type...")
       continue
   ```

3. **Check for help:**
   ```python
   if user.strip().lower() in ("help", "?", "tips"):
       print("bot: Try: 'breathing', 'grounding'...")
       continue
   ```

---

### **STEP 3: Crisis Detection**

**File:** `app/chat.py` (line 22)  
**Calls:** `app/safety.py` → `check_crisis()`

**What happens:**

```python
crisis = check_crisis(user, "en")
if crisis:
    print(Fore.RED + crisis)  # Show crisis resources
    continue  # Skip normal flow
```

**Inside `app/safety.py`:**

1. **Check false positives:**
   ```python
   false_positive_patterns = [
       r"die laughing", r"kill time", ...
   ]
   # If matches → return None (not a crisis)
   ```

2. **Check high-risk phrases:**
   ```python
   HIGH_RISK_PHRASES = [
       "kill myself", "end my life", "suicide", ...
   ]
   # If matches → return crisis message
   ```

3. **Check medium-risk phrases:**
   ```python
   MEDIUM_RISK_PHRASES = [
       "self harm", "hurt myself", ...
   ]
   # If matches → return crisis message
   ```

4. **Check contextual patterns:**
   ```python
   CRISIS_CONTEXT_PATTERNS = [
       r"i (really )?want to (die|end it|kill myself)",
       ...
   ]
   # Regex matching
   ```

5. **Check keywords from CSV:**
   ```python
   for kw in CRISIS_KEYWORDS:  # From crisis_keywords.csv
       if kw in text.lower():
           return crisis_message
   ```

**If crisis detected:**
- Returns: `"If you're in danger or might harm yourself, call/text 988..."`
- Bypasses normal conversation flow
- User sees emergency resources immediately

**If no crisis:**
- Returns `None`
- Continues to normal flow

---

### **STEP 4: Routing Message**

**File:** `app/chat.py` (line 26)  
**Calls:** `app/flow_engine.py` → `route_message()`

```python
bot = route_message(user, session, "en")
```

**Inside `app/flow_engine.py` → `route_message()` (lines 192-387):**

#### **4.1 Update Session State**
```python
session["lang"] = "en"
session["last_activity"] = time.time()
session["conversation_history"].append({
    "timestamp": time.time(),
    "user_input": text,
    "language": "en"
})
```

#### **4.2 Handle Control Commands**
```python
if text.strip().lower() in ("reset", "cancel", "restart"):
    _reset_flow(session)
    return "Reset. You can type..."
```

#### **4.3 Handle Mood Responses**
```python
if session.get("awaiting_mood_response"):
    response = _handle_mood_response(text, session)
    _reset_flow(session)
    return response
```

**Inside `_handle_mood_response()`:**
- Extracts number from text (0-10)
- Records as `mood_pre` or `mood_post`
- Calculates mood change
- Provides feedback based on score
- Logs to CSV

#### **4.4 Detect Intent (If No Flow Active)**

**File:** `app/flow_engine.py` (line 229)  
**Calls:** `app/router.py` → `guess_intent()`

```python
if not session.get("flow_id"):
    intent = guess_intent(text)  # Calls router.py
    target = intent_to_flow(intent)
```

---

### **STEP 5: Intent Detection**

**File:** `app/router.py`  
**Function:** `guess_intent(text)` (lines 108-122)

**What happens:**

#### **5.1 Try TensorFlow Model**

```python
model = get_nlu_model()  # Singleton instance
intent, confidence = model.predict(text)
```

**Inside `NLUModel.predict()` (lines 21-48):**

1. **Get inference script path:**
   ```python
   inference_script = os.path.join(self.model_dir, "inference.py")
   # Path: nlu/export_en/english_nlu/inference.py
   ```

2. **Run inference via subprocess:**
   ```python
   result = subprocess.run([
       sys.executable, 
       inference_script, 
       text  # "I'm feeling anxious"
   ], capture_output=True, text=True)
   ```

3. **Inside `inference.py`:**
   - Loads `model_info.json` → Rebuilds architecture
   - Loads `model_weights.weights.h5` → Loads trained weights
   - Loads `label_map.json` → Maps indices to labels
   - Runs prediction:
     ```python
     predictions = model.predict([text])
     intent_idx = np.argmax(predictions["intent"][0])
     intent = label_maps["intent"][str(intent_idx)]
     ```
   - Returns JSON: `{"intent": "check_in_mood", ...}`

4. **Parse output:**
   ```python
   if result.returncode == 0:
       # Extract intent from JSON output
       intent = parse_json_output(result.stdout)
       return intent, 0.8  # Confidence
   ```

#### **5.2 Fallback to Keywords (If Model Fails)**

**File:** `app/router.py`  
**Function:** `_fallback_intent()` (lines 50-73)

```python
def _fallback_intent(text):
    t = text.lower()
    
    if any(word in t for word in ["breath", "breathe", "panic", "anxious"]):
        return "breathing_exercise"
    elif any(word in t for word in ["ground", "dissociate", "numb"]):
        return "grounding_exercise"
    elif any(word in t for word in ["angry", "furious", "rage"]):
        return "anger_management"
    # ... more patterns
    else:
        return "fallback_clarify"
```

**Returns:** Intent string (e.g., `"check_in_mood"`)

---

### **STEP 6: Map Intent to Flow**

**File:** `app/router.py`  
**Function:** `intent_to_flow(intent)` (lines 150-151)

**What happens:**

```python
INTENT_TO_FLOW = {
    "breathing_exercise": "panic",
    "check_in_mood": "panic",
    "grounding_exercise": "detachment",
    "anger_management": "irritation",
    "affirmation_request": "sadness",
    "journal_prompt": "sadness",
    ...
}

def intent_to_flow(intent):
    return INTENT_TO_FLOW.get(intent)
```

**Example:**
- Input: `"check_in_mood"`
- Output: `"panic"` (flow name)

---

### **STEP 7: Start Flow**

**File:** `app/flow_engine.py`  
**Lines:** 228-275

**What happens:**

1. **Set flow in session:**
   ```python
   session["flow_id"] = target  # "panic"
   session["step_idx"] = 0
   flow_just_started = True
   ```

2. **Load flow YAML:**
   ```python
   flow = load_flow("panic")  # Loads content/flows/panic.yaml
   ```

3. **Show first step immediately:**
   ```python
   if flow_just_started:
       first_step = steps[0]  # First step from YAML
       session["step_idx"] = 1  # Advance for next time
       return _render_step(first_step, "en")
   ```

**Example YAML structure (`content/flows/panic.yaml`):**
```yaml
steps:
  - id: breathe_intro
    type: message
    en: "Let's slow things down together. We'll try a short 4–4–6 breath."
  - id: breathe_steps
    type: sequence
    en: ["Inhale through your nose for 4…", "Hold for 4…", ...]
```

**Returns:** First step message to user

---

### **STEP 8: User Continues Flow**

**User Input:** `"okay"`

**File:** `app/flow_engine.py`  
**Lines:** 277-362

**What happens:**

1. **Get current step:**
   ```python
   responding_to_step_idx = session["step_idx"] - 1  # Step user is responding to
   step = steps[responding_to_step_idx]
   ```

2. **Validate input:**
   ```python
   should_advance, reason, new_intent = _should_advance_step(text, session)
   ```

   **Inside `_should_advance_step()`:**
   - Checks for continuation keywords: `"done"`, `"okay"`, `"next"`, `"yes"`
   - Checks for interruption: `"stop"`, `"cancel"`, `"wait"`
   - Checks for new intent: If user says new emotional state
   - Returns: `(True, "continue", None)` if valid

3. **If valid input:**
   ```python
   # Show current step (user was responding to)
   current_step = steps[session["step_idx"]]
   session["step_idx"] += 1  # Advance for next time
   return _render_step(current_step, "en")
   ```

4. **If invalid input:**
   ```python
   # Don't advance
   # Repeat current step with helpful message
   return "I didn't quite understand that. You can type 'done' or 'okay'...\n\n[repeat current step]"
   ```

5. **Log step:**
   ```python
   log_session_event(session, "flow_step", {"step_id": step.get("id")})
   ```

---

### **STEP 9: Session Logging**

**File:** `app/storage_local.py`  
**Function:** `log_session_event()`

**What happens:**

1. **Ensure directories exist:**
   ```python
   os.makedirs("data/logs", exist_ok=True)
   ```

2. **Write to CSV:**
   ```python
   with open("data/logs/sessions.csv", "a", newline="") as f:
       writer = csv.DictWriter(f, fieldnames=...)
       writer.writerow({
           "timestamp": time.time(),
           "session_id": session.get("session_id"),
           "event_type": event_type,  # "flow_start", "flow_step", etc.
           "flow_id": session.get("flow_id"),
           "step_id": data.get("step_id"),
           "mood_pre": session.get("mood_pre"),
           "mood_post": session.get("mood_post"),
       })
   ```

**CSV Structure:**
```csv
timestamp,session_id,event_type,flow_id,step_id,mood_pre,mood_post
1234567890,abc123,flow_start,panic,breathe_intro,,
1234567891,abc123,flow_step,panic,breathe_steps,,
```

---

### **STEP 10: Mood Check-In**

**When flow reaches mood step:**

**File:** `app/flow_engine.py`  
**Lines:** 342-349

**What happens:**

1. **Detect mood prompt step:**
   ```python
   if next_step.get("type") == "prompt" and "mood" in next_step.get("id", ""):
       session["awaiting_mood_response"] = True
       return "0–10, how do you feel now?"
   ```

2. **User responds:** `"7"`

3. **Handle mood response:**
   ```python
   if session.get("awaiting_mood_response"):
       response = _handle_mood_response(text, session)
   ```

   **Inside `_handle_mood_response()` (lines 75-133):**
   - Extracts number: `7`
   - If `mood_pre` is None → Set as baseline
   - If `mood_pre` exists → Set as `mood_post`
   - Calculate change: `7 - 5 = +2`
   - Provide feedback:
     ```python
     if change > 0:
         return f"Great! You went from {pre_mood} to {mood_score}. That's improvement! 🌟"
     ```
   - Log to CSV: `log_session_event(session, "mood_update")`

4. **Reset flow:**
   ```python
   _reset_flow(session)  # Clear flow_id, step_idx
   ```

---

### **STEP 11: Return Response to User**

**File:** `app/chat.py` (line 27)

**What happens:**

```python
bot = route_message(user, session, "en")  # Gets response from flow_engine
print(Fore.MAGENTA + f"bot: {bot}")  # Display to user
```

**User sees:**
```
bot: Let's slow things down together. We'll try a short 4–4–6 breath.
```

---

## 🔄 Complete Example Flow

### **Example Conversation:**

```
1. User: "I'm feeling anxious"
   ↓
   [Crisis Check] → No crisis
   ↓
   [Intent Detection] → "check_in_mood"
   ↓
   [Map to Flow] → "panic"
   ↓
   [Load Flow] → panic.yaml
   ↓
   [Show Step 0] → "Let's slow things down together..."
   ↓
   [Log] → flow_start event

2. User: "okay"
   ↓
   [Validate Input] → Valid ("okay" in continuation keywords)
   ↓
   [Show Step 1] → "Inhale through your nose for 4…"
   ↓
   [Log] → flow_step event

3. User: "done"
   ↓
   [Validate Input] → Valid
   ↓
   [Show Step 2] → "Name 5 things you can see..."
   ↓
   [Log] → flow_step event

4. User: "done"
   ↓
   [Show Step 3] → "You're doing your best..."
   ↓
   [Log] → flow_step event

5. User: "done"
   ↓
   [Show Step 4] → "0–10, how do you feel now?"
   ↓
   [Set] → awaiting_mood_response = True

6. User: "7"
   ↓
   [Handle Mood] → mood_post = 7
   ↓
   [Calculate Change] → +2 improvement
   ↓
   [Response] → "Great! You went from 5 to 7. That's improvement! 🌟"
   ↓
   [Log] → mood_update event
   ↓
   [Reset Flow] → flow_id = None
```

---

## 📁 Files Involved in Runtime

| Step | File | Function | Purpose |
|------|------|----------|---------|
| 1 | `app/chat.py` | `run()` | Main chat loop, user interface |
| 2 | `app/safety.py` | `check_crisis()` | Crisis detection |
| 3 | `app/flow_engine.py` | `route_message()` | Main routing logic |
| 4 | `app/router.py` | `guess_intent()` | Intent detection |
| 5 | `nlu/export_en/english_nlu/inference.py` | `predict()` | TensorFlow model inference |
| 6 | `app/router.py` | `intent_to_flow()` | Map intent to flow name |
| 7 | `app/flow_engine.py` | `load_flow()` | Load YAML flow file |
| 8 | `app/flow_engine.py` | `_should_advance_step()` | Input validation |
| 9 | `app/storage_local.py` | `log_session_event()` | CSV logging |
| 10 | `app/flow_engine.py` | `_handle_mood_response()` | Mood tracking |

---

## 🔑 Key Components

### **1. Session State**
- Maintains conversation context
- Tracks current flow and step
- Stores mood scores
- Records conversation history

### **2. Intent Detection**
- **Primary:** TensorFlow model (via inference.py)
- **Fallback:** Keyword matching
- Returns intent string

### **3. Flow Management**
- YAML-based conversation flows
- Step-by-step progression
- Input validation
- Flow interruption support

### **4. Safety Layer**
- Multi-layer crisis detection
- False positive prevention
- Immediate resource provision

### **5. Logging**
- Anonymous session logging
- Event tracking
- CSV format for analysis

---

## 🎯 Quick Reference

**To run the chatbot:**
```bash
python -m app.chat
```

**Flow:**
1. User input → Crisis check
2. Intent detection → Flow selection
3. Flow step → Input validation
4. Response → Logging
5. Repeat

**Key Files:**
- `app/chat.py` - Entry point
- `app/flow_engine.py` - Core routing
- `app/router.py` - Intent detection
- `app/safety.py` - Crisis detection
- `nlu/export_en/english_nlu/inference.py` - Model inference
- `content/flows/*.yaml` - Conversation flows

---

## 📊 Data Flow Summary

```
User Input
    ↓
Crisis Detection (safety.py)
    ↓
Intent Detection (router.py → inference.py)
    ↓
Flow Selection (router.py)
    ↓
Flow Engine (flow_engine.py)
    ├── Load YAML flow
    ├── Validate input
    ├── Advance steps
    └── Handle mood
    ↓
Logging (storage_local.py)
    ↓
Response to User
```

---

## 🔍 Debugging Tips

**Check session state:**
- `session["flow_id"]` - Current flow
- `session["step_idx"]` - Current step
- `session["awaiting_mood_response"]` - Mood prompt active

**Check intent detection:**
- Model output in `router.py` line 40
- Fallback triggered if model fails

**Check flow loading:**
- YAML files in `content/flows/`
- Flow validation in `load_flow()`

**Check logging:**
- CSV file: `data/logs/sessions.csv`
- Event types: `flow_start`, `flow_step`, `mood_update`

---

**End of Runtime Flow Explanation**

