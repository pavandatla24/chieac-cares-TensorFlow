# ChiEAC CARES: Complete Project Documentation
## From Scratch to Production - Step-by-Step Guide

**Version:** 1.0  
**Date:** 2024  
**Author:** Project Team  
**Purpose:** Complete technical documentation for rebuilding the ChiEAC CARES chatbot from scratch

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Setup & Initial Structure](#2-project-setup--initial-structure)
3. [Data Collection & Preparation](#3-data-collection--preparation)
4. [Data Cleaning Process](#4-data-cleaning-process)
5. [Model Development & Training](#5-model-development--training)
6. [Core Application Implementation](#6-core-application-implementation)
7. [Testing & Validation](#7-testing--validation)
8. [Deployment & Usage](#8-deployment--usage)
9. [File-by-File Reference](#9-file-by-file-reference)

---

## 1. Project Overview

### 1.1 Project Goals
- Build a locally-running, trauma-informed emotional support chatbot
- Use TensorFlow for Natural Language Understanding (NLU)
- Provide evidence-based self-regulation techniques
- Implement crisis detection for user safety
- Track mood changes pre/post interaction
- Ensure complete privacy (local operation only)

### 1.2 Technology Stack
- **Language:** Python 3.8+
- **ML Framework:** TensorFlow 2.x
- **Embeddings:** Universal Sentence Encoder (TensorFlow Hub)
- **Data Format:** CSV for training data, YAML for conversation flows
- **Architecture:** Multi-task classification (intent + emotion)

### 1.3 Key Features
- Multi-intent detection (12 intents)
- 4 conversation flows (panic, detachment, irritation, sadness)
- Crisis detection with false-positive prevention
- Mood tracking (0-10 scale)
- Input validation to prevent flow advancement on invalid input
- Anonymous session logging

---

## 2. Project Setup & Initial Structure

### 2.1 Initial Setup

**Step 1: Create Project Directory**
```bash
mkdir chieac-cares-tensorflow
cd chieac-cares-tensorflow
```

**Step 2: Initialize Git Repository**
```bash
git init
```

**Step 3: Create Folder Structure**
```
chieac-cares-tensorflow/
├── app/                    # Core application code
│   ├── __init__.py
│   ├── chat.py            # Main chat interface
│   ├── flow_engine.py      # Conversation flow management
│   ├── router.py           # Intent detection and routing
│   ├── safety.py           # Crisis detection
│   └── storage_local.py    # Local CSV logging
├── content/                # Conversation content
│   └── flows/              # YAML flow definitions
│       ├── panic.yaml
│       ├── detachment.yaml
│       ├── irritation.yaml
│       └── sadness.yaml
├── nlu/                    # Natural Language Understanding
│   ├── __init__.py
│   ├── data/               # Training data and keywords
│   │   ├── training_seed.csv
│   │   └── crisis_keywords.csv
│   ├── model/              # Model architecture
│   │   └── model_en.py
│   ├── train_en.py         # Training script
│   ├── fix_dataset.py       # Data cleaning script
│   └── export_en/          # Trained model output
├── docs/                   # Documentation
│   ├── intents_schema.md
│   ├── labeling_guide.md
│   └── safety_policy.md
├── data/                   # Runtime data
│   ├── logs/               # Session logs
│   └── feedback/           # User feedback
├── requirements.txt        # Python dependencies
├── README.md               # User-facing documentation
└── .gitignore             # Git ignore rules
```

**Step 4: Create Virtual Environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

**Step 5: Create requirements.txt**
```txt
tensorflow>=2.10.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
pyyaml>=6.0
```

**Step 6: Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2.2 Folder Structure Explanation

**`app/`** - Core application logic
- Contains all runtime Python modules
- Handles user interaction, flow management, routing, and safety

**`content/flows/`** - Conversation flow definitions
- YAML files define step-by-step conversation sequences
- Easy to edit without code changes
- Supports multiple languages (currently English-only)

**`nlu/`** - Natural Language Understanding
- `data/` - Training datasets and crisis keywords
- `model/` - Model architecture definitions
- `train_en.py` - Training script
- `export_en/` - Exported trained models

**`docs/`** - Project documentation
- Schema definitions, labeling guides, safety policies

**`data/`** - Runtime data
- `logs/` - Anonymous session logs (CSV)
- `feedback/` - User feedback and test results

---

## 3. Data Collection & Preparation

### 3.1 Intent Schema Definition

**File:** `docs/intents_schema.md`

First, define the 12 intents the chatbot will recognize:

1. **greeting_start** - User wants to start conversation
2. **check_in_mood** - User expressing emotional state
3. **breathing_exercise** - Request for breathing techniques
4. **grounding_exercise** - Request for grounding techniques
5. **affirmation_request** - Need for reassurance/positive message
6. **journal_prompt** - Request for journaling prompt
7. **info_help** - Seeking information or resources
8. **self_assess** - Self-reflection questions
9. **thanks_goodbye** - Ending conversation
10. **crisis_emergency** - Crisis situation (handled separately)
11. **fallback_clarify** - Unclear or off-topic input
12. **language_switch** - Request to change language (if multilingual)

### 3.2 Data Collection Process

**File:** `docs/labeling_guide.md`

**Step 1: Create Initial Dataset Template**

Create `nlu/data/training_seed.csv` with columns:
- `text` - User message (verbatim, keep emojis/punctuation)
- `intent` - One of the 12 intents
- `valence` - negative | neutral | positive
- `arousal` - low | medium | high
- `lang` - en | es (or just "en" for English-only)

**Step 2: Data Sources**

Collect data from:
- **Scripts & Focus Groups:** Real phrases from target users
- **Paraphrasing:** Create variations of core phrases
- **Slang & Informal Language:** Include "idk", "tbh", "umm", emojis
- **Emotional Variations:** Same intent with different emotional tones

**Step 3: Labeling Rules**

- **One message = one intent** (if unclear → `fallback_clarify`)
- **Crisis words → `crisis_emergency`** (even if context seems casual)
- **Keep slang/short forms** (ok: "idk", "umm", "tbh", emojis)
- **Code-switching:** Label by dominant language
- **No PII:** Don't include names, phones, or identifiers

**Step 4: Valence & Arousal Guidelines**

**Valence:**
- `negative`: anxious, sad, angry, overwhelmed, numb
- `neutral`: information seeking, generic requests
- `positive`: thanks, feeling better

**Arousal:**
- `low`: calm, tired, reflective
- `medium`: typical conversation energy
- `high`: panic, anger, urgent language

**Step 5: Target Counts**

- **50-100 samples per intent per language** (minimum)
- Balance intents; if one is rare, create paraphrases
- Example distribution:
  ```
  greeting_start: 80 samples
  check_in_mood: 100 samples
  breathing_exercise: 90 samples
  grounding_exercise: 85 samples
  affirmation_request: 95 samples
  journal_prompt: 75 samples
  info_help: 70 samples
  self_assess: 80 samples
  thanks_goodbye: 90 samples
  crisis_emergency: 50 samples (handled separately)
  fallback_clarify: 100 samples
  ```

**Step 6: Example Data Collection**

Example entries in `training_seed.csv`:
```csv
text,intent,valence,arousal,lang
"I'm feeling really anxious right now",check_in_mood,negative,high,en
"help me breathe",breathing_exercise,neutral,medium,en
"I feel numb and disconnected",grounding_exercise,negative,low,en
"i need some encouragement",affirmation_request,neutral,low,en
"journal prompt please",journal_prompt,neutral,low,en
"thanks, I'm good now",thanks_goodbye,positive,low,en
"i want to die",crisis_emergency,negative,high,en
"what's happening?",self_assess,neutral,medium,en
```

### 3.3 Crisis Keywords Collection

**File:** `nlu/data/crisis_keywords.csv`

Create a separate CSV with crisis-related keywords:
```csv
keyword
suicide
kill myself
hurt myself
end my life
self harm
```

---

## 4. Data Cleaning Process

### 4.1 Data Quality Issues

Common issues found in raw data:
- Duplicate rows (exact same text+lang)
- Invalid intent labels (typos, wrong values)
- Invalid valence/arousal values
- Empty text fields
- Unicode normalization issues
- Imbalanced class distribution

### 4.2 Cleaning Script

**File:** `nlu/fix_dataset.py`

**Purpose:** Clean and normalize the training dataset

**Key Functions:**

1. **Normalize Text**
   - Strip whitespace
   - Unicode normalization (NFKC)
   - Remove empty strings

2. **Validate Labels**
   - Check intent against allowed list
   - Check valence: negative/neutral/positive
   - Check arousal: low/medium/high
   - Check lang: en/es

3. **Remove Duplicates**
   - Cap exact duplicates (text+lang) to max 1 occurrence
   - Optional: Cap per (intent, lang) combination

4. **Shuffle Data**
   - Randomize order for training

**Usage:**
```bash
python nlu/fix_dataset.py nlu/data/training_seed.csv --max-dupes 1
```

**Output:** `nlu/data/training_seed_clean_fixed.csv`

### 4.3 Data Quality Check

**File:** `nlu/qc_dataset.py` (if created)

Run quality checks:
- Column validation
- Label distribution
- Class balance
- Duplicate detection

**Expected Output:**
```
[INFO] Total rows: 1183
[INFO] Intents: 12
[INFO] Languages: 1 (en)
[INFO] Class distribution:
  breathing_exercise: 95
  grounding_exercise: 88
  affirmation_request: 102
  ...
```

### 4.4 Final Dataset

**File:** `nlu/data/training_seed_clean_fixed.csv`

This is the final cleaned dataset used for training. It contains:
- ~1,200 rows (English only)
- 12 intents
- Balanced distribution
- No duplicates
- Valid labels only

---

## 5. Model Development & Training

### 5.1 Model Architecture

**File:** `nlu/model/model_en.py`

**Architecture:** Multi-task classification model

**Components:**

1. **Text Encoder**
   - Uses Universal Sentence Encoder (USE) from TensorFlow Hub
   - Converts text to 512-dimensional embeddings
   - Handles variable-length input

2. **Multi-Task Heads**
   - **Intent Classification:** 12 classes (softmax)
   - **Valence Classification:** 3 classes (negative/neutral/positive)
   - **Arousal Classification:** 3 classes (low/medium/high)

3. **Model Structure:**
   ```
   Input Text
     ↓
   Universal Sentence Encoder (USE)
     ↓
   Dense Layer (128 units, ReLU)
     ↓
   ┌─────────────┬─────────────┬─────────────┐
   │ Intent Head │ Valence Head│ Arousal Head│
   │ (12 classes)│ (3 classes) │ (3 classes)│
   └─────────────┴─────────────┴─────────────┘
   ```

**Key Code:**
```python
def build_english_model(intent_classes, valence_classes, arousal_classes):
    # Load USE encoder
    encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Input
    text_input = tf.keras.layers.Input(shape=[], dtype=tf.string)
    embedding = encoder(text_input)
    
    # Shared dense layer
    dense = tf.keras.layers.Dense(128, activation='relu')(embedding)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    
    # Multi-task heads
    intent_output = tf.keras.layers.Dense(intent_classes, activation='softmax', name='intent')(dropout)
    valence_output = tf.keras.layers.Dense(valence_classes, activation='softmax', name='valence')(dropout)
    arousal_output = tf.keras.layers.Dense(arousal_classes, activation='softmax', name='arousal')(dropout)
    
    model = tf.keras.Model(inputs=text_input, outputs={
        'intent': intent_output,
        'valence': valence_output,
        'arousal': arousal_output
    })
    
    return model
```

### 5.2 Training Script

**File:** `nlu/train_en.py`

**Purpose:** Train the English NLU model

**Process:**

1. **Load Data**
   - Read `training_seed_clean_fixed.csv`
   - Filter for English only (`lang == "en"`)
   - Extract text and labels

2. **Create Label Maps**
   - Map intent names to indices: `{"breathing_exercise": 0, ...}`
   - Map valence: `{"negative": 0, "neutral": 1, "positive": 2}`
   - Map arousal: `{"low": 0, "medium": 1, "high": 2}`

3. **One-Hot Encoding**
   - Convert labels to one-hot vectors for multi-class classification

4. **Train/Validation Split**
   - 80% training, 20% validation
   - Stratified split (maintains class distribution)

5. **Build Model**
   - Call `build_english_model()` with class counts

6. **Compile Model**
   ```python
   model.compile(
       optimizer='adam',
       loss={
           'intent': 'categorical_crossentropy',
           'valence': 'categorical_crossentropy',
           'arousal': 'categorical_crossentropy'
       },
       metrics=['accuracy']
   )
   ```

7. **Train**
   - Batch size: 32
   - Epochs: 10-20
   - Early stopping if validation loss doesn't improve
   - Model checkpointing

8. **Evaluate**
   - Print classification reports for each task
   - Calculate accuracy, precision, recall, F1-score

9. **Export Model**
   - Save model weights
   - Save label maps (for inference)
   - Create inference script
   - Save metrics

**Usage:**
```bash
python nlu/train_en.py --data nlu/data/training_seed_clean_fixed.csv --epochs 10
```

**Output:**
- `nlu/export_en/english_nlu/model_weights.weights.h5` - Model weights
- `nlu/export_en/english_nlu/label_map.json` - Label mappings
- `nlu/export_en/english_nlu/inference.py` - Inference script
- `nlu/export_en/metrics.json` - Training metrics

### 5.3 Model Export

**File:** `nlu/model/model_en.py` (export function)

**Purpose:** Package trained model for deployment

**Exports:**
1. **Model Weights** - Trained weights in H5 format
2. **Label Maps** - JSON file mapping indices to labels
3. **Inference Script** - Standalone script for predictions
4. **Model Info** - Metadata (architecture, training date, etc.)

**Inference Script Structure:**
```python
import tensorflow as tf
import tensorflow_hub as hub
import json
import sys

# Load model components
encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# ... load weights and label maps ...

def predict(text):
    # Run inference
    predictions = model.predict([text])
    intent_idx = np.argmax(predictions['intent'][0])
    intent = label_map['intent'][str(intent_idx)]
    return intent
```

---

## 6. Core Application Implementation

### 6.1 Router (Intent Detection)

**File:** `app/router.py`

**Purpose:** Detect user intent from text input

**Components:**

1. **NLUModel Class**
   - Loads trained model
   - Runs inference via subprocess (calls `inference.py`)
   - Falls back to keyword matching if model fails

2. **Intent Detection Flow:**
   ```
   User Input
     ↓
   Try TensorFlow Model (via inference.py)
     ↓
   [If fails] → Keyword Fallback
     ↓
   Map Intent to Flow
   ```

3. **Keyword Fallback**
   - Pattern matching for common phrases
   - Handles edge cases model might miss
   - Examples:
     - "breath", "breathe", "panic", "anxious" → `breathing_exercise`
     - "ground", "dissociate", "numb" → `grounding_exercise`
     - "angry", "furious", "rage" → `anger_management`
     - "sad", "depressed" → `affirmation_request`

4. **Intent to Flow Mapping**
   ```python
   INTENT_TO_FLOW = {
       "breathing_exercise": "panic",
       "check_in_mood": "panic",
       "grounding_exercise": "detachment",
       "anger_management": "irritation",
       "affirmation_request": "sadness",
       "journal_prompt": "sadness",
   }
   ```

**Key Functions:**
- `get_nlu_model()` - Singleton model instance
- `guess_intent(text)` - Main intent detection
- `intent_to_flow(intent)` - Map intent to conversation flow

### 6.2 Flow Engine

**File:** `app/flow_engine.py`

**Purpose:** Manage conversation flows and session state

**Components:**

1. **Flow Loading**
   - Loads YAML files from `content/flows/`
   - Validates flow structure (id, title, steps)
   - Caches loaded flows

2. **Session Management**
   ```python
   session = {
       "lang": "en",
       "flow_id": "panic",  # Current flow
       "step_idx": 2,       # Current step index
       "mood_pre": None,     # Pre-interaction mood
       "mood_post": None,    # Post-interaction mood
       "awaiting_mood_response": False,
       "conversation_history": [],
   }
   ```

3. **Step Progression**
   - Validates user input before advancing
   - Only advances on valid keywords ("done", "okay", "next")
   - Repeats current step on invalid input
   - Handles different step types (message, prompt, sequence)

4. **Input Validation**
   - Continuation keywords: "done", "okay", "next", "yes", "continue"
   - Interruption keywords: "stop", "cancel", "wait"
   - New intent detection: If user expresses new emotional state, interrupt current flow
   - Invalid input: Stay on current step, show helpful message

5. **Mood Tracking**
   - Pre-mood: First mood score (if provided)
   - Post-mood: After flow completion
   - Mood change calculation
   - Smart recommendations based on mood scores

**Key Functions:**
- `load_flow(name)` - Load YAML flow file
- `start_session()` - Initialize new session
- `route_message(text, session)` - Main routing logic
- `_should_advance_step(text, session)` - Input validation
- `_handle_mood_response(text, session)` - Mood input handling

### 6.3 Safety Layer

**File:** `app/safety.py`

**Purpose:** Detect crisis situations and provide immediate resources

**Components:**

1. **Multi-Layer Detection**
   - **High-Risk Phrases:** "kill myself", "end my life", "suicide"
   - **Medium-Risk Phrases:** "self harm", "hurt myself", "can't go on"
   - **Contextual Patterns:** Regex patterns for crisis language
   - **Keyword Matching:** From `crisis_keywords.csv`

2. **False Positive Prevention**
   - Patterns to exclude: "die laughing", "kill time", "suicide mission"
   - Context-aware detection

3. **Crisis Response**
   - Returns emergency resources message
   - Bypasses normal conversation flow
   - Provides immediate help information

**Key Function:**
- `check_crisis(text, lang="en")` - Returns crisis message if detected, None otherwise

### 6.4 Chat Interface

**File:** `app/chat.py`

**Purpose:** Terminal-based user interface

**Components:**

1. **Welcome Message**
   - Safety disclaimer
   - Instructions
   - Available commands

2. **Main Loop**
   ```python
   while True:
       user_input = input("you: ")
       
       # Handle commands
       if user_input.lower() == "exit":
           break
       if user_input.lower() == "help":
           show_help()
           continue
       if user_input.lower() == "reset":
           session = start_session()
           continue
       
       # Crisis check
       crisis_msg = check_crisis(user_input)
       if crisis_msg:
           print(f"bot: {crisis_msg}")
           continue
       
       # Route message
       response = route_message(user_input, session, "en")
       print(f"bot: {response}")
   ```

3. **Command Handling**
   - `help` - Show available options
   - `reset` - Restart current flow
   - `exit` - End conversation

### 6.5 Local Storage

**File:** `app/storage_local.py`

**Purpose:** Anonymous session logging

**Components:**

1. **Session Logging**
   - Logs to `data/logs/sessions.csv`
   - Fields: timestamp, session_id, event_type, flow_id, step_id, mood_pre, mood_post
   - No personal data collected

2. **Event Types**
   - `session_start` - New session
   - `flow_start` - Flow initiated
   - `flow_step` - Step progression
   - `flow_complete` - Flow finished
   - `mood_baseline` - Pre-mood recorded
   - `mood_update` - Post-mood recorded

**Key Functions:**
- `log_session_event(session, event_type, data=None)` - Log event to CSV

---

## 7. Testing & Validation

### 7.1 Crisis Detection Testing

**File:** `test_crisis_detection.py`

**Purpose:** Verify crisis detection accuracy

**Tests:**
- High-risk phrases (should trigger)
- Medium-risk phrases (should trigger)
- False positives (should NOT trigger)
- Edge cases

**Usage:**
```bash
python test_crisis_detection.py
```

**Expected:** 100% accuracy (all crisis cases detected, no false positives)

### 7.2 Model Performance Testing

**File:** `test_model_performance.py`

**Purpose:** Evaluate intent detection accuracy

**Tests:**
- Intent classification accuracy
- Per-intent performance
- Confusion matrix
- Edge cases

**Usage:**
```bash
python test_model_performance.py
```

**Expected:** >95% accuracy on test set

### 7.3 End-to-End Validation

**File:** `final_validation.py` or `comprehensive_test.py`

**Purpose:** Test complete user journeys

**Tests:**
- All conversation flows
- Mood tracking
- Invalid input handling
- Flow interruption
- Crisis detection integration

**Usage:**
```bash
python comprehensive_test.py
```

**Expected:** All tests pass

### 7.4 Test Results Summary

- **Crisis Detection:** 48/48 tests passed (100%)
- **Intent Classification:** 45/46 tests passed (97.8%)
- **End-to-End Validation:** All user journeys successful
- **Invalid Input Handling:** Correctly prevents flow advancement
- **Mood Tracking:** Accurate mood capture and recommendations

---

## 8. Deployment & Usage

### 8.1 Running the Chatbot

**Command:**
```bash
python -m app.chat
```

**Expected Output:**
```
ChiEAC CARES (English). Type 'help' for tips, 'reset' to restart, 'exit' to quit.
bot: I'm here with you. Before we start: this is not medical care. 
If you're in danger or might harm yourself, reply 'HELP'. 
How can I help you today? You can try breathing exercises, 
grounding techniques, affirmations, or journal prompts. 
Or just tell me how you're feeling.
you: 
```

### 8.2 Example Conversation

```
you: I'm feeling anxious
bot: Let's slow things down together. We'll try a short 4–4–6 breath.

you: okay
bot: Inhale through your nose for 4…
Hold for 4…
Exhale slowly for 6…
Repeat 4 times. I'll count with you.

you: done
bot: Name 5 things you can see, 4 you can feel, 3 you can hear, 
2 you can smell, 1 you can taste.

you: done
bot: You're doing your best right now. Your feelings are real. 
They will pass.

you: done
bot: 0–10, how do you feel now?

you: 7
bot: Thanks, noted your mood at 7/10. How can I help you today?
```

### 8.3 Available Commands

- **Emotional States:** "I'm anxious", "I'm angry", "I feel sad", "I feel numb"
- **Specific Requests:** "breathing exercises", "grounding techniques", "affirmations", "journal prompts"
- **Control Commands:**
  - `help` - Show available options
  - `reset` - Restart current flow
  - `exit` - End conversation

---

## 9. File-by-File Reference

### 9.1 Application Files (`app/`)

#### `app/__init__.py`
- Empty file to make `app` a Python package

#### `app/chat.py`
- **Purpose:** Main chat interface
- **Key Functions:**
  - `run()` - Main chat loop
  - `show_help()` - Display help message
- **Dependencies:** `flow_engine`, `safety`

#### `app/router.py`
- **Purpose:** Intent detection and routing
- **Key Classes:**
  - `NLUModel` - Wraps TensorFlow model
- **Key Functions:**
  - `get_nlu_model()` - Get singleton model instance
  - `guess_intent(text)` - Detect intent from text
  - `intent_to_flow(intent)` - Map intent to flow
- **Dependencies:** TensorFlow, subprocess

#### `app/flow_engine.py`
- **Purpose:** Conversation flow management
- **Key Functions:**
  - `load_flow(name)` - Load YAML flow file
  - `start_session()` - Initialize session
  - `route_message(text, session)` - Main routing logic
  - `_should_advance_step(text, session)` - Input validation
  - `_handle_mood_response(text, session)` - Mood handling
- **Dependencies:** `yaml`, `router`, `storage_local`

#### `app/safety.py`
- **Purpose:** Crisis detection
- **Key Function:**
  - `check_crisis(text, lang)` - Detect crisis situations
- **Dependencies:** `csv`, `re`

#### `app/storage_local.py`
- **Purpose:** Local CSV logging
- **Key Function:**
  - `log_session_event(session, event_type, data)` - Log to CSV
- **Dependencies:** `csv`, `os`, `uuid`

### 9.2 Content Files (`content/`)

#### `content/flows/panic.yaml`
- **Purpose:** Panic/anxiety conversation flow
- **Structure:**
  ```yaml
  id: panic
  title: Panic / Overwhelm
  steps:
    - id: breathe_intro
      type: message
      en: "Let's slow things down together..."
    - id: breathe_steps
      type: sequence
      en: ["Inhale...", "Hold...", "Exhale..."]
    - id: grounding
      type: message
      en: "Name 5 things you can see..."
    - id: affirmation
      type: message
      en: "You're doing your best..."
    - id: mood_post
      type: prompt
      en: "0–10, how do you feel now?"
  ```

#### `content/flows/detachment.yaml`
- **Purpose:** Detachment/numbness flow
- Similar structure to panic.yaml

#### `content/flows/irritation.yaml`
- **Purpose:** Anger/irritation flow
- Similar structure to panic.yaml

#### `content/flows/sadness.yaml`
- **Purpose:** Sadness/depression flow
- Similar structure to panic.yaml

### 9.3 NLU Files (`nlu/`)

#### `nlu/train_en.py`
- **Purpose:** Train English NLU model
- **Process:**
  1. Load CSV data
  2. Create label maps
  3. Split train/val
  4. Build model
  5. Train
  6. Evaluate
  7. Export
- **Usage:** `python nlu/train_en.py --data <csv> --epochs 10`

#### `nlu/model/model_en.py`
- **Purpose:** Model architecture and export
- **Key Functions:**
  - `build_english_model()` - Build model architecture
  - `export_english_model()` - Export trained model

#### `nlu/fix_dataset.py`
- **Purpose:** Clean training dataset
- **Process:**
  1. Normalize text
  2. Validate labels
  3. Remove duplicates
  4. Shuffle
- **Usage:** `python nlu/fix_dataset.py <input.csv>`

#### `nlu/data/training_seed_clean_fixed.csv`
- **Purpose:** Final cleaned training dataset
- **Format:** CSV with columns: text, intent, valence, arousal, lang
- **Size:** ~1,200 rows (English only)

#### `nlu/data/crisis_keywords.csv`
- **Purpose:** Crisis detection keywords
- **Format:** CSV with column: keyword

#### `nlu/export_en/english_nlu/`
- **Purpose:** Exported trained model
- **Contents:**
  - `model_weights.weights.h5` - Model weights
  - `label_map.json` - Label mappings
  - `inference.py` - Inference script
  - `model_info.json` - Model metadata

### 9.4 Documentation Files (`docs/`)

#### `docs/intents_schema.md`
- **Purpose:** Intent definitions and schema
- **Contents:** All 12 intents with descriptions

#### `docs/labeling_guide.md`
- **Purpose:** Data labeling guidelines
- **Contents:** Rules for labeling training data

#### `docs/safety_policy.md`
- **Purpose:** Safety and crisis detection policy
- **Contents:** Crisis detection rules and resources

### 9.5 Test Files (Root)

#### `test_crisis_detection.py`
- **Purpose:** Test crisis detection accuracy
- **Usage:** `python test_crisis_detection.py`

#### `test_model_performance.py`
- **Purpose:** Test model accuracy
- **Usage:** `python test_model_performance.py`

#### `comprehensive_test.py`
- **Purpose:** End-to-end testing
- **Usage:** `python comprehensive_test.py`

#### `final_validation.py`
- **Purpose:** Final system validation
- **Usage:** `python final_validation.py`

### 9.6 Configuration Files (Root)

#### `requirements.txt`
- **Purpose:** Python dependencies
- **Contents:**
  ```
  tensorflow>=2.10.0
  pandas>=1.5.0
  numpy>=1.23.0
  scikit-learn>=1.1.0
  pyyaml>=6.0
  ```

#### `README.md`
- **Purpose:** User-facing documentation
- **Contents:** Quick start, usage, features

#### `.gitignore`
- **Purpose:** Git ignore rules
- **Contents:**
  ```
  venv/
  __pycache__/
  *.pyc
  nlu/export_en/
  data/logs/
  ```

---

## 10. Step-by-Step Rebuild Instructions

### Step 1: Project Setup
1. Create project directory
2. Initialize Git
3. Create folder structure
4. Create virtual environment
5. Install dependencies

### Step 2: Data Collection
1. Define intent schema (`docs/intents_schema.md`)
2. Create labeling guide (`docs/labeling_guide.md`)
3. Collect training data (50-100 samples per intent)
4. Create `nlu/data/training_seed.csv`
5. Collect crisis keywords (`nlu/data/crisis_keywords.csv`)

### Step 3: Data Cleaning
1. Run `nlu/fix_dataset.py` to clean data
2. Verify output: `nlu/data/training_seed_clean_fixed.csv`
3. Check data quality (distribution, duplicates)

### Step 4: Model Development
1. Create model architecture (`nlu/model/model_en.py`)
2. Implement training script (`nlu/train_en.py`)
3. Train model: `python nlu/train_en.py`
4. Verify model export in `nlu/export_en/`

### Step 5: Core Application
1. Implement router (`app/router.py`)
2. Implement flow engine (`app/flow_engine.py`)
3. Implement safety layer (`app/safety.py`)
4. Implement chat interface (`app/chat.py`)
5. Implement storage (`app/storage_local.py`)

### Step 6: Conversation Flows
1. Create YAML flows in `content/flows/`
2. Define panic flow
3. Define detachment flow
4. Define irritation flow
5. Define sadness flow

### Step 7: Testing
1. Test crisis detection
2. Test model performance
3. Test end-to-end flows
4. Fix any issues

### Step 8: Documentation
1. Create README.md
2. Document usage
3. Document customization

### Step 9: Deployment
1. Test locally
2. Prepare for GitHub release
3. Create release notes

---

## 11. Key Design Decisions

### 11.1 Why TensorFlow?
- Industry-standard ML framework
- Good support for text embeddings
- Easy model export and deployment
- Universal Sentence Encoder integration

### 11.2 Why YAML for Flows?
- Easy to edit without code changes
- Non-technical users can modify content
- Version control friendly
- Supports multiple languages

### 11.3 Why Local Operation?
- Privacy: No data leaves user's device
- Security: No external API calls
- Reliability: Works offline
- Compliance: No data storage concerns

### 11.4 Why Multi-Task Learning?
- Intent + emotion detection in one model
- More efficient than separate models
- Shared representations improve accuracy
- Single inference call

### 11.5 Why Input Validation?
- Prevents flow advancement on gibberish
- Better user experience
- Clearer conversation flow
- Reduces confusion

---

## 12. Troubleshooting

### Issue: Model not loading
**Solution:** Check that `nlu/export_en/english_nlu/` exists and contains all required files

### Issue: Intent detection inaccurate
**Solution:** 
1. Check training data quality
2. Retrain with more data
3. Adjust keyword fallback patterns

### Issue: Flow not advancing
**Solution:** Check input validation - user must type "done", "okay", or "next"

### Issue: Crisis detection false positives
**Solution:** Update false positive patterns in `app/safety.py`

### Issue: Import errors
**Solution:** Ensure virtual environment is activated and dependencies installed

---

## 13. Future Enhancements

### Potential Improvements:
1. **Multilingual Support:** Add Spanish flows and training data
2. **Voice Interface:** Add speech-to-text and text-to-speech
3. **Mobile App:** Create mobile interface
4. **Analytics Dashboard:** Visualize mood trends
5. **Custom Flows:** Allow users to create custom flows
6. **Integration:** Connect with external mental health resources

---

## 14. Conclusion

This documentation provides a complete guide to rebuilding the ChiEAC CARES chatbot from scratch. Follow the steps in order, and you'll have a fully functional, locally-running emotional support chatbot.

**Key Takeaways:**
- Data quality is critical for model performance
- Input validation improves user experience
- Crisis detection must be accurate and sensitive
- Local operation ensures privacy
- Testing is essential before deployment

**Questions or Issues?**
Refer to the specific file documentation or test scripts for detailed implementation details.

---

**End of Documentation**

