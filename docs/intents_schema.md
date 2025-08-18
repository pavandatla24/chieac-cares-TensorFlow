
# ChiEAC CARES – Intent & Emotion Schema

This document is the single source of truth for:

- What each intent means
- Example user phrases (EN/ES)
- How the bot should route (flow mapping)
- Emotion labels used by the model

## Emotion Labels (Multi-Task Outputs)

- **Valence:** `negative`, `neutral`, `positive`
- **Arousal (stress level):** `low`, `medium`, `high`

> The NLU predicts: `intent`, `valence`, and `arousal`. Flows can use any or all.

---

## Intents

### 1) `greeting_start`

**Purpose:** User opens/joins the conversation.**Route:** Welcome message → consent → quick options.**EN examples:**

- "hi" / "hello"
- "hey can we talk?"
- "i need to chat"
- "is anyone there?"
  **ES examples:**
- "hola"
- "¿podemos hablar?"
- "necesito chatear"
- "¿hay alguien?"

---

### 2) `check_in_mood`

**Purpose:** User shares how they feel (names emotion/symptom).**Route:** Ask for 0–10 mood; suggest flow (panic/detachment/irritation/sadness) based on valence/arousal.**EN examples:**

- "i feel anxious/panicky"
- "i'm numb / not feeling anything"
- "i'm angry/irritated"
- "i feel sad/down today"
  **ES examples:**
- "me siento ansioso/ansiosa"
- "estoy como desconectado/a"
- "estoy enojado/a" / "me irrita todo"
- "me siento triste"

---

### 3) `breathing_exercise`

**Purpose:** Requests a breathwork guide.**Route:** Offer 4‑4‑6, box breathing, or 4‑6 based on arousal.**EN examples:**

- "can you help me breathe?"
- "breathing exercise please"
- "i need a calm breathing routine"
- "guide my breathing"
  **ES examples:**
- "ayúdame a respirar"
- "un ejercicio de respiración"
- "necesito calmarme con respiración"
- "guíame la respiración"

---

### 4) `grounding_exercise`

**Purpose:** Requests grounding/anchoring to the present.**Route:** Offer 5‑4‑3‑2‑1 senses or orientation prompts.**EN examples:**

- "help me ground"
- "grounding exercise please"
- "i’m dissociating / not here"
- "need to feel present"
  **ES examples:**
- "necesito arraigarme"
- "un ejercicio de arraigo"
- "me estoy desconectando"
- "quiero sentirme presente"

---

### 5) `affirmation_request`

**Purpose:** Wants positive statements/mantras.**Route:** Offer 1–3 short bilingual affirmations; invite user to pick one.**EN examples:**

- "i need something positive"
- "give me an affirmation"
- "say something encouraging"
- "need reassurance"
  **ES examples:**
- "necesito algo positivo"
- "dame una afirmación"
- "algo que me anime"
- "quiero una frase de apoyo"

---

### 6) `journal_prompt`

**Purpose:** Requests a short reflective prompt.**Route:** Offer 1–2 prompts; encourage 2–3 sentence response.**EN examples:**

- "journal prompt please"
- "help me reflect"
- "give me a writing prompt"
- "i want to write about this"
  **ES examples:**
- "una consigna para escribir"
- "ayúdame a reflexionar"
- "un prompt de diario"
- "quiero escribir sobre esto"

---

### 7) `info_help`

**Purpose:** Asks for general support info/resources (non‑crisis).**Route:** Show resource list (school counselor, local orgs, non‑urgent lines); never clinical claims.**EN examples:**

- "what resources can help?"
- "who can i talk to nearby?"
- "non‑emergency help options"
- "information about support"
  **ES examples:**
- "¿qué recursos hay?"
- "¿con quién puedo hablar?"
- "opciones de ayuda no urgente"
- "información de apoyo"

---

### 8) `self_assess`

**Purpose:** Psychoeducation/understanding (“why do i feel like this?”).**Route:** Short neuroscience‑informed explanation; suggest matching flow.**EN examples:**

- "why am i feeling this way?"
- "what’s happening in my brain?"
- "is this normal?"
- "how do stress and emotions work?"
  **ES examples:**
- "¿por qué me siento así?"
- "¿qué pasa en mi cerebro?"
- "¿esto es normal?"
- "¿cómo funcionan el estrés y las emociones?"

---

### 9) `language_switch`

**Purpose:** Switch between English and Spanish.**Route:** Toggle `lang` state; confirm in new language.**EN examples:**

- "speak spanish please"
- "can we switch to english?"
- "spanish mode"
- "english please"
  **ES examples:**
- "hablemos en español"
- "cambiemos a inglés"
- "modo español"
- "inglés por favor"

---

### 10) `thanks_goodbye`

**Purpose:** Ends/pauses the session.**Route:** Close with care statement + reminder of resources; optional post‑mood check.**EN examples:**

- "thanks, bye"
- "this helped, goodnight"
- "i'll come back later"
- "that’s all for now"
  **ES examples:**
- "gracias, adiós"
- "me ayudó, buenas noches"
- "vuelvo luego"
- "eso es todo por ahora"

---

### 11) `crisis_emergency`

**Purpose:** Mentions self‑harm, suicide, harm to others, imminent danger.**Route:** Immediate crisis message with resources; **no** other flow.**EN examples:**

- "i want to die" / "kill myself"
- "i will hurt myself"
- "i’m in danger right now"
- "i might overdose"
  **ES examples:**
- "me quiero morir" / "matarme"
- "voy a lastimarme"
- "estoy en peligro ahora"
- "podría tener una sobredosis"

---

### 12) `fallback_clarify`

**Purpose:** Low confidence / out‑of‑domain / unclear input.**Route:** Brief clarify question + show 3–4 quick options (breathing/grounding/affirmation/journal).**EN examples:**

- "???" / emojis only
- "idk"
- off‑topic: "what time is the game?"
- "hmm…"
  **ES examples:**
- "no sé"
- solo emojis
- fuera de tema: "¿a qué hora es el partido?"
- "mmm…"

---

## Flow Mapping (initial)

- `check_in_mood` → choose among `panic`, `detachment`, `irritation`, `sadness` using valence+arousal.
- `breathing_exercise` → `panic` or breathing sub‑flow.
- `grounding_exercise` → `detachment` or grounding sub‑flow.
- `affirmation_request` → affirmation sub‑flow (any state).
- `journal_prompt` → journaling sub‑flow (any state).
- `info_help` → resources card (non‑crisis).
- `self_assess` → psychoeducation snippet + suggest next flow.
- `language_switch` → toggle lang and confirm.
- `thanks_goodbye` → closeout + optional mood_post.
- `crisis_emergency` → crisis script only.
- `fallback_clarify` → clarifying question + menu.

## Notes

- Never make clinical claims; always include opt‑out and “talk to a person” option.
- Crisis detection is conservative; when in doubt, route to `crisis_emergency`.
- Keep responses short, warm, and culturally affirming; avoid jargon.
