# ChiEAC CARES – Labeling Guide (Week 2)

## Columns

- text: user message (verbatim; keep emojis/punctuation)
- intent: one of the 12 intents from docs/intents_schema.md
- valence: negative | neutral | positive
- arousal: low | medium | high
- lang: en | es

## Rules

- One message = one intent. If unclear → fallback_clarify.
- Crisis words → crisis_emergency (even if context seems casual).
- Keep slang/short forms (ok: “idk”, “umm”, “tbh”, emojis).
- Code-switching: label by dominant language; if mix, choose the language of the main words.
- Don’t include names, phones, or identifiers.

## Valence & Arousal cues

- negative: anxious, sad, angry, overwhelmed, numb
- neutral: information seeking, generic requests
- positive: thanks, feeling better
- low arousal: calm, tired, reflective
- medium: typical conversation energy
- high: panic, anger, urgent language

## Target counts (starter)

- 50–100 samples per intent × per language (en & es).
- Balance intents; if one is rare, create paraphrases and short variants.

## Examples per intent (short)

- greeting_start: “hey”, “hola”, “can we talk?”, “¿estás ahí?”
- check_in_mood: “i feel anxious”, “me siento triste”, “estoy enojado”
- breathing_exercise: “help me breathe”, “respiración guiada”
- grounding_exercise: “help me ground”, “necesito arraigo”
- affirmation_request: “need reassurance”, “una afirmación”
- journal_prompt: “journal prompt?”, “una consigna para escribir”
- info_help: “who can i talk to?”, “¿qué recursos hay?”
- self_assess: “why am i like this?”, “¿esto es normal?”
- language_switch: “spanish please”, “cambiemos a inglés”
- thanks_goodbye: “thanks, bye”, “gracias, adiós”
- crisis_emergency: “i want to die”, “me quiero morir”
- fallback_clarify: “??”, “no sé”, off-topic
