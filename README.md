
# ChiEAC CARES – Local Bilingual Emotional Support Chatbot (TensorFlow)

**Local, privacy-first, bilingual (EN/ES)** chatbot that routes users to community co-designed self-regulation flows (breathing, grounding, affirmations, journaling). TensorFlow model detects **intent** and **emotional state**; rule-based safety layer flags crisis terms. Runs fully **offline/local**; zero recurring cost.

## Features

- TensorFlow multi-task NLU (intent + valence + arousal)
- Trauma-informed, community-authored flows (EN/ES)
- Rule-based crisis detection with immediate resources
- Local CSV logging (anonymous) for mood pre/post and flow completion

## Quickstart

```bash
git clone https://github.com/<you>/chieac-cares.git
cd chieac-cares
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python nlu/train.py              # trains a tiny model from seed data
python app/chat.py               # starts local terminal chatbot
```


## Structure

* `content/flows/*` – YAML flows (EN/ES)
* `nlu/data/` – training CSV + crisis keywords
* `app/` – chat loop, flow engine, safety checks
* `docs/` – consent, safety policy, model card

## Safety & Consent

See `docs/consent.md` and `docs/safety_policy.md`. This tool is not medical care. If in danger, call/text **988** (U.S.) or your local emergency number.
