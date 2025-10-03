# ChiEAC CARES – Local Emotional Support Chatbot (TensorFlow)

**Local, privacy-first (English-only)** chatbot that routes users to community co-designed self-regulation flows (breathing, grounding, affirmations, journaling). TensorFlow model detects **intent** and **emotional state**; rule-based safety layer flags crisis terms. Runs fully **offline/local**.

## Features

- TensorFlow multi-task NLU (intent + valence + arousal)
- Trauma-informed, community-authored flows (EN)
- Rule-based crisis detection with immediate resources
- Local CSV logging (anonymous) for mood pre/post and flow completion

## Quickstart

```bash
git clone https://github.com/<you>/chieac-cares.git
cd chieac-cares
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt
# (optional) train: python -m nlu.train_en --data nlu/data/training_seed_english_only.csv --export nlu/export_en --epochs 3
python -m app.chat                # starts local terminal chatbot
```

## Structure

* `content/flows/*` – YAML flows (EN)
* `nlu/data/` – training CSV + crisis keywords
* `app/` – chat loop, flow engine, safety checks
* `docs/` – consent, safety policy, model card

## Chat usage

- Type how you feel, or one of: `breathing`, `grounding`, `affirmation`, `journal`
- Commands: `help` (tips), `reset` (restart current flow), `exit` (quit)
- Mood check-ins: on prompt, reply `0-10`. First number becomes baseline; later numbers show change

## Safety & Consent

See `docs/consent.md` and `docs/safety_policy.md`. This tool is not medical care. If in danger, call/text **988** (U.S.) or your local emergency number.


## Notes

- Trained model files under `nlu/export_en/` are not committed (see .gitignore). Use the training command above or provide your own.
- Logs are stored locally at `data/logs/sessions.csv` (anonymous event log).