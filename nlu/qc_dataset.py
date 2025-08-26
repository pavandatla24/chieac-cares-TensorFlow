# nlu/qc_dataset.py
import sys
import unicodedata
from collections import Counter
import pandas as pd

# --- Config: allowed schema ---
ALLOWED_INTENTS = [
    "greeting_start","check_in_mood","breathing_exercise","grounding_exercise",
    "affirmation_request","journal_prompt","info_help","self_assess",
    "language_switch","thanks_goodbye","crisis_emergency","fallback_clarify",
]
ALLOWED_VALENCE = {"negative","neutral","positive"}
ALLOWED_AROUSAL = {"low","medium","high"}
ALLOWED_LANG = {"en","es"}

REQUIRED_COLS = ["text","intent","valence","arousal","lang"]

def normalize_text(s: str) -> str:
    # Trim, standardize spaces, and normalize unicode (keeps emojis)
    s = str(s).strip()
    s = " ".join(s.split())
    s = unicodedata.normalize("NFKC", s)
    return s

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "nlu/data/training_seed.csv"
    print(f"[QC] Loading: {path}")
    df = pd.read_csv(path)

    # --- Column checks ---
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Missing required columns: {missing}")

    # --- Clean up & normalize ---
    for c in REQUIRED_COLS:
        df[c] = df[c].astype(str).map(normalize_text)

    # --- Basic errors ---
    empties = df["text"].eq("")
    print(f"[QC] Empty texts: {int(empties.sum())}")

    # Duplicates by (text, lang)
    dup_mask = df.duplicated(subset=["text","lang"], keep=False)
    dup_count = int(dup_mask.sum())
    print(f"[QC] Duplicate rows (same text+lang): {dup_count}")

    # --- Label validation ---
    bad_intent = ~df["intent"].isin(ALLOWED_INTENTS)
    bad_valence = ~df["valence"].isin(ALLOWED_VALENCE)
    bad_arousal = ~df["arousal"].isin(ALLOWED_AROUSAL)
    bad_lang = ~df["lang"].isin(ALLOWED_LANG)

    print(f"[QC] Invalid intents: {int(bad_intent.sum())}")
    print(f"[QC] Invalid valence: {int(bad_valence.sum())}")
    print(f"[QC] Invalid arousal: {int(bad_arousal.sum())}")
    print(f"[QC] Invalid lang: {int(bad_lang.sum())}")

    if bad_intent.any():
        print("[HINT] Unexpected intents:\n", df.loc[bad_intent, "intent"].value_counts().head(10))
    if bad_lang.any():
        print("[HINT] Unexpected lang codes:\n", df.loc[bad_lang, "lang"].value_counts().head(10))

    # --- Class balance summaries ---
    print("\n[QC] Counts by intent:")
    print(df["intent"].value_counts().sort_index())

    print("\n[QC] Counts by language:")
    print(df["lang"].value_counts())

    print("\n[QC] Counts by valence:")
    print(df["valence"].value_counts())

    print("\n[QC] Counts by arousal:")
    print(df["arousal"].value_counts())

    print("\n[QC] Counts by intent x lang:")
    intent_lang = df.groupby(["intent","lang"]).size().unstack(fill_value=0).sort_index()
    print(intent_lang)

    # --- Simple heuristics: overly long/short texts ---
    df["len"] = df["text"].str.len()
    too_short = df["len"] < 2
    too_long = df["len"] > 200  # adjust if you expect long messages
    print(f"\n[QC] Too short (<2 chars): {int(too_short.sum())}")
    print(f"[QC] Too long  (>200 chars): {int(too_long.sum())}")

    # --- Save cleaned copy ---
    out_path = path.replace(".csv", "_clean.csv")
    df.drop(columns=["len"]).to_csv(out_path, index=False)
    print(f"\n[QC] Wrote cleaned CSV: {out_path}")

    # --- Optional: light duplicate report sample ---
    if dup_count:
        sample_dups = df.loc[dup_mask, ["text","lang","intent"]].head(10)
        print("\n[QC] Sample duplicates (first 10):")
        print(sample_dups.to_string(index=False))

if __name__ == "__main__":
    main()
