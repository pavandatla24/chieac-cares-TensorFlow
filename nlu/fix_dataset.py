# nlu/fix_dataset.py
import sys
import argparse
import unicodedata
import pandas as pd
from collections import defaultdict, Counter

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
    s = str(s)
    s = " ".join(s.strip().split())
    return unicodedata.normalize("NFKC", s)

def fix_dataset(path, out=None, max_dupes=1, max_per_intent_lang=None, shuffle=True, seed=123):
    df = pd.read_csv(path)

    # Ensure schema
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Missing required columns: {missing}")

    # Normalize columns
    for c in REQUIRED_COLS:
        df[c] = df[c].astype(str).map(normalize_text)

    # Drop empties
    before = len(df)
    df = df[df["text"] != ""].copy()
    print(f"[fix] Dropped empty texts: {before - len(df)}")

    # Filter invalid labels (keep only allowed; report drops)
    def drop_invalid(col, allowed):
        nonlocal df
        before = len(df)
        df = df[df[col].isin(allowed)].copy()
        print(f"[fix] Dropped rows with invalid {col}: {before - len(df)}")

    drop_invalid("intent", set(ALLOWED_INTENTS))
    drop_invalid("valence", ALLOWED_VALENCE)
    drop_invalid("arousal", ALLOWED_AROUSAL)
    drop_invalid("lang", ALLOWED_LANG)

    # Cap duplicates by (text, lang)
    if max_dupes is not None:
        counts = defaultdict(int)
        keep_mask = []
        for _, row in df.iterrows():
            key = (row["text"], row["lang"])
            counts[key] += 1
            keep_mask.append(counts[key] <= max_dupes)
        dup_dropped = len(df) - sum(keep_mask)
        df = df[keep_mask].copy()
        print(f"[fix] Capped exact duplicates (text+lang) to {max_dupes}: dropped {dup_dropped}")

    # Optional cap per (intent, lang)
    if max_per_intent_lang:
        kept_rows = []
        bucket_counts = defaultdict(int)
        for _, row in df.iterrows():
            key = (row["intent"], row["lang"])
            if bucket_counts[key] < max_per_intent_lang:
                kept_rows.append(True)
                bucket_counts[key] += 1
            else:
                kept_rows.append(False)
        capped = len(df) - sum(kept_rows)
        df = df[kept_rows].copy()
        print(f"[fix] Capped rows per (intent,lang) to {max_per_intent_lang}: dropped {capped}")

    # Shuffle (optional)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Write output
    if out is None:
        out = path.replace(".csv", "_fixed.csv")
    df.to_csv(out, index=False)
    print(f"[fix] Wrote: {out}")

    # Quick summary
    print("\n[summary] counts by intent x lang:")
    print(df.groupby(["intent","lang"]).size().unstack(fill_value=0).sort_index())

def parse_args(argv):
    p = argparse.ArgumentParser(description="Clean and cap training dataset.")
    p.add_argument("path", help="Input CSV (e.g., nlu/data/training_seed.csv)")
    p.add_argument("--out", help="Output CSV (default: *_fixed.csv)")
    p.add_argument("--max-dupes", type=int, default=1,
                   help="Max occurrences of the exact same (text+lang). Default: 1")
    p.add_argument("--max-per-intent-lang", type=int, default=None,
                   help="Cap rows per (intent, lang). E.g., 120")
    p.add_argument("--no-shuffle", action="store_true", help="Do not shuffle output")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    fix_dataset(
        path=args.path,
        out=args.out,
        max_dupes=args.max_dupes,
        max_per_intent_lang=args.max_per_intent_lang,
        shuffle=not args.no_shuffle,
    )
