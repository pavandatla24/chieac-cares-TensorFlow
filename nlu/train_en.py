import os
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nlu.model.model_en import build_english_model, export_english_model

# English-only intents
ALLOWED_INTENTS = [
    "greeting_start", "check_in_mood", "breathing_exercise", "grounding_exercise",
    "affirmation_request", "journal_prompt", "info_help", "self_assess",
    "thanks_goodbye", "crisis_emergency", "fallback_clarify",
]
ALLOWED_VALENCE = ["negative", "neutral", "positive"]
ALLOWED_AROUSAL = ["low", "medium", "high"]

def one_hot(indices, num_classes):
    arr = np.zeros((len(indices), num_classes), dtype="float32")
    arr[np.arange(len(indices)), indices] = 1.0
    return arr

def make_label_maps(df):
    intents = [i for i in ALLOWED_INTENTS if i in set(df["intent"].unique())]
    maps = {
        "intent": {name: i for i, name in enumerate(intents)},
        "valence": {name: i for i, name in enumerate(ALLOWED_VALENCE)},
        "arousal": {name: i for i, name in enumerate(ALLOWED_AROUSAL)},
    }
    return maps

def load_english_data(csv_path):
    df = pd.read_csv(csv_path)
    # Filter for English only
    df = df[df["lang"] == "en"].copy()
    df = df[["text", "intent", "valence", "arousal"]].copy()
    df["text"] = df["text"].astype(str).str.strip()

    maps = make_label_maps(df)
    y_intent_idx = df["intent"].map(maps["intent"]).values
    y_valence_idx = df["valence"].map(maps["valence"]).values
    y_arousal_idx = df["arousal"].map(maps["arousal"]).values

    # Handle NaN values (in case some intents are missing)
    y_intent_idx = np.nan_to_num(y_intent_idx, nan=0).astype(int)
    y_valence_idx = np.nan_to_num(y_valence_idx, nan=0).astype(int)
    y_arousal_idx = np.nan_to_num(y_arousal_idx, nan=0).astype(int)

    y_intent = one_hot(y_intent_idx, len(maps["intent"]))
    y_valence = one_hot(y_valence_idx, len(maps["valence"]))
    y_arousal = one_hot(y_arousal_idx, len(maps["arousal"]))

    return df["text"].values, (y_intent, y_valence, y_arousal), maps, y_intent_idx

def make_dataset(texts, labels, batch_size=32, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (texts, {"intent": labels[0], "valence": labels[1], "arousal": labels[2]})
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, len(texts)), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="nlu/data/training_seed_clean_fixed.csv", help="Path to training CSV")
    parser.add_argument("--export", default="nlu/export_en", help="Export directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Load English data only
    print(f"[INFO] Loading English data from {args.data}")
    texts, (y_int, y_val, y_ar), maps, y_int_idx = load_english_data(args.data)
    print(f"[INFO] Loaded {len(texts)} English samples")

    # Train/val split
    trX, vaX, trI, vaI = train_test_split(
        texts, y_int, test_size=0.2, random_state=42, stratify=y_int_idx
    )
    _, _, trV, vaV = train_test_split(
        texts, y_val, test_size=0.2, random_state=42, stratify=y_int_idx
    )
    _, _, trA, vaA = train_test_split(
        texts, y_ar, test_size=0.2, random_state=42, stratify=y_int_idx
    )

    train_ds = make_dataset(trX, (trI, trV, trA), batch_size=args.batch_size, shuffle=True)
    val_ds = make_dataset(vaX, (vaI, vaV, vaA), batch_size=args.batch_size, shuffle=False)

    # Build & train English model
    model = build_english_model(
        intent_classes=trI.shape[1],
        valence_classes=trV.shape[1],
        arousal_classes=trA.shape[1],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_intent_acc", patience=3, mode="max", restore_best_weights=True
        ),
    ]

    # Train the model
    if args.epochs > 0:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        print(f"[WARNING] Skipping training because epochs=0")
        history = None

    # Evaluate
    print("\n[INFO] Evaluating on validation set…")
    pred_ds = tf.data.Dataset.from_tensor_slices(vaX).batch(args.batch_size)
    preds = model.predict(pred_ds, verbose=0)

    intent_pred_idx = preds["intent"].argmax(axis=1)
    intent_true_idx = vaI.argmax(axis=1)

    inv_intent = {v: k for k, v in maps["intent"].items()}
    target_names = [inv_intent[i] for i in range(len(inv_intent))]

    print("\n[Intent classification report]")
    print(classification_report(intent_true_idx, intent_pred_idx, target_names=target_names, digits=3))
    
    # Save metrics
    metrics = {
        "val_intent_acc": float(np.mean(intent_pred_idx == intent_true_idx)),
        "val_samples": len(intent_true_idx),
    }
    if history is not None:
        metrics["history"] = history.history
    
    os.makedirs(args.export, exist_ok=True)
    metrics_path = os.path.join(args.export, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")

    # Export model (this should work!)
    if args.epochs > 0:
        export_dir = os.path.join(args.export, "english_nlu")
        print(f"\n[INFO] Exporting English model to: {export_dir}")
        export_english_model(model, export_dir, maps)
    else:
        print(f"\n[WARNING] Skipping model export because epochs=0 (no training performed)")

if __name__ == "__main__":
    main()
