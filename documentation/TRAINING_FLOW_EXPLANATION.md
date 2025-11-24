# Complete Training Flow: From Data to Saved Model

## 📊 Visual Flow Diagram

```
training_seed_clean_fixed.csv
         ↓
    [train_en.py]
         ↓
  load_english_data()
         ↓
    Filter English
    Create Label Maps
    One-Hot Encode
         ↓
    Train/Val Split (80/20)
         ↓
    [model_en.py]
         ↓
  build_english_model()
         ↓
    Create Model Architecture
    Compile with Losses/Metrics
         ↓
    model.fit() - TRAINING
         ↓
    Evaluate on Validation Set
         ↓
    [model_en.py]
         ↓
  export_english_model()
         ↓
    Save to: nlu/export_en/english_nlu/
         ├── model_weights.weights.h5
         ├── label_map.json
         ├── model_info.json
         └── inference.py
```

---

## 🔍 Step-by-Step Detailed Explanation

### **STEP 1: Data File Location**

**File:** `nlu/data/training_seed_clean_fixed.csv`

**What it contains:**
```csv
text,intent,valence,arousal,lang
"I'm feeling anxious",check_in_mood,negative,high,en
"help me breathe",breathing_exercise,neutral,medium,en
"I feel numb",grounding_exercise,negative,low,en
...
```

**Where it's used:** Line 67 in `nlu/train_en.py`
```python
parser.add_argument("--data", default="nlu/data/training_seed_clean_fixed.csv", ...)
```

---

### **STEP 2: Loading Data in train_en.py**

**File:** `nlu/train_en.py`  
**Function:** `load_english_data(csv_path)` (lines 34-55)

**What happens:**

1. **Read CSV file:**
   ```python
   df = pd.read_csv(csv_path)  # Reads training_seed_clean_fixed.csv
   ```

2. **Filter for English only:**
   ```python
   df = df[df["lang"] == "en"].copy()  # Only English data
   ```

3. **Extract columns:**
   ```python
   df = df[["text", "intent", "valence", "arousal"]].copy()
   ```

4. **Create label maps:**
   ```python
   maps = make_label_maps(df)
   # Creates: {"intent": {"breathing_exercise": 0, ...}, 
   #          "valence": {"negative": 0, ...},
   #          "arousal": {"low": 0, ...}}
   ```

5. **Convert to indices:**
   ```python
   y_intent_idx = df["intent"].map(maps["intent"]).values
   # "breathing_exercise" → 0
   # "grounding_exercise" → 1
   # etc.
   ```

6. **One-hot encode:**
   ```python
   y_intent = one_hot(y_intent_idx, len(maps["intent"]))
   # [0, 1, 0, 0, ...] for "grounding_exercise"
   ```

7. **Return:**
   - `texts` - Array of text strings
   - `(y_intent, y_valence, y_ar)` - One-hot encoded labels
   - `maps` - Label mappings
   - `y_int_idx` - Intent indices (for stratification)

**Called from:** Line 75 in `main()`
```python
texts, (y_int, y_val, y_ar), maps, y_int_idx = load_english_data(args.data)
```

---

### **STEP 3: Train/Validation Split**

**File:** `nlu/train_en.py`  
**Lines:** 78-87

**What happens:**

```python
# Split into 80% training, 20% validation
trX, vaX, trI, vaI = train_test_split(
    texts, y_int, test_size=0.2, random_state=42, stratify=y_int_idx
)
# Same for valence and arousal
```

**Result:**
- `trX` - Training texts (80%)
- `vaX` - Validation texts (20%)
- `trI`, `trV`, `trA` - Training labels
- `vaI`, `vaV`, `vaA` - Validation labels

---

### **STEP 4: Create TensorFlow Dataset**

**File:** `nlu/train_en.py`  
**Function:** `make_dataset()` (lines 57-63)

**What happens:**

```python
train_ds = make_dataset(trX, (trI, trV, trA), batch_size=32, shuffle=True)
val_ds = make_dataset(vaX, (vaI, vaV, vaA), batch_size=32, shuffle=False)
```

**Creates:**
- Batched TensorFlow datasets
- Shuffled training data
- Prefetched for performance

---

### **STEP 5: Build Model Using model_en.py**

**File:** `nlu/train_en.py` (line 93)  
**Calls:** `nlu/model/model_en.py` → `build_english_model()`

**What happens:**

```python
# In train_en.py line 93:
model = build_english_model(
    intent_classes=trI.shape[1],      # e.g., 12
    valence_classes=trV.shape[1],     # 3
    arousal_classes=trA.shape[1],      # 3
)
```

**Inside `model_en.py` (lines 50-63):**

1. **Create model instance:**
   ```python
   model = EnglishNLU(intent_classes, valence_classes, arousal_classes, hub_url)
   ```

2. **Define losses:**
   ```python
   losses = {
       "intent": tf.keras.losses.CategoricalCrossentropy(),
       "valence": tf.keras.losses.CategoricalCrossentropy(),
       "arousal": tf.keras.losses.CategoricalCrossentropy(),
   }
   ```

3. **Define metrics:**
   ```python
   metrics = {
       "intent": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
       "valence": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
       "arousal": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
   }
   ```

4. **Compile model:**
   ```python
   model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), 
                 loss=losses, 
                 metrics=metrics)
   ```

5. **Return compiled model**

**Model Architecture (from `EnglishNLU` class):**
```
Input Text (string)
    ↓
Universal Sentence Encoder (USE)
    ↓
Dropout (0.2)
    ↓
Dense Layer (256 units, ReLU)
    ↓
┌─────────────┬─────────────┬─────────────┐
│ Intent Head │ Valence Head│ Arousal Head│
│ (12 classes)│ (3 classes) │ (3 classes) │
└─────────────┴─────────────┴─────────────┘
```

---

### **STEP 6: Training the Model**

**File:** `nlu/train_en.py`  
**Lines:** 107-114

**What happens:**

```python
history = model.fit(
    train_ds,                    # Training data
    validation_data=val_ds,       # Validation data
    epochs=args.epochs,          # e.g., 10 epochs
    callbacks=callbacks,         # Early stopping
    verbose=1,                   # Show progress
)
```

**During training:**
- Model processes batches of text
- USE encoder converts text to embeddings
- Model predicts intent, valence, arousal
- Loss is calculated and backpropagated
- Weights are updated
- Validation accuracy is monitored
- Early stopping if no improvement

**Output:**
- Trained model with updated weights
- Training history (loss, accuracy per epoch)

---

### **STEP 7: Evaluation**

**File:** `nlu/train_en.py`  
**Lines:** 119-131

**What happens:**

```python
# Predict on validation set
preds = model.predict(pred_ds, verbose=0)

# Get predicted and true labels
intent_pred_idx = preds["intent"].argmax(axis=1)
intent_true_idx = vaI.argmax(axis=1)

# Print classification report
print(classification_report(intent_true_idx, intent_pred_idx, ...))
```

**Output:**
- Accuracy, precision, recall, F1-score per intent
- Overall model performance metrics

---

### **STEP 8: Export Model Using model_en.py**

**File:** `nlu/train_en.py` (line 151)  
**Calls:** `nlu/model/model_en.py` → `export_english_model()`

**What happens:**

```python
# In train_en.py line 151:
export_dir = os.path.join(args.export, "english_nlu")  # nlu/export_en/english_nlu
export_english_model(model, export_dir, maps)
```

**Inside `export_english_model()` (lines 65-198 in model_en.py):**

#### **8.1 Save Label Map**
```python
with open(os.path.join(export_dir, "label_map.json"), "w") as f:
    json.dump(label_map, f, ...)
```
**Saves:** `nlu/export_en/english_nlu/label_map.json`
```json
{
  "intent": {
    "breathing_exercise": 0,
    "grounding_exercise": 1,
    ...
  },
  "valence": {"negative": 0, "neutral": 1, "positive": 2},
  "arousal": {"low": 0, "medium": 1, "high": 2}
}
```

#### **8.2 Save Model Weights**
```python
weights_path = os.path.join(export_dir, "model_weights.weights.h5")
model.save_weights(weights_path)
```
**Saves:** `nlu/export_en/english_nlu/model_weights.weights.h5`
- Contains all trained weights (USE embeddings, dense layers, output heads)

#### **8.3 Save Model Info**
```python
model_info = {
    "intent_classes": model.intent_head.units,      # 12
    "valence_classes": model.valence_head.units,   # 3
    "arousal_classes": model.arousal_head.units,   # 3
    "shared_units": model.shared.units,             # 256
    "dropout_rate": model.dropout.rate,             # 0.2
    "hub_url": "https://tfhub.dev/google/universal-sentence-encoder/4"
}
```
**Saves:** `nlu/export_en/english_nlu/model_info.json`

#### **8.4 Create Inference Script**
```python
inference_script = '''...'''  # Full Python script
with open(os.path.join(export_dir, "inference.py"), "w") as f:
    f.write(inference_script)
```
**Saves:** `nlu/export_en/english_nlu/inference.py`

**What inference.py does:**
- Loads model info and label maps
- Rebuilds model architecture
- Loads trained weights
- Provides `predict()` function for inference

---

### **STEP 9: Final Saved Model Structure**

**Location:** `nlu/export_en/english_nlu/`

**Files created:**

1. **`model_weights.weights.h5`**
   - Trained model weights
   - ~5-10 MB
   - Contains all learned parameters

2. **`label_map.json`**
   - Maps indices to label names
   - Used to convert predictions to readable labels

3. **`model_info.json`**
   - Model architecture metadata
   - Used to rebuild model during inference

4. **`inference.py`**
   - Standalone inference script
   - Can be run independently
   - Used by `app/router.py` at runtime

---

## 🔄 How It's Used at Runtime

### **When Chatbot Runs:**

1. **User types message:** "I'm feeling anxious"

2. **`app/router.py` loads model:**
   ```python
   model = NLUModel(model_dir="nlu/export_en/english_nlu")
   ```

3. **Calls inference script:**
   ```python
   result = subprocess.run([
       sys.executable, 
       "nlu/export_en/english_nlu/inference.py", 
       "I'm feeling anxious"
   ])
   ```

4. **Inference script:**
   - Loads `model_info.json` → Rebuilds architecture
   - Loads `model_weights.weights.h5` → Loads weights
   - Loads `label_map.json` → Maps predictions to labels
   - Runs prediction
   - Returns: `{"intent": "check_in_mood", ...}`

5. **Router gets intent:**
   ```python
   intent = "check_in_mood"
   flow = intent_to_flow(intent)  # Returns "panic"
   ```

6. **Flow engine starts panic flow**

---

## 📝 Summary

| Step | File | Function | What It Does |
|------|------|----------|--------------|
| 1 | `train_en.py` | `load_english_data()` | Reads CSV, filters English, creates label maps |
| 2 | `train_en.py` | `train_test_split()` | Splits data 80/20 |
| 3 | `train_en.py` | `make_dataset()` | Creates TensorFlow datasets |
| 4 | `train_en.py` | `build_english_model()` | **Calls model_en.py** |
| 5 | `model_en.py` | `build_english_model()` | Creates model architecture |
| 6 | `train_en.py` | `model.fit()` | **Trains the model** |
| 7 | `train_en.py` | `model.predict()` | Evaluates on validation set |
| 8 | `train_en.py` | `export_english_model()` | **Calls model_en.py** |
| 9 | `model_en.py` | `export_english_model()` | Saves weights, maps, info, inference script |

**Key Points:**
- `training_seed_clean_fixed.csv` → **Input data**
- `train_en.py` → **Training script** (orchestrates everything)
- `model_en.py` → **Model definition** (architecture + export)
- `nlu/export_en/english_nlu/` → **Saved trained model** (used at runtime)

---

## 🎯 Quick Reference

**To train the model:**
```bash
python nlu/train_en.py --data nlu/data/training_seed_clean_fixed.csv --epochs 10
```

**What gets created:**
- `nlu/export_en/english_nlu/model_weights.weights.h5`
- `nlu/export_en/english_nlu/label_map.json`
- `nlu/export_en/english_nlu/model_info.json`
- `nlu/export_en/english_nlu/inference.py`

**What gets used at runtime:**
- All files in `nlu/export_en/english_nlu/`
- Called by `app/router.py` via subprocess

