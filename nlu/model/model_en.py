import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# English-only Universal Sentence Encoder
HUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

class EnglishNLU(tf.keras.Model):
    """
    English-only NLU model using Universal Sentence Encoder.
    Much simpler than LaBSE - no complex tokenization needed.
    """
    def __init__(
        self,
        intent_classes: int,
        valence_classes: int,
        arousal_classes: int,
        hub_url: str = HUB_URL,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        # Simple USE encoder (no tokenization needed!)
        self.encoder = hub.KerasLayer(hub_url, trainable=False, name="use_encoder")
        
        # Shared & task-specific layers
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.shared = tf.keras.layers.Dense(256, activation="relu", name="shared_dense")
        
        self.intent_head = tf.keras.layers.Dense(intent_classes, activation="softmax", name="intent")
        self.valence_head = tf.keras.layers.Dense(valence_classes, activation="softmax", name="valence")
        self.arousal_head = tf.keras.layers.Dense(arousal_classes, activation="softmax", name="arousal")

    def call(self, inputs, training: bool = False):
        # inputs: tf.Tensor(shape=[None], dtype=tf.string)
        # USE encoder handles strings directly - no tokenization needed!
        embeddings = self.encoder(inputs)  # [batch, 512]
        
        x = self.dropout(embeddings, training=training)
        x = self.shared(x)
        
        return {
            "intent": self.intent_head(x),
            "valence": self.valence_head(x),
            "arousal": self.arousal_head(x),
        }

def build_english_model(intent_classes, valence_classes, arousal_classes, hub_url=HUB_URL):
    model = EnglishNLU(intent_classes, valence_classes, arousal_classes, hub_url)
    losses = {
        "intent": tf.keras.losses.CategoricalCrossentropy(),
        "valence": tf.keras.losses.CategoricalCrossentropy(),
        "arousal": tf.keras.losses.CategoricalCrossentropy(),
    }
    metrics = {
        "intent": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
        "valence": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
        "arousal": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
    }
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=losses, metrics=metrics)
    return model

def export_english_model(model, export_dir, label_map):
    """
    Export English-only model using weights + inference script approach.
    This avoids TensorFlow Hub export issues.
    """
    os.makedirs(export_dir, exist_ok=True)

    # 1) Save label map
    with open(os.path.join(export_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # 2) Save model weights
    weights_path = os.path.join(export_dir, "model_weights.weights.h5")
    model.save_weights(weights_path)
    print(f"[INFO] Model weights saved to {weights_path}")

    # 3) Save model architecture info
    model_info = {
        "intent_classes": model.intent_head.units,
        "valence_classes": model.valence_head.units,
        "arousal_classes": model.arousal_head.units,
        "shared_units": model.shared.units,
        "dropout_rate": model.dropout.rate,
        "hub_url": "https://tfhub.dev/google/universal-sentence-encoder/4"
    }
    
    with open(os.path.join(export_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Model info saved to {os.path.join(export_dir, 'model_info.json')}")

    # 4) Create inference script
    inference_script = '''import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class EnglishNLUInference:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.load_model()
    
    def load_model(self):
        # Load model info
        with open(os.path.join(self.model_dir, "model_info.json"), "r") as f:
            self.model_info = json.load(f)
        
        # Load label maps
        with open(os.path.join(self.model_dir, "label_map.json"), "r") as f:
            self.label_maps = json.load(f)
        
        # Load encoder
        self.encoder = hub.KerasLayer(self.model_info["hub_url"], trainable=False)
        
        # Rebuild model architecture
        self.rebuild_model()
        
        # Load weights
        weights_path = os.path.join(self.model_dir, "model_weights.weights.h5")
        self.model.load_weights(weights_path)
    
    def rebuild_model(self):
        # Rebuild the model architecture
        inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
        
        # USE encoder (handles strings directly)
        embeddings = self.encoder(inputs)  # [batch, 512]
        
        # Dense layers
        x = tf.keras.layers.Dropout(self.model_info["dropout_rate"])(embeddings)
        x = tf.keras.layers.Dense(self.model_info["shared_units"], activation="relu")(x)
        
        # Output heads
        intent_out = tf.keras.layers.Dense(self.model_info["intent_classes"], activation="softmax", name="intent")(x)
        valence_out = tf.keras.layers.Dense(self.model_info["valence_classes"], activation="softmax", name="valence")(x)
        arousal_out = tf.keras.layers.Dense(self.model_info["arousal_classes"], activation="softmax", name="arousal")(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs={"intent": intent_out, "valence": valence_out, "arousal": arousal_out})
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = self.model.predict(texts, verbose=0)
        
        # Convert to labels
        results = []
        for i in range(len(texts)):
            intent_idx = np.argmax(predictions["intent"][i])
            valence_idx = np.argmax(predictions["valence"][i])
            arousal_idx = np.argmax(predictions["arousal"][i])
            
            intent_labels = {v: k for k, v in self.label_maps["intent"].items()}
            valence_labels = {v: k for k, v in self.label_maps["valence"].items()}
            arousal_labels = {v: k for k, v in self.label_maps["arousal"].items()}
            
            results.append({
                "text": texts[i],
                "intent": intent_labels[intent_idx],
                "valence": valence_labels[valence_idx],
                "arousal": arousal_labels[arousal_idx],
                "intent_probs": predictions["intent"][i].tolist(),
                "valence_probs": predictions["valence"][i].tolist(),
                "arousal_probs": predictions["arousal"][i].tolist()
            })
        
        return results[0] if len(results) == 1 else results

# Example usage:
if __name__ == "__main__":
    import sys
    import json
    
    # Load the model
    nlu = EnglishNLUInference(".")
    
    # Get text from command line argument
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "I need help with breathing exercises"
    
    # Test prediction
    result = nlu.predict(text)
    print(json.dumps(result))
'''
    
    with open(os.path.join(export_dir, "inference.py"), "w", encoding="utf-8") as f:
        f.write(inference_script)
    print(f"[INFO] Inference script saved to {os.path.join(export_dir, 'inference.py')}")
    
    print(f"[INFO] English model export completed successfully!")
    print(f"[INFO] To use the model, run: python {os.path.join(export_dir, 'inference.py')}")
