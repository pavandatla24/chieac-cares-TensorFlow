import os
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
    # Load the model
    nlu = EnglishNLUInference(".")
    
    # Test prediction
    result = nlu.predict("I need help with breathing exercises")
    print("Prediction:", result)
