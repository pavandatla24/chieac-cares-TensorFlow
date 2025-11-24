#!/usr/bin/env python3
"""
Week 9 - Model Retraining
Retrain the NLU model with augmented dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_hub as hub
import json
from datetime import datetime

class ModelRetrainer:
    def __init__(self, dataset_path="nlu/data/training_seed_clean_fixed_augmented.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.load_dataset()
    
    def load_dataset(self):
        """Load the augmented training dataset"""
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded dataset: {len(self.df)} rows")
        else:
            print(f"❌ Dataset not found: {self.dataset_path}")
            print("   Please run augment_dataset.py first")
            self.df = pd.DataFrame()
    
    def prepare_data(self):
        """Prepare data for training"""
        if self.df.empty:
            return None, None, None, None
        
        # Extract features and labels
        texts = self.df['text'].values
        intents = self.df['intent'].values
        valences = self.df['valence'].values
        arousals = self.df['arousal'].values
        
        # Encode labels
        self.label_encoders['intent'] = LabelEncoder()
        self.label_encoders['valence'] = LabelEncoder()
        self.label_encoders['arousal'] = LabelEncoder()
        
        intent_encoded = self.label_encoders['intent'].fit_transform(intents)
        valence_encoded = self.label_encoders['valence'].fit_transform(valences)
        arousal_encoded = self.label_encoders['arousal'].fit_transform(arousals)
        
        # Split data
        X_train, X_test, y_intent_train, y_intent_test, y_valence_train, y_valence_test, y_arousal_train, y_arousal_test = train_test_split(
            texts, intent_encoded, valence_encoded, arousal_encoded, 
            test_size=0.2, random_state=42, stratify=intent_encoded
        )
        
        print(f"📊 Training set: {len(X_train)} examples")
        print(f"📊 Test set: {len(X_test)} examples")
        
        return (X_train, X_test, y_intent_train, y_intent_test, 
                y_valence_train, y_valence_test, y_arousal_train, y_arousal_test)
    
    def build_model(self):
        """Build the multi-task classification model"""
        # Load Universal Sentence Encoder
        hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        hub_layer = hub.KerasLayer(hub_url, input_shape=[], dtype=tf.string, trainable=False)
        
        # Input layer
        text_input = layers.Input(shape=[], dtype=tf.string, name='text')
        
        # Universal Sentence Encoder
        text_embedding = hub_layer(text_input)
        
        # Shared dense layers
        shared_dense = layers.Dense(128, activation='relu')(text_embedding)
        shared_dropout = layers.Dropout(0.3)(shared_dense)
        shared_dense2 = layers.Dense(64, activation='relu')(shared_dropout)
        shared_dropout2 = layers.Dropout(0.3)(shared_dense2)
        
        # Intent classification head
        intent_dense = layers.Dense(32, activation='relu')(shared_dropout2)
        intent_dropout = layers.Dropout(0.2)(intent_dense)
        intent_output = layers.Dense(len(self.label_encoders['intent'].classes_), 
                                   activation='softmax', name='intent')(intent_dropout)
        
        # Valence classification head
        valence_dense = layers.Dense(32, activation='relu')(shared_dropout2)
        valence_dropout = layers.Dropout(0.2)(valence_dense)
        valence_output = layers.Dense(len(self.label_encoders['valence'].classes_), 
                                    activation='softmax', name='valence')(valence_dropout)
        
        # Arousal classification head
        arousal_dense = layers.Dense(32, activation='relu')(shared_dropout2)
        arousal_dropout = layers.Dropout(0.2)(arousal_dense)
        arousal_output = layers.Dense(len(self.label_encoders['arousal'].classes_), 
                                    activation='softmax', name='arousal')(arousal_dropout)
        
        # Create model
        model = Model(inputs=text_input, outputs=[intent_output, valence_output, arousal_output])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss={
                'intent': 'sparse_categorical_crossentropy',
                'valence': 'sparse_categorical_crossentropy',
                'arousal': 'sparse_categorical_crossentropy'
            },
            loss_weights={'intent': 1.0, 'valence': 0.5, 'arousal': 0.5},
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the model"""
        if self.df.empty:
            print("❌ No dataset loaded")
            return None
        
        # Prepare data
        data = self.prepare_data()
        if data is None:
            return None
        
        (X_train, X_test, y_intent_train, y_intent_test, 
         y_valence_train, y_valence_test, y_arousal_train, y_arousal_test) = data
        
        # Build model
        self.model = self.build_model()
        
        print(f"\n🏗️ Model architecture:")
        self.model.summary()
        
        # Train model
        print(f"\n🚀 Training model for {epochs} epochs...")
        
        history = self.model.fit(
            X_train,
            {
                'intent': y_intent_train,
                'valence': y_valence_train,
                'arousal': y_arousal_train
            },
            validation_data=(
                X_test,
                {
                    'intent': y_intent_test,
                    'valence': y_valence_test,
                    'arousal': y_arousal_test
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        test_loss = self.model.evaluate(
            X_test,
            {
                'intent': y_intent_test,
                'valence': y_valence_test,
                'arousal': y_arousal_test
            },
            verbose=0
        )
        
        print(f"Test Loss: {test_loss[0]:.4f}")
        print(f"Intent Accuracy: {test_loss[4]:.4f}")
        print(f"Valence Accuracy: {test_loss[5]:.4f}")
        print(f"Arousal Accuracy: {test_loss[6]:.4f}")
        
        return history
    
    def save_model(self, output_dir="nlu/export_en/english_nlu"):
        """Save the trained model and label maps"""
        if self.model is None:
            print("❌ No model to save")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, "model_weights.weights.h5")
        self.model.save_weights(model_path)
        print(f"✅ Model weights saved: {model_path}")
        
        # Save label maps
        label_map_path = os.path.join(output_dir, "label_map.json")
        label_maps = {}
        for task, encoder in self.label_encoders.items():
            label_maps[task] = {
                "classes": encoder.classes_.tolist(),
                "num_classes": len(encoder.classes_)
            }
        
        with open(label_map_path, 'w', encoding='utf-8') as f:
            json.dump(label_maps, f, indent=2)
        print(f"✅ Label maps saved: {label_map_path}")
        
        # Save model info
        model_info_path = os.path.join(output_dir, "model_info.json")
        model_info = {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": self.dataset_path,
            "dataset_size": len(self.df),
            "intent_classes": len(self.label_encoders['intent'].classes_),
            "valence_classes": len(self.label_encoders['valence'].classes_),
            "arousal_classes": len(self.label_encoders['arousal'].classes_),
            "model_architecture": "Multi-task classification with Universal Sentence Encoder"
        }
        
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        print(f"✅ Model info saved: {model_info_path}")
        
        # Generate inference script
        self._generate_inference_script(output_dir)
    
    def _generate_inference_script(self, output_dir):
        """Generate inference script for the trained model"""
        inference_script = f'''#!/usr/bin/env python3
"""
Inference script for ChiEAC CARES NLU model
"""

import sys
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder

class NLUInference:
    def __init__(self, model_dir="{output_dir}"):
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = {{}}
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and label encoders"""
        # Load label maps
        with open(f"{{self.model_dir}}/label_map.json", 'r') as f:
            label_maps = json.load(f)
        
        # Recreate label encoders
        for task, label_map in label_maps.items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(label_map['classes'])
            self.label_encoders[task] = encoder
        
        # Load model weights
        model_path = f"{{self.model_dir}}/model_weights.weights.h5"
        
        # Rebuild model architecture
        hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        hub_layer = hub.KerasLayer(hub_url, input_shape=[], dtype=tf.string, trainable=False)
        
        text_input = tf.keras.layers.Input(shape=[], dtype=tf.string, name='text')
        text_embedding = hub_layer(text_input)
        
        shared_dense = tf.keras.layers.Dense(128, activation='relu')(text_embedding)
        shared_dropout = tf.keras.layers.Dropout(0.3)(shared_dense)
        shared_dense2 = tf.keras.layers.Dense(64, activation='relu')(shared_dropout)
        shared_dropout2 = tf.keras.layers.Dropout(0.3)(shared_dense2)
        
        intent_dense = tf.keras.layers.Dense(32, activation='relu')(shared_dropout2)
        intent_dropout = tf.keras.layers.Dropout(0.2)(intent_dense)
        intent_output = tf.keras.layers.Dense(len(self.label_encoders['intent'].classes_), 
                                            activation='softmax', name='intent')(intent_dropout)
        
        valence_dense = tf.keras.layers.Dense(32, activation='relu')(shared_dropout2)
        valence_dropout = tf.keras.layers.Dropout(0.2)(valence_dense)
        valence_output = tf.keras.layers.Dense(len(self.label_encoders['valence'].classes_), 
                                             activation='softmax', name='valence')(valence_dropout)
        
        arousal_dense = tf.keras.layers.Dense(32, activation='relu')(shared_dropout2)
        arousal_dropout = tf.keras.layers.Dropout(0.2)(arousal_dense)
        arousal_output = tf.keras.layers.Dense(len(self.label_encoders['arousal'].classes_), 
                                             activation='softmax', name='arousal')(arousal_dropout)
        
        self.model = tf.keras.Model(inputs=text_input, outputs=[intent_output, valence_output, arousal_output])
        
        # Load weights
        self.model.load_weights(model_path)
    
    def predict(self, text):
        """Make predictions on input text"""
        # Prepare input
        input_text = np.array([text])
        
        # Make predictions
        predictions = self.model.predict(input_text)
        
        # Get predicted classes
        intent_pred = np.argmax(predictions[0][0])
        valence_pred = np.argmax(predictions[1][0])
        arousal_pred = np.argmax(predictions[2][0])
        
        # Get class names
        intent_class = self.label_encoders['intent'].inverse_transform([intent_pred])[0]
        valence_class = self.label_encoders['valence'].inverse_transform([valence_pred])[0]
        arousal_class = self.label_encoders['arousal'].inverse_transform([arousal_pred])[0]
        
        # Get probabilities
        intent_probs = predictions[0][0].tolist()
        valence_probs = predictions[1][0].tolist()
        arousal_probs = predictions[2][0].tolist()
        
        return {{
            "intent": intent_class,
            "intent_probs": intent_probs,
            "valence": valence_class,
            "valence_probs": valence_probs,
            "arousal": arousal_class,
            "arousal_probs": arousal_probs
        }}

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <text>")
        sys.exit(1)
    
    text = sys.argv[1]
    
    # Initialize inference
    nlu = NLUInference()
    
    # Make prediction
    result = nlu.predict(text)
    
    # Output result as JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''
        
        inference_path = os.path.join(output_dir, "inference.py")
        with open(inference_path, 'w', encoding='utf-8') as f:
            f.write(inference_script)
        print(f"✅ Inference script saved: {inference_path}")

def main():
    """Main function for model retraining"""
    print("🔄 ChiEAC CARES - Model Retraining")
    print("=" * 50)
    
    retrainer = ModelRetrainer()
    
    if retrainer.df.empty:
        print("❌ No dataset loaded. Please run augment_dataset.py first.")
        return
    
    while True:
        print("\n1. Analyze dataset")
        print("2. Train model")
        print("3. Save model")
        print("4. Full retraining pipeline")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            print(f"📊 Dataset size: {len(retrainer.df)} examples")
            print(f"📊 Intents: {retrainer.df['intent'].nunique()}")
            print(f"📊 Valence values: {retrainer.df['valence'].nunique()}")
            print(f"📊 Arousal values: {retrainer.df['arousal'].nunique()}")
        elif choice == "2":
            epochs = int(input("Enter number of epochs (default 50): ") or "50")
            batch_size = int(input("Enter batch size (default 32): ") or "32")
            retrainer.train_model(epochs, batch_size)
        elif choice == "3":
            output_dir = input("Enter output directory (or press Enter for default): ").strip()
            retrainer.save_model(output_dir if output_dir else None)
        elif choice == "4":
            epochs = int(input("Enter number of epochs (default 50): ") or "50")
            batch_size = int(input("Enter batch size (default 32): ") or "32")
            retrainer.train_model(epochs, batch_size)
            retrainer.save_model()
        elif choice == "5":
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
