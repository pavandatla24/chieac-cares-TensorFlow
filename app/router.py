import os
import sys
import subprocess
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple model interface using the inference script
class NLUModel:
    def __init__(self, model_dir="nlu/export_en/english_nlu"):
        self.model_dir = model_dir
        self.label_maps = None
        self._load_label_maps()
    
    def _load_label_maps(self):
        # Load label maps
        with open(os.path.join(self.model_dir, "label_map.json"), "r") as f:
            self.label_maps = json.load(f)
    
    def predict(self, text):
        """Use the inference script to get predictions"""
        try:
            # Use the inference script
            inference_script = os.path.join(self.model_dir, "inference.py")
            result = subprocess.run([
                sys.executable, inference_script, text
            ], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if result.returncode == 0:
                # Parse the output (assuming it prints the result)
                output = result.stdout.strip()
                # Extract intent from the output
                if "intent" in output:
                    # Simple parsing - in real implementation, you'd use proper JSON parsing
                    lines = output.split('\n')
                    for line in lines:
                        if 'intent' in line:
                            # Extract intent from line like "intent: breathing_exercise"
                            intent = line.split(':')[-1].strip().strip("'\"")
                            return intent, 0.8  # Default confidence
                
            # Fallback to keyword matching
            return self._fallback_intent(text), 0.5
            
        except Exception as e:
            print(f"NLU Error: {e}")
            return self._fallback_intent(text), 0.3
    
    def _fallback_intent(self, text):
        """Fallback keyword-based intent detection"""
        t = text.lower()
        
        if any(word in t for word in ["breath", "breathe", "calm", "panic", "anxious"]):
            return "breathing_exercise"
        elif any(word in t for word in ["ground", "present", "dissociate", "numb", "disconnect"]):
            return "grounding_exercise"
        elif any(word in t for word in ["affirm", "positive", "reassurance", "support"]):
            return "affirmation_request"
        elif any(word in t for word in ["journal", "write", "prompt"]):
            return "journal_prompt"
        elif any(word in t for word in ["hi", "hello", "help", "talk"]):
            return "greeting_start"
        elif any(word in t for word in ["sad", "angry", "anxious", "overwhelmed", "feel"]):
            return "check_in_mood"
        else:
            return "fallback_clarify"

# Initialize the model (singleton)
_nlu_model = None

def get_nlu_model():
    global _nlu_model
    if _nlu_model is None:
        _nlu_model = NLUModel()
    return _nlu_model

INTENT_TO_FLOW = {
    # Panic/Overwhelm flows
    "breathing_exercise": "panic",
    "check_in_mood": "panic", 
    "greeting_start": "panic",
    "self_assess": "panic",
    "info_help": "panic",
    "crisis_emergency": "panic",
    
    # Detachment/Numbness flows
    "grounding_exercise": "detachment",
    
    # Irritation/Anger flows
    "anger_management": "irritation",
    
    # Sadness flows
    "affirmation_request": "sadness",
    "journal_prompt": "sadness",
    
    # Fallback
    "fallback_clarify": None,
    "thanks_goodbye": None,
}

def guess_intent(text: str) -> str:
    """Use keyword-based intent detection (simplified for now)"""
    try:
        model = get_nlu_model()
        intent, confidence = model.predict(text)
        
        # Only return the intent if confidence is reasonable
        if confidence > 0.3:  # Threshold for confidence
            return intent
        else:
            return "fallback_clarify"
    except Exception as e:
        print(f"NLU Error: {e}")
        # Fallback to simple keyword matching
        return _fallback_intent_detection(text)

def _fallback_intent_detection(text: str) -> str:
    """Fallback regex-based intent detection"""
    t = text.lower()
    
    # Simple keyword patterns as fallback
    if any(word in t for word in ["breath", "breathe", "calm", "panic", "anxious"]):
        return "breathing_exercise"
    elif any(word in t for word in ["ground", "present", "dissociate", "numb"]):
        return "grounding_exercise"
    elif any(word in t for word in ["affirm", "positive", "reassurance"]):
        return "affirmation_request"
    elif any(word in t for word in ["journal", "write", "prompt"]):
        return "journal_prompt"
    elif any(word in t for word in ["hi", "hello", "help", "talk"]):
        return "greeting_start"
    elif any(word in t for word in ["sad", "angry", "anxious", "overwhelmed"]):
        return "check_in_mood"
    else:
        return "fallback_clarify"

def intent_to_flow(intent: str):
    return INTENT_TO_FLOW.get(intent)
