# ChiEAC CARES: Trauma-Informed Emotional Support Chatbot

A locally-running, TensorFlow-powered chatbot designed to provide immediate emotional support through evidence-based techniques. Built with trauma-informed principles and crisis detection capabilities.

## 🌟 Features

- **Multi-Intent Detection**: Recognizes anxiety, anger, sadness, and detachment states
- **Evidence-Based Flows**: Breathing exercises, grounding techniques, affirmations, and journal prompts
- **Crisis Detection**: Advanced pattern recognition for immediate safety intervention
- **Mood Tracking**: Pre/post interaction mood monitoring with smart recommendations
- **Local Operation**: Complete privacy - no data leaves your device
- **Session Logging**: Anonymous CSV logging for research and improvement

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/chieac-cares-tensorflow.git
   cd chieac-cares-tensorflow
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the chatbot**
   ```bash
   python -m app.chat
   ```

## 💬 Usage

### Starting a Conversation
The chatbot will greet you with:
```
I'm here with you. Before we start: this is not medical care. 
If you're in danger or might harm yourself, reply 'HELP'. 
How can I help you today?
```

### Available Commands
- **Emotional States**: "I'm anxious", "I'm angry", "I feel sad", "I feel numb"
- **Specific Requests**: "breathing exercises", "grounding techniques", "affirmations", "journal prompts"
- **Control Commands**: 
  - `help` - Show available options
  - `reset` - Restart current flow
  - `exit` - End conversation

### Example Conversation
```
you: I'm feeling anxious
bot: Let's slow things down together. We'll try a short 4–4–6 breath.

you: okay
bot: Inhale through your nose for 4…
Hold for 4…
Exhale slowly for 6…
Repeat 4 times. I'll count with you.

you: next
bot: Name 5 things you can see, 4 you can feel, 3 you can hear, 2 you can smell, 1 you can taste.

you: next
bot: You're doing your best right now. Your feelings are real. They will pass.

you: thanks
bot: 0–10, how do you feel now?

you: 7
bot: Great! You went from 5 to 7. That's improvement! 🌟 Do you need help with anything else?
```

## 🧠 How It Works

### Architecture Overview
```
User Input → Crisis Detection → Intent Classification → Flow Selection → Response Generation
```

### Core Components

1. **NLU Model** (`app/router.py`)
   - TensorFlow-based intent classification
   - Fallback keyword matching
   - Confidence thresholding

2. **Flow Engine** (`app/flow_engine.py`)
   - YAML-based conversation flows
   - Session state management
   - Mood tracking integration

3. **Safety Layer** (`app/safety.py`)
   - Multi-layer crisis detection
   - Pattern-based risk assessment
   - Emergency resource provision

4. **Chat Interface** (`app/chat.py`)
   - Terminal-based interaction
   - Color-coded responses
   - Command handling

### Conversation Flows

#### Panic/Anxiety Flow
- 4-4-6 breathing technique
- 5-4-3-2-1 grounding exercise
- Affirmative messaging
- Mood check-in

#### Anger Management Flow
- Box breathing (4-4-4-4)
- Trigger identification
- Reframing techniques
- Mood assessment

#### Sadness Flow
- 4-6 breathing pattern
- Self-compassion exercises
- Gentle action planning
- Mood tracking

#### Detachment Flow
- Present-moment awareness
- Physical grounding techniques
- Safety affirmation
- Mood evaluation

## 🔧 Customization

### Adding New Flows
1. Create a new YAML file in `content/flows/`
2. Define steps with appropriate IDs and messages
3. Update intent mapping in `app/router.py`

### Modifying Crisis Detection
1. Edit `nlu/data/crisis_keywords.csv` for keyword-based detection
2. Update patterns in `app/safety.py` for phrase-based detection
3. Test with `python test_crisis_detection.py`

### Training the Model
1. Prepare training data in `nlu/data/`
2. Run `python nlu/train_model.py`
3. Test performance with `python test_model_performance.py`

## 📊 Testing

### Run All Tests
```bash
# Crisis detection testing
python test_crisis_detection.py

# Model performance testing  
python test_model_performance.py

# End-to-end validation
python final_validation.py
```

### Test Results
- **Crisis Detection**: 48/48 tests passed (100%)
- **Intent Classification**: 45/46 tests passed (97.8%)
- **End-to-End Validation**: All user journeys successful

## 📁 Project Structure

```
chieac-cares-tensorflow/
├── app/                    # Core application code
│   ├── chat.py            # Main chat interface
│   ├── flow_engine.py     # Conversation flow management
│   ├── router.py          # Intent detection and routing
│   ├── safety.py          # Crisis detection
│   └── storage_local.py   # Local logging
├── content/               # Conversation content
│   └── flows/             # YAML flow definitions
├── nlu/                   # Natural Language Understanding
│   ├── data/              # Training data and keywords
│   └── export_en/         # Trained model
├── docs/                  # Documentation
├── tests/                 # Test scripts
└── requirements.txt       # Dependencies
```

## 🛡️ Safety & Ethics

### Crisis Detection
- Multi-layer pattern recognition
- False positive prevention
- Immediate resource provision
- Emergency contact information

### Privacy Protection
- Local operation only
- Anonymous session logging
- No personal data collection
- User-controlled data retention

### Trauma-Informed Design
- Non-judgmental language
- User autonomy respect
- Evidence-based techniques
- Cultural sensitivity

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- Mental health professionals who provided guidance
- Open source community for inspiration and tools
- Beta testers who provided valuable feedback

## 📞 Support

- **Crisis Resources**: 988 (U.S. Suicide & Crisis Lifeline)
- **Project Issues**: GitHub Issues
- **Documentation**: [Wiki](https://github.com/yourusername/chieac-cares-tensorflow/wiki)

## 🔄 Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Enhanced crisis detection and mood tracking
- **v1.2.0** - Improved intent classification and user experience

---

**⚠️ Important Disclaimer**: This chatbot is not a substitute for professional mental health care. If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.