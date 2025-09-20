#!/usr/bin/env python3
"""
Test individual flows to verify they work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.flow_engine import start_session, route_message, load_flow

def test_flow(flow_name, test_inputs):
    """Test a specific flow with given inputs"""
    print(f"\n🧪 Testing flow: {flow_name}")
    print("-" * 40)
    
    # Load and validate flow
    flow = load_flow(flow_name)
    if not flow:
        print(f"❌ Failed to load flow: {flow_name}")
        return False
    
    print(f"✅ Flow loaded: {flow['title']}")
    print(f"   Steps: {len(flow['steps'])}")
    print(f"   Language: English only")
    
    # Test flow execution
    session = start_session()
    session["flow_id"] = flow_name
    session["step_idx"] = 0
    
    for i, (text, lang) in enumerate(test_inputs):
        print(f"\n   Step {i+1}: '{text}'")
        response = route_message(text, session, "en")
        print(f"   Bot: {response}")
        print(f"   Session: flow_id={session.get('flow_id')}, step={session.get('step_idx')}")
        
        if session.get("step_idx") >= len(flow["steps"]):
            print("   ✅ Flow completed!")
            break
    
    return True

def main():
    print("🧪 Individual Flow Testing")
    print("=" * 50)
    
    # Test panic flow
    test_flow("panic", [
        ("I'm having a panic attack", "en"),
        ("okay", "en"),
        ("I'm ready", "en"),
        ("done", "en"),
        ("5", "en"),  # Mood response
    ])
    
    # Test detachment flow
    test_flow("detachment", [
        ("I feel disconnected", "en"),
        ("okay", "en"),
        ("I'm in my room", "en"),
        ("done", "en"),
        ("3", "en"),  # Mood response
    ])
    
    # Test sadness flow
    test_flow("sadness", [
        ("I'm feeling really sad", "en"),
        ("okay", "en"),
        ("I'll try", "en"),
        ("water", "en"),
        ("4", "en"),  # Mood response
    ])
    
    # Test another English flow
    test_flow("irritation", [
        ("I'm really angry right now", "en"),
        ("okay", "en"),
        ("work", "en"),
        ("I'll try", "en"),
        ("6", "en"),  # Mood response
    ])

if __name__ == "__main__":
    main()
