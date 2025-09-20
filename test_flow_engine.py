#!/usr/bin/env python3
"""
Test script for Week 6 - Flow Engine
Tests the conversation flow routing and session management
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.flow_engine import start_session, route_message, get_available_flows, validate_flow_exists
from app.router import guess_intent, intent_to_flow

def test_flow_engine():
    print("🧪 Testing ChiEAC CARES Flow Engine (Week 6)")
    print("=" * 50)
    
    # Test 1: Available flows
    print("\n1. Testing available flows...")
    flows = get_available_flows()
    print(f"   Found flows: {flows}")
    
    for flow in flows:
        is_valid = validate_flow_exists(flow)
        print(f"   - {flow}: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test 2: Intent detection and flow mapping
    print("\n2. Testing intent detection and flow mapping...")
    test_inputs = [
        "I need breathing exercises",
        "I feel disconnected and numb", 
        "I'm really angry right now",
        "I need some positive words",
        "I want to write about my feelings",
        "Hello, I need help"
    ]
    
    for text in test_inputs:
        intent = guess_intent(text)
        flow = intent_to_flow(intent)
        print(f"   '{text}' -> intent: {intent} -> flow: {flow}")
    
    # Test 3: Session management and flow execution
    print("\n3. Testing session management and flow execution...")
    session = start_session()
    print(f"   Initial session: {session}")
    
    # Test conversation flow
    test_conversation = [
        ("I'm having a panic attack", "en"),
        ("I need help", "en"),
        ("5", "en"),  # Mood response
        ("I feel sad", "en"),
        ("7", "en"),  # Mood response
    ]
    
    for i, (text, lang) in enumerate(test_conversation):
        print(f"\n   Turn {i+1}: '{text}'")
        response = route_message(text, session, "en")
        print(f"   Bot: {response}")
        print(f"   Session state: flow_id={session.get('flow_id')}, step={session.get('step_idx')}")
    
    print("\n✅ Flow engine test completed!")

if __name__ == "__main__":
    test_flow_engine()
