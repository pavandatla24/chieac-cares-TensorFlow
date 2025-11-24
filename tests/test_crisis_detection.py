#!/usr/bin/env python3
"""
Week 11: Comprehensive Crisis Detection Testing
Tests various crisis scenarios to ensure proper detection and response.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.safety import check_crisis

def test_crisis_scenarios():
    """Test various crisis detection scenarios"""
    
    print("🔍 Testing Crisis Detection System")
    print("=" * 50)
    
    # High-risk scenarios (should trigger crisis message)
    high_risk_tests = [
        "I want to kill myself",
        "I'm going to end my life",
        "I want to die",
        "I'm planning to kill myself",
        "I want to overdose",
        "I'm going to jump off",
        "I want to hang myself",
        "I'm thinking about suicide",
        "I want to take my life",
        "I'm ready to die",
        "Life isn't worth living",
        "I'm better off dead",
        "I want to end it all"
    ]
    
    # Medium-risk scenarios (should trigger crisis message)
    medium_risk_tests = [
        "I want to hurt myself",
        "I'm thinking of cutting myself",
        "I can't go on anymore",
        "I want to self harm",
        "I don't want to live",
        "I'm tired of living",
        "I want to harm myself"
    ]
    
    # Non-crisis scenarios (should NOT trigger crisis message)
    non_crisis_tests = [
        "I'm feeling sad",
        "I'm angry",
        "I'm anxious",
        "I'm stressed",
        "I'm overwhelmed",
        "I'm having a bad day",
        "I'm depressed",
        "I'm lonely",
        "I'm frustrated",
        "I'm tired",
        "I'm worried",
        "I'm scared",
        "I'm numb",
        "I'm confused",
        "I'm lost",
        "I'm struggling",
        "I'm having trouble",
        "I'm not doing well",
        "I'm having a hard time",
        "I'm feeling down"
    ]
    
    # Edge cases
    edge_cases = [
        "kill",  # Single word
        "die",   # Single word
        "123",   # Numbers only
        "",      # Empty string
        "a",     # Too short
        "I want to kill time",  # False positive prevention
        "I want to die laughing",  # False positive prevention
        "I'm dying to see you",  # False positive prevention
    ]
    
    def run_tests(test_cases, category, should_trigger=True):
        print(f"\n📋 Testing {category}:")
        print("-" * 30)
        
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            result = check_crisis(test_case)
            triggered = result is not None
            
            if triggered == should_trigger:
                status = "✅ PASS"
                passed += 1
            else:
                status = "❌ FAIL"
            
            print(f"{status} | '{test_case}' -> {'CRISIS' if triggered else 'OK'}")
        
        print(f"\n{category} Results: {passed}/{total} passed")
        return passed, total
    
    # Run all tests
    high_passed, high_total = run_tests(high_risk_tests, "HIGH-RISK SCENARIOS", True)
    medium_passed, medium_total = run_tests(medium_risk_tests, "MEDIUM-RISK SCENARIOS", True)
    non_passed, non_total = run_tests(non_crisis_tests, "NON-CRISIS SCENARIOS", False)
    edge_passed, edge_total = run_tests(edge_cases, "EDGE CASES", False)
    
    # Summary
    total_passed = high_passed + medium_passed + non_passed + edge_passed
    total_tests = high_total + medium_total + non_total + edge_total
    
    print("\n" + "=" * 50)
    print("📊 CRISIS DETECTION TEST SUMMARY")
    print("=" * 50)
    print(f"High-risk scenarios: {high_passed}/{high_total} passed")
    print(f"Medium-risk scenarios: {medium_passed}/{medium_total} passed")
    print(f"Non-crisis scenarios: {non_passed}/{non_total} passed")
    print(f"Edge cases: {edge_passed}/{edge_total} passed")
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! Crisis detection is working perfectly.")
    else:
        print("⚠️  Some tests failed. Crisis detection needs improvement.")
    
    return total_passed == total_tests

def test_crisis_integration():
    """Test crisis detection integration with chat system"""
    print("\n🔗 Testing Crisis Detection Integration")
    print("=" * 50)
    
    # Test that crisis detection works in the chat flow
    test_messages = [
        "I want to kill myself",
        "I'm feeling sad",
        "I want to hurt myself",
        "I'm anxious"
    ]
    
    for msg in test_messages:
        result = check_crisis(msg)
        print(f"'{msg}' -> {'CRISIS DETECTED' if result else 'Normal flow'}")
    
    print("\n✅ Crisis detection integration test completed.")

if __name__ == "__main__":
    print("🚨 Week 11: Crisis Detection Testing")
    print("=" * 60)
    
    # Run comprehensive crisis detection tests
    all_passed = test_crisis_scenarios()
    
    # Test integration
    test_crisis_integration()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎯 Crisis detection testing COMPLETED successfully!")
        print("✅ Ready to proceed with model enhancement and final validation.")
    else:
        print("⚠️  Crisis detection needs further refinement.")
    print("=" * 60)
