#!/usr/bin/env python3
"""
Week 11: Model Performance Testing & Enhancement
Tests the NLU model performance and identifies areas for improvement.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.router import guess_intent, intent_to_flow
import json

def test_model_performance():
    """Test the NLU model performance across different intents"""
    
    print("🧠 Testing NLU Model Performance")
    print("=" * 50)
    
    # Test cases organized by expected intent
    test_cases = {
        "breathing_exercise": [
            "I'm feeling anxious",
            "I need breathing exercises", 
            "I'm panicking",
            "I can't breathe",
            "I'm having a panic attack",
            "I feel overwhelmed",
            "I'm hyperventilating",
            "I need to calm down"
        ],
        "anger_management": [
            "I'm angry",
            "I'm furious",
            "I'm irritated",
            "I'm frustrated",
            "I'm mad",
            "I want to punch something",
            "I'm boiling with rage",
            "I'm livid"
        ],
        "affirmation_request": [
            "I feel sad",
            "I'm depressed",
            "I'm feeling down",
            "I'm blue",
            "I'm feeling low",
            "I need encouragement",
            "I'm feeling hopeless",
            "I need support"
        ],
        "grounding_exercise": [
            "I feel numb",
            "I'm dissociating",
            "I feel disconnected",
            "I'm spaced out",
            "I feel detached",
            "I'm not present",
            "I feel like I'm floating",
            "I need grounding"
        ],
        "journal_prompt": [
            "I want to journal",
            "I need to write",
            "I want a journal prompt",
            "I need to process my thoughts",
            "I want to reflect",
            "I need writing prompts"
        ],
        "thanks_goodbye": [
            "I'm good thanks",
            "I'm fine",
            "I'm okay",
            "I'm great",
            "I'm better",
            "Thank you",
            "Thanks",
            "I'm all set"
        ]
    }
    
    results = {}
    total_tests = 0
    total_correct = 0
    
    for expected_intent, test_inputs in test_cases.items():
        print(f"\n📋 Testing {expected_intent.upper()}:")
        print("-" * 40)
        
        correct = 0
        for test_input in test_inputs:
            predicted_intent = guess_intent(test_input)
            predicted_flow = intent_to_flow(predicted_intent)
            
            # Check if prediction is correct
            is_correct = predicted_intent == expected_intent
            
            if is_correct:
                status = "✅"
                correct += 1
            else:
                status = "❌"
            
            print(f"{status} '{test_input}' -> {predicted_intent} ({predicted_flow})")
            total_tests += 1
        
        accuracy = (correct / len(test_inputs)) * 100
        results[expected_intent] = {
            'correct': correct,
            'total': len(test_inputs),
            'accuracy': accuracy
        }
        
        total_correct += correct
        print(f"Accuracy: {correct}/{len(test_inputs)} ({accuracy:.1f}%)")
    
    # Overall summary
    overall_accuracy = (total_correct / total_tests) * 100
    
    print("\n" + "=" * 50)
    print("📊 MODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    for intent, stats in results.items():
        print(f"{intent}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    print(f"\nOverall Accuracy: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
    
    if overall_accuracy >= 80:
        print("🎉 Model performance is EXCELLENT!")
    elif overall_accuracy >= 70:
        print("✅ Model performance is GOOD!")
    elif overall_accuracy >= 60:
        print("⚠️  Model performance needs improvement.")
    else:
        print("❌ Model performance is POOR - needs significant improvement.")
    
    return results, overall_accuracy

def identify_improvement_areas(results):
    """Identify areas where the model needs improvement"""
    
    print("\n🔍 IDENTIFYING IMPROVEMENT AREAS")
    print("=" * 50)
    
    improvement_needed = []
    
    for intent, stats in results.items():
        if stats['accuracy'] < 80:
            improvement_needed.append({
                'intent': intent,
                'accuracy': stats['accuracy'],
                'correct': stats['correct'],
                'total': stats['total']
            })
    
    if improvement_needed:
        print("Areas needing improvement:")
        for area in improvement_needed:
            print(f"  • {area['intent']}: {area['accuracy']:.1f}% ({area['correct']}/{area['total']})")
        
        print("\n💡 Recommendations:")
        print("  • Add more training examples for low-performing intents")
        print("  • Review keyword patterns in router.py")
        print("  • Consider retraining the model with additional data")
    else:
        print("✅ All intents performing well (≥80% accuracy)")
    
    return improvement_needed

def test_edge_cases():
    """Test edge cases and unusual inputs"""
    
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    edge_cases = [
        ("", "Empty input"),
        ("a", "Single character"),
        ("123", "Numbers only"),
        ("hello world", "Generic greeting"),
        ("I don't know", "Uncertainty"),
        ("maybe", "Ambiguous"),
        ("help me", "Generic help"),
        ("what should I do", "Generic question"),
        ("I'm not sure", "Uncertainty"),
        ("I need help", "Generic help")
    ]
    
    for test_input, description in edge_cases:
        predicted_intent = guess_intent(test_input)
        predicted_flow = intent_to_flow(predicted_intent)
        print(f"'{test_input}' ({description}) -> {predicted_intent} ({predicted_flow})")

def generate_enhancement_recommendations(results, overall_accuracy):
    """Generate specific recommendations for model enhancement"""
    
    print("\n💡 MODEL ENHANCEMENT RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = []
    
    if overall_accuracy < 80:
        recommendations.append("🔧 Overall model accuracy below 80% - consider retraining")
    
    # Check specific intents
    for intent, stats in results.items():
        if stats['accuracy'] < 70:
            recommendations.append(f"🔧 {intent} accuracy very low ({stats['accuracy']:.1f}%) - needs immediate attention")
        elif stats['accuracy'] < 80:
            recommendations.append(f"⚠️  {intent} accuracy below 80% ({stats['accuracy']:.1f}%) - could be improved")
    
    # General recommendations
    recommendations.extend([
        "📚 Add more diverse training examples",
        "🔄 Implement data augmentation techniques", 
        "🎯 Fine-tune keyword patterns in router.py",
        "🧪 Test with more edge cases",
        "📊 Collect user feedback for model improvement"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return recommendations

if __name__ == "__main__":
    print("🚀 Week 11: Model Performance Testing")
    print("=" * 60)
    
    # Run comprehensive model testing
    results, overall_accuracy = test_model_performance()
    
    # Identify improvement areas
    improvement_areas = identify_improvement_areas(results)
    
    # Test edge cases
    test_edge_cases()
    
    # Generate recommendations
    recommendations = generate_enhancement_recommendations(results, overall_accuracy)
    
    print("\n" + "=" * 60)
    print("🎯 Model testing COMPLETED!")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    
    if overall_accuracy >= 80:
        print("✅ Model is performing well - ready for final validation!")
    else:
        print("⚠️  Model needs enhancement before final release.")
    print("=" * 60)
