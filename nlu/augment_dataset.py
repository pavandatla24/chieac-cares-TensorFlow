#!/usr/bin/env python3
"""
Week 9 - Dataset Augmentation
Add more training examples for poorly performing intents
"""

import os
import pandas as pd
import json
import random
from typing import List, Dict

class DatasetAugmenter:
    def __init__(self, dataset_path="nlu/data/training_seed_clean_fixed.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the existing training dataset"""
        if os.path.exists(self.dataset_path):
            self.df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded dataset: {len(self.df)} rows")
        else:
            print(f"❌ Dataset not found: {self.dataset_path}")
            self.df = pd.DataFrame()
    
    def analyze_intent_distribution(self):
        """Analyze the distribution of intents in the dataset"""
        if self.df.empty:
            return {}
        
        intent_counts = self.df['intent'].value_counts()
        print("\n📊 Intent Distribution:")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count} examples")
        
        return intent_counts.to_dict()
    
    def get_weak_intents(self, min_examples=50):
        """Identify intents with fewer than min_examples"""
        intent_counts = self.df['intent'].value_counts()
        weak_intents = intent_counts[intent_counts < min_examples].index.tolist()
        
        print(f"\n🔍 Weak intents (< {min_examples} examples):")
        for intent in weak_intents:
            count = intent_counts[intent]
            print(f"  {intent}: {count} examples")
        
        return weak_intents
    
    def generate_paraphrases(self, text: str, num_variations: int = 3) -> List[str]:
        """Generate paraphrases for a given text"""
        paraphrases = []
        
        # Simple paraphrasing patterns
        patterns = [
            # Add filler words
            lambda t: f"I {t.lower()}",
            lambda t: f"I'm feeling {t.lower()}",
            lambda t: f"I need help with {t.lower()}",
            lambda t: f"Can you help me with {t.lower()}",
            lambda t: f"I'm struggling with {t.lower()}",
            
            # Change tense
            lambda t: t.replace("I am", "I'm").replace("I will", "I'll"),
            lambda t: t.replace("I'm", "I am").replace("I'll", "I will"),
            
            # Add emotional context
            lambda t: f"Right now, {t.lower()}",
            lambda t: f"Today, {t.lower()}",
            lambda t: f"I really {t.lower()}",
            
            # Add urgency
            lambda t: f"I urgently need {t.lower()}",
            lambda t: f"Please help me {t.lower()}",
            lambda t: f"I desperately need {t.lower()}",
        ]
        
        # Generate variations
        for _ in range(num_variations):
            pattern = random.choice(patterns)
            try:
                paraphrase = pattern(text)
                if paraphrase != text and paraphrase not in paraphrases:
                    paraphrases.append(paraphrase)
            except:
                continue
        
        return paraphrases[:num_variations]
    
    def augment_intent(self, intent: str, target_count: int = 100):
        """Augment a specific intent with more examples"""
        if self.df.empty:
            return
        
        # Get existing examples for this intent
        intent_examples = self.df[self.df['intent'] == intent]
        
        if intent_examples.empty:
            print(f"❌ No examples found for intent: {intent}")
            return
        
        current_count = len(intent_examples)
        needed = target_count - current_count
        
        if needed <= 0:
            print(f"✅ Intent '{intent}' already has {current_count} examples")
            return
        
        print(f"\n🔄 Augmenting intent '{intent}' from {current_count} to {target_count} examples")
        
        new_examples = []
        
        # Generate paraphrases from existing examples
        for _, row in intent_examples.iterrows():
            text = row['text']
            paraphrases = self.generate_paraphrases(text, num_variations=2)
            
            for paraphrase in paraphrases:
                if len(new_examples) >= needed:
                    break
                
                new_example = row.copy()
                new_example['text'] = paraphrase
                new_examples.append(new_example)
        
        # Add some generic examples if we still need more
        if len(new_examples) < needed:
            generic_examples = self._get_generic_examples(intent, needed - len(new_examples))
            new_examples.extend(generic_examples)
        
        # Add new examples to dataset
        if new_examples:
            new_df = pd.DataFrame(new_examples)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            print(f"✅ Added {len(new_examples)} new examples for '{intent}'")
    
    def _get_generic_examples(self, intent: str, count: int) -> List[Dict]:
        """Get generic examples for an intent"""
        generic_examples = {
            "breathing_exercise": [
                "I need to breathe",
                "Help me calm down",
                "I'm panicking",
                "I can't breathe properly",
                "I need breathing help",
                "I'm hyperventilating",
                "I need to slow down my breathing",
                "I'm having trouble breathing",
                "I need breathing techniques",
                "I want to learn to breathe better"
            ],
            "grounding_technique": [
                "I feel disconnected",
                "I'm dissociating",
                "I need to ground myself",
                "I feel like I'm floating",
                "I need to feel present",
                "I'm feeling detached",
                "I need grounding techniques",
                "I feel disconnected from reality",
                "I need to come back to earth",
                "I'm feeling spaced out"
            ],
            "affirmation": [
                "I need encouragement",
                "I feel worthless",
                "I need positive thoughts",
                "I'm struggling with self-esteem",
                "I need motivation",
                "I feel like a failure",
                "I need uplifting words",
                "I'm feeling down about myself",
                "I need confidence",
                "I need to feel better about myself"
            ],
            "journal_prompt": [
                "I want to write about my feelings",
                "I need to process my emotions",
                "I want to reflect on today",
                "I need journaling prompts",
                "I want to express myself",
                "I need to write down my thoughts",
                "I want to explore my feelings",
                "I need writing prompts",
                "I want to document my day",
                "I need to get my thoughts out"
            ],
            "anger_management": [
                "I'm so angry",
                "I need to control my temper",
                "I'm furious",
                "I need anger management",
                "I'm about to explode",
                "I need to calm my anger",
                "I'm feeling rage",
                "I need to manage my anger",
                "I'm losing my temper",
                "I need anger control techniques"
            ],
            "crisis_emergency": [
                "I'm in crisis",
                "I need emergency help",
                "I'm having a mental health crisis",
                "I need immediate support",
                "I'm in a bad place",
                "I need crisis intervention",
                "I'm having a breakdown",
                "I need urgent help",
                "I'm in a mental health emergency",
                "I need crisis support"
            ]
        }
        
        examples = generic_examples.get(intent, [])
        if not examples:
            return []
        
        # Get random examples
        selected = random.sample(examples, min(count, len(examples)))
        
        # Create new examples
        new_examples = []
        for text in selected:
            # Get a random example from the intent to copy other fields
            random_example = self.df[self.df['intent'] == intent].iloc[0]
            new_example = random_example.copy()
            new_example['text'] = text
            new_examples.append(new_example)
        
        return new_examples
    
    def save_augmented_dataset(self, output_path=None):
        """Save the augmented dataset"""
        if output_path is None:
            output_path = self.dataset_path.replace('.csv', '_augmented.csv')
        
        self.df.to_csv(output_path, index=False)
        print(f"✅ Augmented dataset saved: {output_path}")
        print(f"   Total examples: {len(self.df)}")
        
        # Show final distribution
        intent_counts = self.df['intent'].value_counts()
        print("\n📊 Final Intent Distribution:")
        for intent, count in intent_counts.items():
            print(f"  {intent}: {count} examples")
    
    def augment_weak_intents(self, target_count=100):
        """Augment all weak intents to target_count"""
        weak_intents = self.get_weak_intents(target_count)
        
        for intent in weak_intents:
            self.augment_intent(intent, target_count)
    
    def create_balanced_dataset(self, target_count=100):
        """Create a balanced dataset with target_count examples per intent"""
        all_intents = self.df['intent'].unique()
        
        for intent in all_intents:
            self.augment_intent(intent, target_count)

def main():
    """Main function for dataset augmentation"""
    print("🔄 ChiEAC CARES - Dataset Augmentation")
    print("=" * 50)
    
    augmenter = DatasetAugmenter()
    
    if augmenter.df.empty:
        print("❌ No dataset loaded. Exiting.")
        return
    
    # Analyze current distribution
    augmenter.analyze_intent_distribution()
    
    while True:
        print("\n1. Analyze intent distribution")
        print("2. Identify weak intents")
        print("3. Augment specific intent")
        print("4. Augment all weak intents")
        print("5. Create balanced dataset")
        print("6. Save augmented dataset")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            augmenter.analyze_intent_distribution()
        elif choice == "2":
            target = int(input("Enter minimum examples per intent (default 50): ") or "50")
            augmenter.get_weak_intents(target)
        elif choice == "3":
            intent = input("Enter intent name: ").strip()
            target = int(input("Enter target count (default 100): ") or "100")
            augmenter.augment_intent(intent, target)
        elif choice == "4":
            target = int(input("Enter target count (default 100): ") or "100")
            augmenter.augment_weak_intents(target)
        elif choice == "5":
            target = int(input("Enter target count per intent (default 100): ") or "100")
            augmenter.create_balanced_dataset(target)
        elif choice == "6":
            output = input("Enter output path (or press Enter for default): ").strip()
            augmenter.save_augmented_dataset(output if output else None)
        elif choice == "7":
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
