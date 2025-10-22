#!/usr/bin/env python3
import os
from single_model_test_few_shot import FewShotModelTester

def test_one_dataset():
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found")
        return
    
    tester = FewShotModelTester()
    
    # Test connection
    if not tester.test_connection():
        print("ERROR: Cannot connect to BESESR")
        return
    
    # Test one dataset
    dataset = "D_ASAP-AES"  # Change this to test different datasets
    
    print(f"Testing few-shot on {dataset}")
    print("This will:")
    print("1. Download training data")
    print("2. Select 5 diverse examples") 
    print("3. Test few-shot prompting on a few essays")
    
    # Test the training data download first
    print("\nStep 1: Testing training data download...")
    training_df = tester.download_training_data(dataset)
    
    if training_df is None:
        print("Failed to download training data")
        return
    
    print(f"Success! Found {len(training_df)} training examples")
    print(f"Columns: {list(training_df.columns)}")
    
    # Test example selection
    print("\nStep 2: Testing example selection...")
    examples = tester.select_training_examples(training_df, dataset, num_examples=5)
    
    if not examples:
        print("Failed to select examples")
        return
    
    print(f"Success! Selected {len(examples)} examples:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. Score: {ex['score']} | Text: {ex['text'][:100]}...")
    
    # Test prompt creation
    print("\nStep 3: Testing prompt creation...")
    test_essay = "Technology has revolutionized education in many ways. Students can now access information instantly and collaborate globally."
    range_info = tester.get_dataset_range_info(dataset)
    
    prompt = tester.create_few_shot_prompt(dataset, test_essay, range_info)
    
    print("Success! Generated few-shot prompt:")
    print("=" * 50)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("=" * 50)
    
    # Test actual API call
    print("\nStep 4: Testing API call...")
    confirm = input("Test actual API call? (costs ~$0.01) (y/N): ")
    
    if confirm.lower() == 'y':
        score = tester.call_gemini_flash_few_shot(test_essay, dataset)
        if score is not None:
            print(f"Success! Predicted score: {score}")
        else:
            print("API call failed")
    
    print("\nFew-shot test completed!")

if __name__ == "__main__":
    test_one_dataset()