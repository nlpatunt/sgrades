#!/usr/bin/env python3
import os
import json
import time
import sys
from mllm_evaluation.lama_exp.single_model_test import SingleModelTester

def test_zero_shot_all_datasets(model_code=None, model_name=None, num_essays=3):
    tester = SingleModelTester()
    
    if not tester.test_connection():
        print("ERROR: Cannot connect to BESESR at localhost:8000")
        print("Make sure your BESESR server is running!")
        return
    
    # Get all available datasets
    datasets = tester.get_available_datasets()
    
    if not datasets:
        print("ERROR: No datasets found. Check your BESESR API.")
        return
    
    # Filter for test datasets (D_ prefix)
    test_datasets = [ds for ds in datasets if ds.startswith('D_')]
    
    print(f"DEBUG: Found {len(test_datasets)} test datasets")
    print("Test datasets:", test_datasets[:5], "..." if len(test_datasets) > 5 else "")

    # Use parameters or defaults
    model = model_code or "gemini-flash" 
    LEADERBOARD_MODEL_NAME = model_name or "Zero_Shot_5_Essays_Gemini-Flash-1.5"
    
    results = []
    successful_tests = 0
    
    for i, dataset in enumerate(test_datasets, 1):
        print(f"\n{'='*60}")
        print(f"[{LEADERBOARD_MODEL_NAME}] Testing {i}/{len(test_datasets)}: {dataset}")
        print(f"{'='*60}")
        
        try:
            result = tester.run_single_test(dataset, model, num_essays=num_essays, 
                              leaderboard_name=LEADERBOARD_MODEL_NAME)
            
            if result and result.get('success'):
                results.append(result)
                successful_tests += 1
                print(f"✓ [{LEADERBOARD_MODEL_NAME}] {dataset}: SUCCESS - Correlation: {result['metrics'].get('pearson_correlation', 0):.3f}")
            else:
                print(f"✗ [{LEADERBOARD_MODEL_NAME}] {dataset}: FAILED - No ground truth evaluation")
                
        except Exception as e:
            print(f"✗ [{LEADERBOARD_MODEL_NAME}] {dataset}: ERROR - {e}")
  
        time.sleep(5)
    
    # Save results
    timestamp = int(time.time())
    results_file = f"zero_shot_{model}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"[{LEADERBOARD_MODEL_NAME}] ZERO-SHOT TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful tests: {successful_tests}/{len(test_datasets)}")
    print(f"Results saved to: {results_file}")
    
    # Summary of performance
    if results:
        correlations = [r['metrics'].get('pearson_correlation', 0) for r in results]
        maes = [r['metrics'].get('mean_absolute_error', 0) for r in results]
        
        print(f"[{LEADERBOARD_MODEL_NAME}] Average Correlation: {sum(correlations)/len(correlations):.3f}")
        print(f"[{LEADERBOARD_MODEL_NAME}] Average MAE: {sum(maes)/len(maes):.3f}")
        
        # Show best and worst performing datasets
        results_sorted = sorted(results, key=lambda x: x['metrics'].get('pearson_correlation', 0), reverse=True)
        print(f"\n[{LEADERBOARD_MODEL_NAME}] Best performing dataset: {results_sorted[0]['dataset']} (r={results_sorted[0]['metrics'].get('pearson_correlation', 0):.3f})")
        print(f"[{LEADERBOARD_MODEL_NAME}] Worst performing dataset: {results_sorted[-1]['dataset']} (r={results_sorted[-1]['metrics'].get('pearson_correlation', 0):.3f})")
    
    return results

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        print("Create a .env file with your OpenRouter API key")
        exit(1)
    
    model_mappings = {
        "gpt4o": ("gpt4o", "zero-shot-gpt4o_test1"),
        "gpt4o-mini": ("gpt4o-mini", "zero-shot-gpt4o-mini_test1"),
        "claude-sonnet": ("claude-sonnet", "zero-shot-claude-sonnet_test1"),
        "claude-haiku": ("claude-haiku", "zero-shot-claude-haiku_test1"),
        "gemini-flash": ("gemini-flash", "zero-shot-gemini-flash_test1"),
        "gemini-pro": ("gemini-pro", "zero-shot-gemini-pro_test1")
    }
        
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if model_arg in model_mappings:
            model_code, model_name = model_mappings[model_arg]
        else:
            print(f"Unknown model: {model_arg}")
            print(f"Available models: {list(model_mappings.keys())}")
            exit(1)
    else:
        model_code, model_name = model_mappings["gemini-flash"]  # Default
    
    print(f"Starting {model_name} benchmark on all datasets...")
    print("This will process ALL essays in each dataset")
    print("Estimated cost per model: $10-20 in API calls")
    
    confirm = input(f"Continue with {model_name}? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        exit(0)
    
    results = test_zero_shot_all_datasets(model_code, model_name)
    
    if results:
        print(f"\n[{model_name}] Ready to submit {len(results)} successful evaluations to leaderboard!")
        print("Results are already automatically submitted to BESESR")
    else:
        print("No successful tests to submit")