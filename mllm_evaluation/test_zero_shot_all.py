#!/usr/bin/env python3
import os
import json
import time
from single_model_test import SingleModelTester

def test_zero_shot_all_datasets():
    """Test zero-shot on all available datasets"""
    
    tester = SingleModelTester()
    
    # Test connection first
    if not tester.test_connection():
        print("ERROR: Cannot connect to BESESR at localhost:8000")
        print("Make sure your BESESR server is running!")
        return
    
    # Get all available datasets
    datasets = tester.get_available_datasets()
    
    if not datasets:
        print("ERROR: No datasets found. Check your BESESR API.")
        return
    
    # Filter for test datasets (D_ prefix) or create them
    test_datasets = []
    for ds in datasets:
        if ds.startswith('D_'):
            test_datasets.append(ds)
        else:
            # Check if D_ version exists
            d_version = f"D_{ds}"
            if d_version in datasets:
                test_datasets.append(d_version)
    
    print(f"Found {len(test_datasets)} test datasets")
    print("Test datasets:", test_datasets[:5], "..." if len(test_datasets) > 5 else "")

    model = "gemini-flash" 
    
    results = []
    successful_tests = 0
    
    for i, dataset in enumerate(test_datasets, 1):
        print(f"\n{'='*60}")
        print(f"Testing {i}/{len(test_datasets)}: {dataset}")
        print(f"{'='*60}")
        
        try:
            # Test with limited essays first (10 essays per dataset)
            result = tester.run_single_test(dataset, model, num_essays=None)
            
            if result and result.get('success'):
                results.append(result)
                successful_tests += 1
                print(f"✓ {dataset}: SUCCESS - Correlation: {result['metrics'].get('pearson_correlation', 0):.3f}")
            else:
                print(f"✗ {dataset}: FAILED - No ground truth evaluation")
                
        except Exception as e:
            print(f"✗ {dataset}: ERROR - {e}")
            
        # Rate limiting - important for API calls
        time.sleep(10)  # 3 second delay between datasets
    
    # Save results
    timestamp = int(time.time())
    results_file = f"zero_shot_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful tests: {successful_tests}/{len(test_datasets)}")
    print(f"Results saved to: {results_file}")
    
    # Summary of performance
    if results:
        correlations = [r['metrics'].get('pearson_correlation', 0) for r in results]
        maes = [r['metrics'].get('mean_absolute_error', 0) for r in results]
        
        print(f"Average Correlation: {sum(correlations)/len(correlations):.3f}")
        print(f"Average MAE: {sum(maes)/len(maes):.3f}")
        
        # Show best and worst performing datasets
        results_sorted = sorted(results, key=lambda x: x['metrics'].get('pearson_correlation', 0), reverse=True)
        print(f"\nBest performing dataset: {results_sorted[0]['dataset']} (r={results_sorted[0]['metrics'].get('pearson_correlation', 0):.3f})")
        print(f"Worst performing dataset: {results_sorted[-1]['dataset']} (r={results_sorted[-1]['metrics'].get('pearson_correlation', 0):.3f})")
    
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
    
    # Add confirmation prompt here
    print("Starting complete Gemini benchmark on all datasets...")
    print("This will process ALL essays in each dataset (may take 2-4 hours)")
    print("Estimated cost: $10-20 in API calls")
    
    confirm = input("Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        exit(0)
    
    results = test_zero_shot_all_datasets()
    
    if results:
        print(f"\nReady to submit {len(results)} successful evaluations to leaderboard!")
        print("Results are already automatically submitted to BESESR with methodology='zero-shot'")
    else:
        print("No successful tests to submit")