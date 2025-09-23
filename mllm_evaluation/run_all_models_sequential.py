#!/usr/bin/env python3
import os
import json
import time
import sys
from datetime import datetime
from test_zero_shot_all import test_zero_shot_all_datasets

def run_single_model_with_retries(model_code, model_name, max_retries=3):
    """Run a single model with retry logic for better reliability"""
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}")
            # Modified to include num_essays=100 parameter
            results = test_zero_shot_all_datasets(model_code, model_name, num_essays=100)
            return results
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Waiting 5 minutes before retry...")
                time.sleep(300)  # Wait 5 minutes before retry
            else:
                print(f"  All {max_retries} attempts failed")
                raise

def test_pipeline_validation():
    """Test the pipeline with a small subset first"""
    print("Running validation test...")
    
    test_datasets = ["D_BEEtlE_3way", "D_Rice_Chem_Q1", "D_ASAP-AES"]
    
    for dataset in test_datasets:
        try:
            print(f"Testing {dataset}...")
            # Import your single model tester here
            from single_model_test import SingleModelTester
            tester = SingleModelTester()
            result = tester.run_single_test(dataset, "gemini-flash", num_essays=5)
            
            if result and result.get('success'):
                print(f"  ✓ {dataset} passed")
            else:
                print(f"  ✗ {dataset} failed")
                return False
        except Exception as e:
            print(f"  ✗ {dataset} error: {e}")
            return False
    
    print("✓ Pipeline validation passed")
    return True

def run_all_models_sequential():
    """Run all 5 models sequentially through the night"""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('../.env')
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        print("Create a .env file with your OpenRouter API key")
        return
    
    print("Running pipeline validation first...")
    if not test_pipeline_validation():
        print("Pipeline validation failed. Fix issues before running full test.")
        return
    
    # Define all models to test
    models_to_test = [
        ("gpt4o-mini", "zero-shot-gpt4o-mini_test1"),
        #("claude-haiku", "zero-shot-claude-haiku_test1"),
        
        #("gemini-flash", "zero-shot-gemini-flash_test1"),
        #("gemini-pro", "zero-shot-gemini-pro_test1"),
        #("claude-sonnet", "zero-shot-claude-sonnet_test1")
    ]
    
    print("=" * 70)
    print("SEQUENTIAL ZERO-SHOT TESTING - ALL MODELS (100 essays per dataset)")
    print("=" * 70)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to test: {len(models_to_test)}")
    for i, (model_code, model_name) in enumerate(models_to_test, 1):
        print(f"  {i}. {model_name}")
    
    print("\nThis will run automatically through the night.")
    print("Each dataset will use 100 essays (or all available if fewer than 100)")
    print("Estimated total time: 2-3 hours per model")
    print("Estimated total cost: $30-50 per model in API calls")
    
    confirm = input("\nContinue with sequential testing? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return
    
    # Results tracking
    all_results = []
    successful_models = 0
    failed_models = []
    
    start_time = datetime.now()
    
    for i, (model_code, model_name) in enumerate(models_to_test, 1):
        model_start_time = datetime.now()
        
        print("\n" + "=" * 70)
        print(f"TESTING MODEL {i}/{len(models_to_test)}: {model_name}")
        print(f"Started at: {model_start_time.strftime('%H:%M:%S')}")
        print("=" * 70)
        
        try:
            # Run the test for this model with 100 essays per dataset
            results = run_single_model_with_retries(model_code, model_name, max_retries=3)
            
            if results and len(results) > 0:
                all_results.extend(results)
                successful_models += 1
                
                model_end_time = datetime.now()
                model_duration = model_end_time - model_start_time
                
                print(f"\n✓ {model_name} COMPLETED SUCCESSFULLY")
                print(f"  Duration: {model_duration}")
                print(f"  Datasets tested: {len(results)}")
                
                # Calculate average performance
                if results:
                    avg_correlation = sum(r['metrics'].get('pearson_correlation', 0) for r in results) / len(results)
                    avg_mae = sum(r['metrics'].get('mean_absolute_error', 0) for r in results) / len(results)
                    print(f"  Average correlation: {avg_correlation:.3f}")
                    print(f"  Average MAE: {avg_mae:.3f}")
                
            else:
                print(f"\n✗ {model_name} FAILED - No successful results")
                failed_models.append(model_name)
            
        except Exception as e:
            print(f"\n✗ {model_name} FAILED with exception: {e}")
            failed_models.append(model_name)
        
        # Save progress after each model
        timestamp = int(time.time())
        progress_file = f"sequential_progress_{timestamp}.json"
        
        progress_data = {
            "completed_models": successful_models,
            "total_models": len(models_to_test),
            "successful_results": len(all_results),
            "failed_models": failed_models,
            "last_updated": datetime.now().isoformat(),
            "results": all_results
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)
        
        print(f"Progress saved to: {progress_file}")
        
        # Break between models (except for the last one)
        if i < len(models_to_test):
            print(f"\nWaiting 60 seconds before starting next model...")
            time.sleep(60)
    
    # Final summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("SEQUENTIAL TESTING COMPLETE")
    print("=" * 70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration}")
    print(f"Successful models: {successful_models}/{len(models_to_test)}")
    print(f"Total successful tests: {len(all_results)}")
    
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    
    # Save final results
    final_timestamp = int(time.time())
    final_results_file = f"sequential_final_results_{final_timestamp}.json"
    
    final_data = {
        "summary": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration": str(total_duration),
            "successful_models": successful_models,
            "total_models": len(models_to_test),
            "total_successful_tests": len(all_results),
            "failed_models": failed_models
        },
        "all_results": all_results
    }
    
    with open(final_results_file, 'w') as f:
        json.dump(final_data, f, indent=2, default=str)
    
    print(f"Final results saved to: {final_results_file}")
    
    # Check leaderboard status
    print(f"\nTo check leaderboard status:")
    print(f"sqlite3 besesr.db \"SELECT submitter_name, COUNT(DISTINCT dataset_name) as datasets FROM output_submissions GROUP BY submitter_name ORDER BY datasets DESC;\"")
    
    return all_results

if __name__ == "__main__":
    results = run_all_models_sequential()
    
    if results:
        print(f"\nAll testing complete! {len(results)} total successful evaluations.")
        print("Check your leaderboard - all models should appear if they completed successfully.")
    else:
        print("No successful tests completed.")