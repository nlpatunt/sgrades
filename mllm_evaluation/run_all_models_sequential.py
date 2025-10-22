#!/usr/bin/env python3
import os
import json
import time
import sys
import requests
import sqlite3
from datetime import datetime
from test_zero_shot_all import test_zero_shot_all_datasets

def submit_to_database(dataset_name, model_name, metrics, evaluation_details):
    try:
        db_path = '../besesr.db'
        
        evaluation_result = {
            'real_evaluation': {
                'status': 'success',
                'metrics': {
                    'quadratic_weighted_kappa': float(metrics.get('quadratic_weighted_kappa', 0)),
                    'pearson_correlation': float(metrics.get('pearson_correlation', 0)),
                    'mean_absolute_error': float(metrics.get('mean_absolute_error', 0)),
                    'mean_squared_error': float(metrics.get('mean_squared_error', 0)),
                    'root_mean_squared_error': float(metrics.get('root_mean_squared_error', 0)),
                    'f1_score': float(metrics.get('f1_score', 0)),
                    'precision': float(metrics.get('precision', 0)),
                    'recall': float(metrics.get('recall', 0)),
                    'accuracy': float(metrics.get('accuracy', 0))
                },
                'evaluation_timestamp': datetime.now().isoformat(),
                'file_hash_at_evaluation': 'sequential_batch_import'
            },
            'evaluation_timestamp': datetime.now().isoformat(),
            'file_hash_at_evaluation': 'sequential_batch_import'
        }
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM output_submissions 
            WHERE submitter_name = ? AND dataset_name = ?
        """, (model_name, dataset_name))
        
        if cursor.fetchone():
            print(f"    Database: Entry already exists for {dataset_name}, skipping")
            conn.close()
            return True
        
        # Insert new entry with correct column names
        cursor.execute("""
            INSERT INTO output_submissions 
            (dataset_name, submitter_name, submitter_email, 
             evaluation_result, status, description, 
             upload_timestamp, stored_file_path, original_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_name,
            model_name,
            'Tester 1',
            json.dumps(evaluation_result),
            'completed',
            model_name,
            datetime.now().isoformat(),
            f'sequential_batch_{dataset_name}.csv',
            f'sequential_batch_{dataset_name}.csv'
        ))
        
        conn.commit()
        conn.close()
        
        print(f"    Database: ✓ Saved {dataset_name} to database")
        return True
        
    except Exception as e:
        print(f"    Database: ✗ Failed to save {dataset_name}: {e}")
        return False

def run_single_model_with_retries(model_code, model_name, max_retries=3):
    """Run a single model with retry logic and auto database save"""
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}")
            
            # Run the evaluation
            results = test_zero_shot_all_datasets(model_code, model_name, num_essays=None)
            
            if results and len(results) > 0:
                print(f"  ✓ Evaluation completed with {len(results)} datasets")
                print(f"  📤 Submitting {len(results)} results to database...")
                
                # Submit each result to the database
                successful_submissions = 0
                for result in results:
                    dataset_name = result.get('dataset')
                    metrics = result.get('metrics', {})
                    
                    if dataset_name and metrics:
                        if submit_to_database(dataset_name, model_name, metrics, result):
                            successful_submissions += 1
                        time.sleep(0.5)  # Small delay between database operations
                
                print(f"  ✓ Database submissions: {successful_submissions}/{len(results)}")
                
                # Check current database status
                try:
                    import sqlite3
                    conn = sqlite3.connect('../besesr.db')
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(DISTINCT dataset_name) 
                        FROM output_submissions 
                        WHERE submitter_name = ?
                    """, (model_name,))
                    dataset_count = cursor.fetchone()[0]
                    conn.close()
                    
                    print(f"  📊 {model_name} now has {dataset_count} datasets in database")
                    
                    if dataset_count >= 22:  # Your threshold
                        print(f"  🏆 {model_name} should appear on leaderboard!")
                    
                except Exception as e:
                    print(f"  Database check failed: {e}")
            
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
    
    test_datasets = ["D_BEEtlE_3way"]
    
    for dataset in test_datasets:
        try:
            print(f"Testing {dataset}...")
            # Import your single model tester here
            from mllm_evaluation.single_model_test import SingleModelTester
            tester = SingleModelTester()
            result = tester.run_single_test(dataset, "claude-haiku", num_essays=5)
            
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

def check_leaderboard_status():
    """Check current leaderboard status"""
    try:
        import sqlite3
        conn = sqlite3.connect('../besesr.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT submitter_name, COUNT(DISTINCT dataset_name) as datasets 
            FROM output_submissions 
            GROUP BY submitter_name 
            HAVING COUNT(DISTINCT dataset_name) >= 20
            ORDER BY datasets DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        print("\n📊 Current Leaderboard Status:")
        print("=" * 50)
        for model_name, dataset_count in results:
            status = "🏆 ON LEADERBOARD" if dataset_count >= 22 else "⚠️  NEEDS MORE"
            print(f"{model_name}: {dataset_count} datasets {status}")
        
        if not results:
            print("No models with 20+ datasets found")
        
        return results
        
    except Exception as e:
        print(f"Failed to check leaderboard status: {e}")
        return []

def run_all_models_sequential():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv('../.env')
    
    # Check API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        print("Create a .env file with your OpenRouter API key")
        return
    
    # Check current leaderboard status
    check_leaderboard_status()
    
    print("\nRunning pipeline validation first...")
    if not test_pipeline_validation():
        print("Pipeline validation failed. Fix issues before running full test.")
        return
    
    # Define all models to test
    models_to_test = [
        ("claude-haiku", "zero-shot-claude-haiku_test_full"),
        ("gpt4o-mini", "zero-shot-gpt4o-mini_test_full"),
    ]
    
    print("=" * 70)
    print("SEQUENTIAL ZERO-SHOT TESTING WITH AUTO DATABASE SAVE")
    print("=" * 70)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to test: {len(models_to_test)}")
    for i, (model_code, model_name) in enumerate(models_to_test, 1):
        print(f"  {i}. {model_name}")
    
    print("\n📋 Features:")
    print("  • 100 essays per dataset (or all available)")
    print("  • Automatic database submission after each evaluation")
    print("  • Real-time leaderboard status updates")
    print("  • Results appear on leaderboard immediately when complete")
    print("  • JSON backup files for safety")
    
    print("\n⚠️  Estimates:")
    print("  • Time: 2-3 hours per model")
    print("  • Cost: $30-50 per model in API calls")
    
    confirm = input("\nContinue with sequential testing? (y/N): ")
    if confirm.lower() != 'y':
        print("Cancelled")
        return
    
    # Results tracking
    all_results = []
    successful_models = 0
    failed_models = []
    model_results = {}  # Track results per model
    
    start_time = datetime.now()
    
    for i, (model_code, model_name) in enumerate(models_to_test, 1):
        model_start_time = datetime.now()
        
        print("\n" + "=" * 70)
        print(f"TESTING MODEL {i}/{len(models_to_test)}: {model_name}")
        print(f"Started at: {model_start_time.strftime('%H:%M:%S')}")
        print("=" * 70)
        
        try:
            # Run the test for this model with auto database submission
            results = run_single_model_with_retries(model_code, model_name, max_retries=3)
            
            if results and len(results) > 0:
                all_results.extend(results)
                model_results[model_name] = results  # Store results for this model
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
                    best_dataset = max(results, key=lambda x: x['metrics'].get('pearson_correlation', 0))
                    worst_dataset = min(results, key=lambda x: x['metrics'].get('pearson_correlation', 0))
                    
                    print(f"  Average correlation: {avg_correlation:.3f}")
                    print(f"  Average MAE: {avg_mae:.3f}")
                    print(f"  Best performing dataset: {best_dataset['dataset']} (r={best_dataset['metrics'].get('pearson_correlation', 0):.3f})")
                    print(f"  Worst performing dataset: {worst_dataset['dataset']} (r={worst_dataset['metrics'].get('pearson_correlation', 0):.3f})")
                
                # Save individual model results file immediately after completion
                individual_timestamp = int(time.time())
                individual_file = f"{model_name}_{individual_timestamp}.json"
                
                individual_data = {
                    "model": model_name,
                    "completed_at": model_end_time.isoformat(),
                    "duration": str(model_duration),
                    "datasets_tested": len(results),
                    "results": results
                }
                
                with open(individual_file, 'w') as f:
                    json.dump(individual_data, f, indent=2, default=str)
                
                print(f"  Individual results saved to: {individual_file}")
                
                # Check updated leaderboard status
                check_leaderboard_status()
                
            else:
                print(f"\n✗ {model_name} FAILED - No successful results")
                failed_models.append(model_name)
            
        except Exception as e:
            print(f"\n✗ {model_name} FAILED with exception: {e}")
            failed_models.append(model_name)
        
        # Save progress after each model (backup)
        timestamp = int(time.time())
        progress_file = f"sequential_progress_{model_name}_{timestamp}.json"
        
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
    
    # Final leaderboard status
    print("\n🏆 FINAL LEADERBOARD STATUS:")
    check_leaderboard_status()
    
    print(f"\nAll testing complete! {len(all_results)} total successful evaluations.")
    print("Check your leaderboard - all models should appear if they completed successfully.")
    
    return all_results

if __name__ == "__main__":
    results = run_all_models_sequential()
    
    if results:
        print(f"\nAll testing complete! {len(results)} total successful evaluations.")
        print("Check your leaderboard - all models should appear if they completed successfully.")
    else:
        print("No successful tests completed.")