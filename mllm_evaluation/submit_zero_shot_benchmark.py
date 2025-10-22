#!/usr/bin/env python3
import requests
import json
import os

def submit_complete_zero_shot_benchmark():
    """Submit complete zero-shot benchmark to BESESR leaderboard"""
    
    # Load your test results
    results_file = "zero_shot_results_[timestamp].json"  # Use your actual file
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    base_url = "http://localhost:8000"
    
    
    # Submit each dataset result
    for result in results:
        if result['success']:
            dataset_name = result['dataset']
            
            # Create predictions CSV format
            predictions_csv = create_predictions_csv(result)
            
            # Submit to BESESR
            files = {'file': ('predictions.csv', predictions_csv, 'text/csv')}
            data = {
                'dataset_name': dataset_name,
                'methodology': 'zero-shot'
            }
            
            response = requests.post(
                f"{base_url}/api/submissions/test-single-dataset",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                print(f"✓ Submitted {dataset_name}")
            else:
                print(f"✗ Failed to submit {dataset_name}: {response.status_code}")

def create_predictions_csv(result):
    """Convert result to CSV format"""
    # Implementation depends on your specific result format
    pass

if __name__ == "__main__":
    submit_complete_zero_shot_benchmark()