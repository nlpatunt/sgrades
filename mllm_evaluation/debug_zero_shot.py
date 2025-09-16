#!/usr/bin/env python3
from single_model_test import SingleModelTester
import json

def debug_zero_shot_order():
    tester = SingleModelTester()
    datasets = tester.get_available_datasets()
    test_datasets = [ds for ds in datasets if ds.startswith('D_')]
    
    print(f"Found {len(test_datasets)} test datasets")
    
    # Test the specific problematic datasets first
    problem_datasets = ["D_Ielst_Writing_Task_2_Dataset", "D_Ielts_Writing_Dataset"]
    
    for dataset in problem_datasets:
        if dataset in test_datasets:
            print(f"\nTesting {dataset} (quick test)...")
            try:
                # Test just connection and data loading (no API calls)
                df = tester.download_test_data(dataset, num_essays=1)
                if df is not None:
                    essays = tester.prepare_essays_for_prediction(df, dataset)
                    if essays:
                        print(f"✓ {dataset}: Data loads successfully ({len(essays)} essays prepared)")
                    else:
                        print(f"✗ {dataset}: No essays could be prepared")
                else:
                    print(f"✗ {dataset}: Failed to download test data")
                    
            except Exception as e:
                print(f"✗ {dataset}: Exception - {e}")
        else:
            print(f"✗ {dataset}: Not found in dataset list")

if __name__ == "__main__":
    debug_zero_shot_order()