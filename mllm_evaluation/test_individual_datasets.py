#!/usr/bin/env python3
from mllm_evaluation.single_model_test import SingleModelTester

def test_individual_datasets():
    tester = SingleModelTester()
    datasets = tester.get_available_datasets()
    test_datasets = [ds for ds in datasets if ds.startswith('D_')]
    
    working_datasets = []
    problematic_datasets = []
    
    for i, dataset in enumerate(test_datasets, 1): 
        print(f"\nTesting {i}/{len(test_datasets)}: {dataset}")
        try:
            result = tester.run_single_test(dataset, "gemini-flash", num_essays=10)
            if result and result.get('success'):
                working_datasets.append(dataset)
                print(f"✓ {dataset}: WORKS")
            else:
                problematic_datasets.append(dataset)
                print(f"✗ {dataset}: FAILED")
        except Exception as e:
            problematic_datasets.append(dataset)
            print(f"✗ {dataset}: ERROR - {e}")
    
    print(f"\nWorking datasets: {working_datasets}")
    print(f"Problematic datasets: {problematic_datasets}")

if __name__ == "__main__":
    test_individual_datasets()