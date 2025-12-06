#!/usr/bin/env python3
"""Sample 14% - CSV ONLY - Keep ALL OS_Dataset data"""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset

SAMPLE_PERCENTAGE = 0.14
RANDOM_SEED = 42
OUTPUT_DIR = "sampled_datasets"
os.environ["HF_TOKEN"] = "REMOVED_KEY"

TEST_DATASETS = {
    'ASAP-AES': {'hf_path': 'nlpatunt/D_ASAP-AES', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'ASAP2': {'hf_path': 'nlpatunt/D_ASAP2', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'ASAP_plus_plus': {'hf_path': 'nlpatunt/D_ASAP_plus_plus', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'persuade_2': {'hf_path': 'nlpatunt/D_persuade_2', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'Ielts_Writing_Dataset': {'hf_path': 'nlpatunt/D_Ielts_Writing_Dataset', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'Ielts_Writing_Task_2_Dataset': {'hf_path': 'nlpatunt/D_Ielts_Writing_Task_2_Dataset', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'Regrading_Dataset_J2C': {'hf_path': 'nlpatunt/D_Regrading_Dataset_J2C', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'ASAP-SAS': {'hf_path': 'nlpatunt/D_ASAP-SAS', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'CSEE': {'hf_path': 'nlpatunt/D_CSEE', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'Mohlar': {'hf_path': 'nlpatunt/D_Mohlar', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'BEEtlE': {'hf_path': 'nlpatunt/D_BEEtlE', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    'SciEntSBank': {'hf_path': 'nlpatunt/D_SciEntSBank', 'split': 'test', 'config': None, 'filter_column': None, 'keep_all': False},
    
    # OS_Dataset - KEEP ALL DATA (too small to sample)
    'OS_Dataset_q1': {'hf_path': 'nlpatunt/D_OS_Dataset', 'split': 'test', 'config': None, 'filter_column': ('question_id', 1), 'keep_all': True},
    'OS_Dataset_q2': {'hf_path': 'nlpatunt/D_OS_Dataset', 'split': 'test', 'config': None, 'filter_column': ('question_id', 2), 'keep_all': True},
    'OS_Dataset_q3': {'hf_path': 'nlpatunt/D_OS_Dataset', 'split': 'test', 'config': None, 'filter_column': ('question_id', 3), 'keep_all': True},
    'OS_Dataset_q4': {'hf_path': 'nlpatunt/D_OS_Dataset', 'split': 'test', 'config': None, 'filter_column': ('question_id', 4), 'keep_all': True},
    'OS_Dataset_q5': {'hf_path': 'nlpatunt/D_OS_Dataset', 'split': 'test', 'config': None, 'filter_column': ('question_id', 5), 'keep_all': True},
    
    # Rice_Chem - Sample 14%
    'Rice_Chem_Q1': {'hf_path': 'nlpatunt/D_Rice_Chem', 'split': 'test', 'config': 'Q1', 'filter_column': None, 'keep_all': False},
    'Rice_Chem_Q2': {'hf_path': 'nlpatunt/D_Rice_Chem', 'split': 'test', 'config': 'Q2', 'filter_column': None, 'keep_all': False},
    'Rice_Chem_Q3': {'hf_path': 'nlpatunt/D_Rice_Chem', 'split': 'test', 'config': 'Q3', 'filter_column': None, 'keep_all': False},
    'Rice_Chem_Q4': {'hf_path': 'nlpatunt/D_Rice_Chem', 'split': 'test', 'config': 'Q4', 'filter_column': None, 'keep_all': False},
}

def sample_dataset(dataset_key, dataset_info, sample_pct=0.14, seed=42):
    print(f"\n📊 {dataset_key}...", end=" ")
    try:
        # Load dataset
        if dataset_info.get('config'):
            dataset = load_dataset(dataset_info['hf_path'], dataset_info['config'], split=dataset_info['split'], trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_info['hf_path'], split=dataset_info['split'], trust_remote_code=True)
        
        df = dataset.to_pandas()
        
        # Apply filter if specified
        if dataset_info.get('filter_column'):
            filter_col, filter_val = dataset_info['filter_column']
            df = df[df[filter_col] == filter_val].reset_index(drop=True)
        
        original_size = len(df)
        
        # Check if we should keep all data
        if dataset_info.get('keep_all'):
            sampled_df = df
            print(f"✅ {original_size} → {len(sampled_df)} (KEPT ALL)")
        else:
            # Sample 14%
            sample_size = max(1, int(original_size * sample_pct))
            np.random.seed(seed)
            sampled_df = df.sample(n=sample_size, random_state=seed)
            print(f"✅ {original_size} → {len(sampled_df)} (14%)")
        
        return {
            'status': 'success',
            'dataset_name': dataset_key,
            'original_size': original_size,
            'sample_size': len(sampled_df),
            'data': sampled_df
        }
    except Exception as e:
        print(f"❌ {e}")
        return {'status': 'error', 'dataset_name': dataset_key}

def main():
    print("="*70)
    print("SAMPLING STRATEGY:")
    print("  - Most datasets: 14% sample")
    print("  - OS_Dataset (q1-q5): 100% (too small to sample)")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    
    for key, info in TEST_DATASETS.items():
        result = sample_dataset(key, info, SAMPLE_PERCENTAGE, RANDOM_SEED)
        if result['status'] == 'success':
            csv_path = os.path.join(OUTPUT_DIR, f"{result['dataset_name']}_sampled_14pct.csv")
            result['data'].to_csv(csv_path, index=False)
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_orig = sum(r['original_size'] for r in results)
    total_samp = sum(r['sample_size'] for r in results)
    print(f"Datasets saved: {len(results)}")
    print(f"Total original: {total_orig:,}")
    print(f"Total sampled: {total_samp:,}")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)

if __name__ == "__main__":
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == 'y':
        main()
    else:
        print("Cancelled.")