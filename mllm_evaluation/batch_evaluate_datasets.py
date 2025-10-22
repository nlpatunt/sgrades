#!/usr/bin/env python3
"""
Batch evaluate all prediction CSV files in a folder
Calculates metrics for each dataset and combined results
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to path so we can import app module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import your evaluation engine
try:
    from app.api.routes.output_submissions import RealEvaluationEngine
    from app.api.routes.dataset_ranges import get_score_range_for_dataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

engine = RealEvaluationEngine()

def extract_dataset_name_from_filename(filename):
    """Extract dataset name from CSV filename - handles all 23 datasets"""
    name = filename.replace('.csv', '').lower()
    
    # Fix common typos
    name = name.replace('3wy', '3way')
    name = name.replace('2wy', '2way')
    name = name.replace('beetle_', 'beetle_')  # Normalize
    
    # ALL 23 DATASETS (ordered by length - longest first)
    dataset_mappings = {
        'ielts_writing_task_2_dataset': 'Ielts_Writing_Task_2_Dataset',
    
    }
    
    for pattern, dataset_name in dataset_mappings.items():
        if pattern in name:
            return 'D_' + dataset_name
    
    return None

def evaluate_single_csv(csv_path):
    """Evaluate a single CSV file"""
    try:
        filename = os.path.basename(csv_path)
        print(f"\n📋 {filename}")
        
        df = pd.read_csv(csv_path)
        dataset_name = extract_dataset_name_from_filename(filename)
        
        if not dataset_name:
            print(f"   ⚠ Could not extract dataset name")
            return None
        
        result = engine.evaluate_submission(dataset_name, df)
        
        if result['status'] != 'success':
            print(f"   ❌ {result.get('error')}")
            return None
        
        metrics = result.get('metrics', {})
        
        # Remove D_ prefix for display
        clean_name = dataset_name.replace('D_', '') if dataset_name.startswith('D_') else dataset_name
        print(f"   Dataset: {clean_name}")
        
        qwk = metrics.get('quadratic_weighted_kappa', 'N/A')
        pearson = metrics.get('pearson_correlation', 'N/A')
        mae = metrics.get('mean_absolute_error', 'N/A')
        mae_pct = metrics.get('mae_percentage', 'N/A')
        
        # Safe formatting that handles strings
        qwk_str = f"{float(qwk):.3f}" if isinstance(qwk, (int, float)) else str(qwk)
        pearson_str = f"{float(pearson):.3f}" if isinstance(pearson, (int, float)) else str(pearson)
        mae_str = f"{float(mae):.3f}" if isinstance(mae, (int, float)) else str(mae)
        mae_pct_str = f"{float(mae_pct):.1f}%" if isinstance(mae_pct, (int, float)) else str(mae_pct)
        
        print(f"   QWK: {qwk_str} | Pearson: {pearson_str} | MAE: {mae_str} | MAE%: {mae_pct_str}")
        
        return {
            'filename': filename,
            'dataset_name': dataset_name,
            'metrics': metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

def batch_evaluate_folder(folder_path):
    """Evaluate all CSV files in folder"""
    print("\n" + "="*80)
    print("BATCH EVALUATION - ALL DATASET PREDICTIONS")
    print("="*80)
    print(f"Folder: {folder_path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    csv_files = sorted(list(Path(folder_path).glob('*.csv')))
    
    if not csv_files:
        print("❌ No CSV files found")
        return None
    
    print(f"Found {len(csv_files)} CSV files to evaluate\n")
    
    all_results = []
    for csv_file in csv_files:
        result = evaluate_single_csv(str(csv_file))
        if result and result['success']:
            all_results.append(result)
    
    if not all_results:
        print("\n❌ No successful evaluations")
        return None
    
    # Aggregate metrics
    print("\n" + "="*80)
    print("DETAILED RESULTS BY DATASET")
    print("="*80 + "\n")
    
    for result in all_results:
        m = result['metrics']
        
        # Remove D_ prefix for cleaner display
        clean_name = result['dataset_name'].replace('D_', '') if result['dataset_name'].startswith('D_') else result['dataset_name']
        print(f"{clean_name}:")
        
        # Safe conversion to float with error handling
        def safe_format(val, decimals=4):
            try:
                return f"{float(val):.{decimals}f}"
            except (ValueError, TypeError):
                return str(val)
        
        print(f"  QWK:      {safe_format(m.get('quadratic_weighted_kappa', 'N/A'))}")
        print(f"  Pearson:  {safe_format(m.get('pearson_correlation', 'N/A'))}")
        print(f"  F1:       {safe_format(m.get('f1_score', 'N/A'))}")
        print(f"  Accuracy: {safe_format(m.get('accuracy', 'N/A'))}")
        print(f"  Precision:{safe_format(m.get('precision', 'N/A'))}")
        print(f"  Recall:   {safe_format(m.get('recall', 'N/A'))}")
        print(f"  MAE:      {safe_format(m.get('mean_absolute_error', 'N/A'))}")
        mae_pct = m.get('mae_percentage', 'N/A')
        mae_pct_str = f"{safe_format(mae_pct, 2)}%" if mae_pct != 'N/A' else 'N/A'
        print(f"  MAE%:     {mae_pct_str}")
        print(f"  RMSE:     {safe_format(m.get('root_mean_squared_error', 'N/A'))}")
        print()
    
    # STEP 1: Calculate combined statistics FIRST
    print("="*80)
    print("COMBINED STATISTICS (Across All Datasets)")
    print("="*80 + "\n")
    
    metric_keys = [
        'quadratic_weighted_kappa', 'pearson_correlation', 'mean_absolute_error',
        'mean_squared_error', 'root_mean_squared_error', 'f1_score',
        'precision', 'recall', 'accuracy', 'mae_percentage'
    ]
    
    combined = {}
    for key in metric_keys:
        values = [r['metrics'].get(key) for r in all_results 
                  if r['metrics'].get(key) is not None]
        if values:
            combined[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
            metric_name = key.upper().replace('_', ' ')
            print(f"{metric_name}:")
            print(f"  Mean: {combined[key]['mean']:.4f}")
            print(f"  Std:  {combined[key]['std']:.4f}")
            print(f"  Min:  {combined[key]['min']:.4f}")
            print(f"  Max:  {combined[key]['max']:.4f}\n")
    
    # STEP 2: Prepare Excel data
    excel_data = []
    for result in all_results:
        m = result['metrics']
        clean_dataset_name = result['dataset_name'].replace('D_', '') if result['dataset_name'].startswith('D_') else result['dataset_name']
        excel_data.append({
            'Dataset': clean_dataset_name,
            'Filename': result['filename'],
            'Avg QWK': m.get('quadratic_weighted_kappa', 'N/A'),
            'Avg Pearson': m.get('pearson_correlation', 'N/A'),
            'Avg F1': m.get('f1_score', 'N/A'),
            'Avg Precision': m.get('precision', 'N/A'),
            'Avg Recall': m.get('recall', 'N/A'),
            'Avg Accuracy': m.get('accuracy', 'N/A'),
            'Avg MAE': m.get('mean_absolute_error', 'N/A'),
            'Avg RMSE': m.get('root_mean_squared_error', 'N/A'),
            'MAE %': m.get('mae_percentage', 'N/A')
        })
    
    # STEP 3: Define timestamp ONCE for both files
    timestamp = int(datetime.now().timestamp())
    base_path = "/home/ts1506.UNT/Desktop/Work/besisr-benchmark-site/mllm_evaluation/lama_exp/inductive_llama_predictions_csv/inductive_llama_predictions_csv/"
    
    # STEP 4: Save Excel file
    excel_file = f"{base_path}batch_evaluation_results_{timestamp}.xlsx"
    try:
        excel_df = pd.DataFrame(excel_data)
        excel_df.to_excel(excel_file, index=False, sheet_name='Results')
        print(f"✅ Excel file created: {excel_file}\n")
    except Exception as e:
        print(f"⚠ Could not create Excel file: {e}\n")
    
    # STEP 5: Save JSON file
    json_file = f"{base_path}batch_evaluation_results_{timestamp}.json"
    try:
        json_output = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_datasets': len(all_results),
                'folder_path': folder_path,
                'successful_evaluations': len(all_results),
                'timestamp': timestamp
            },
            'individual_results': [
                {
                    'dataset': r['dataset_name'].replace('D_', '') if r['dataset_name'].startswith('D_') else r['dataset_name'],
                    'filename': r['filename'],
                    'metrics': r['metrics']
                }
                for r in all_results
            ],
            'combined_statistics': combined,
            'excel_data': excel_data
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"✅ JSON file created: {json_file}\n")
    except Exception as e:
        print(f"⚠ Could not create JSON file: {e}\n")
    
    # STEP 6: Return summary
    return {
        'successful': len(all_results),
        'total_files': len(csv_files),
        'excel_file': excel_file,
        'json_file': json_file
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # Default folder path - change this to your actual folder
        folder_path = "/home/ts1506.UNT/Desktop/Work/besisr-benchmark-site/mllm_evaluation/lama_exp/inductive_llama_predictions_csv/inductive_llama_predictions_csv/"
           
        # Or prompt user for input
        #user_input = input(f"Enter folder path with CSV files (default: {folder_path}): ").strip()
        #if user_input:
            #folder_path = user_input
    
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        print(f"Please create the folder or provide a valid path")
        sys.exit(1)
    
    results = batch_evaluate_folder(folder_path)
    
    if results:
        print(f"\n✅ Batch evaluation complete!")
        print(f"   Evaluated: {results['successful']}/{results['total_files']} datasets")
        print(f"   Excel: {results['excel_file']}")
        print(f"   JSON: {results['json_file']}")
    else:
        print("\n❌ Batch evaluation failed")
        sys.exit(1)