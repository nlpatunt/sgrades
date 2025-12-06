#!/usr/bin/env python3
"""
Evaluate prediction files against gold standards from HuggingFace
Shows sample predictions vs gold for verification
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import evaluation engine
try:
    from app.api.routes.output_submissions import RealEvaluationEngine
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

engine = RealEvaluationEngine()

def extract_dataset_name(filename):
    """
    Extract dataset name from: llama-4-scout_D_ASAP2_3call.csv -> D_ASAP2
    """
    name = filename.replace('.csv', '')
    
    if '_D_' in name:
        # Split on _D_ and take everything after it
        parts = name.split('_D_')[1]
        # Remove _3call, _FULL, etc.
        dataset = re.sub(r'_(3call|1call|FULL).*$', '', parts, flags=re.IGNORECASE)
        return 'D_' + dataset
    
    return None

def normalize_scores(df):
    """Normalize scores: 2, 2.0, 2.00 -> 2.0"""
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce').round(4)
    return df

def print_sample_comparison(pred_df, gold_df, dataset_name):
    """Print 2 sample datapoints: predictions vs gold side by side"""
    print("\n   📊 Sample Predictions vs Gold Standards:")
    print("   " + "-"*70)
    
    # Find common ID column
    id_col = None
    for col in ['id', 'ID', 'essay_id', 'Id']:
        if col in pred_df.columns and col in gold_df.columns:
            id_col = col
            break
    
    if id_col is None:
        print("   ⚠️  Could not find common ID column")
        return
    
    # Find score column
    score_col_pred = None
    score_col_gold = None
    
    for col in ['score', 'band_score', 'domain1_score']:
        if col in pred_df.columns:
            score_col_pred = col
        if col in gold_df.columns:
            score_col_gold = col
    
    if not score_col_pred or not score_col_gold:
        print("   ⚠️  Could not find score columns")
        return
    
    # Get first 2 matching IDs
    pred_ids = set(pred_df[id_col].astype(str))
    gold_ids = set(gold_df[id_col].astype(str))
    common_ids = list(pred_ids & gold_ids)[:2]
    
    if not common_ids:
        print("   ⚠️  No matching IDs found")
        return
    
    print(f"   {'ID':<10} | {'Your Prediction':<20} | {'Gold Standard':<20}")
    print("   " + "-"*70)
    
    for sample_id in common_ids:
        pred_row = pred_df[pred_df[id_col].astype(str) == sample_id]
        gold_row = gold_df[gold_df[id_col].astype(str) == sample_id]
        
        if not pred_row.empty and not gold_row.empty:
            pred_score = pred_row[score_col_pred].values[0]
            gold_score = gold_row[score_col_gold].values[0]
            print(f"   {sample_id:<10} | {pred_score:<20} | {gold_score:<20}")
    
    print("   " + "-"*70)

def clean_dataframe(df):
    """
    Clean dataframe by:
    1. Dropping unnamed columns (Unnamed: 0, Unnamed: 1, etc.)
    2. Normalizing scores
    """
    # Drop all unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col or 'unnamed' in col]
    if unnamed_cols:
        print(f"   🧹 Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    
    # Normalize scores
    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce').round(4)
    
    return df

def evaluate_file(csv_path):
    """Evaluate a single prediction file"""
    try:
        filename = os.path.basename(csv_path)
        print(f"\n📋 {filename}")
        
        # Read CSV
        pred_df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        pred_df = clean_dataframe(pred_df)
        
        # Extract dataset name
        dataset_name = extract_dataset_name(filename)
        if not dataset_name:
            print(f"   ❌ Could not extract dataset name")
            return None
        
        clean_name = dataset_name.replace('D_', '')
        print(f"   📊 Dataset: {clean_name}")
        
        # Get gold standards (we'll intercept this)
        # First, let's try to load gold standards directly
        try:
            from datasets import load_dataset
            # Remove D_ prefix for HuggingFace
            hf_dataset_name = dataset_name.replace('D_', '')
            hf_repo = f"nlpatunt/{hf_dataset_name}"
            print(f"   📥 Loading gold standards from: {hf_repo}")
            
            gold_dataset = load_dataset(hf_repo, split='test')
            gold_df = gold_dataset.to_pandas()
            
            # Print sample comparison
            print_sample_comparison(pred_df, gold_df, dataset_name)
            
        except Exception as e:
            print(f"   ℹ️  Could not load gold for comparison preview: {str(e)}")
        
        # Evaluate against gold standards using engine
        result = engine.evaluate_submission(dataset_name, pred_df)
        
        if result['status'] != 'success':
            print(f"   ❌ {result.get('error')}")
            return None
        
        metrics = result.get('metrics', {})
        
        # Print key metrics
        qwk = metrics.get('quadratic_weighted_kappa', 'N/A')
        pearson = metrics.get('pearson_correlation', 'N/A')
        f1 = metrics.get('f1_score', 'N/A')
        mae = metrics.get('mean_absolute_error', 'N/A')
        
        def format_metric(val):
            if val == 'N/A' or val is None:
                return 'N/A'
            return f"{float(val):.4f}"
        
        print(f"   ✅ QWK: {format_metric(qwk)} | Pearson: {format_metric(pearson)} | F1: {format_metric(f1)} | MAE: {format_metric(mae)}")
        
        return {
            'filename': filename,
            'dataset': clean_name,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_folder(folder_path):
    """Evaluate all prediction files in folder"""
    print("\n" + "="*80)
    print("BATCH EVALUATION - COMPARING PREDICTIONS WITH GOLD STANDARDS")
    print("="*80)
    print(f"Folder: {folder_path}\n")
    
    # Get all CSV files
    all_files = sorted(list(Path(folder_path).glob('*.csv')))
    
    # Filter out _FULL files
    csv_files = [f for f in all_files if '_FULL' not in f.name]
    full_files = [f for f in all_files if '_FULL' in f.name]
    
    print(f"Found {len(all_files)} total CSV files")
    if full_files:
        print(f"   ⏭️  Skipping {len(full_files)} _FULL.csv files")
    print(f"   ✅ Evaluating {len(csv_files)} prediction files\n")
    
    if not csv_files:
        print("❌ No prediction files to evaluate")
        return None
    
    # Evaluate each file
    results = []
    for csv_file in csv_files:
        result = evaluate_file(str(csv_file))
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ No successful evaluations")
        return None
    
    # Print summary
    print("\n" + "="*80)
    print(f"SUMMARY: {len(results)}/{len(csv_files)} files evaluated successfully")
    print("="*80 + "\n")
    
    # Create Excel output
    excel_data = []
    for r in results:
        m = r['metrics']
        excel_data.append({
            'Dataset': r['dataset'],
            'Filename': r['filename'],
            'QWK': m.get('quadratic_weighted_kappa', 'N/A'),
            'Pearson': m.get('pearson_correlation', 'N/A'),
            'F1': m.get('f1_score', 'N/A'),
            'Precision': m.get('precision', 'N/A'),
            'Recall': m.get('recall', 'N/A'),
            'Accuracy': m.get('accuracy', 'N/A'),
            'MAE': m.get('mean_absolute_error', 'N/A'),
            'RMSE': m.get('root_mean_squared_error', 'N/A'),
            'MAE_pct': m.get('mae_percentage', 'N/A')
        })
    
    # Calculate overall average
    metric_keys = ['quadratic_weighted_kappa', 'pearson_correlation', 'f1_score',
                   'precision', 'recall', 'accuracy', 'mean_absolute_error',
                   'root_mean_squared_error', 'mae_percentage']
    
    avg_row = {'Dataset': '🏆 OVERALL AVERAGE', 'Filename': f'({len(results)} datasets)'}
    for key in metric_keys:
        values = [r['metrics'].get(key) for r in results 
                 if r['metrics'].get(key) is not None and r['metrics'].get(key) != 'N/A']
        if values:
            col_name = {
                'quadratic_weighted_kappa': 'QWK',
                'pearson_correlation': 'Pearson',
                'f1_score': 'F1',
                'mean_absolute_error': 'MAE',
                'root_mean_squared_error': 'RMSE',
                'mae_percentage': 'MAE_pct',
                'precision': 'Precision',
                'recall': 'Recall',
                'accuracy': 'Accuracy'
            }.get(key, key)
            avg_row[col_name] = float(np.mean(values))
    
    excel_data.append(avg_row)
    
    # Save Excel
    timestamp = int(datetime.now().timestamp())
    folder_name = os.path.basename(folder_path.rstrip('/'))
    output_file = os.path.join(folder_path, f"evaluation_results_{folder_name}_{timestamp}.xlsx")
    
    try:
        df = pd.DataFrame(excel_data)
        df.to_excel(output_file, index=False, sheet_name='Results')
        print(f"✅ Results saved: {output_file}\n")
    except Exception as e:
        print(f"⚠️  Could not save Excel: {e}\n")
        output_file = None
    
    return {
        'results': results,
        'output_file': output_file,
        'total': len(csv_files),
        'successful': len(results)
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # DEFAULT FOLDER PATH - UPDATE THIS
        folder_path = "/home/ts1506.UNT/Desktop/Work/besisr-benchmark-site/mllm_evaluation/gemini_flash/inductive_deductive_3call_predictions/"
    
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        sys.exit(1)
    
    results = evaluate_folder(folder_path)
    
    if results:
        print(f"✅ Evaluation complete!")
        print(f"   Successfully evaluated: {results['successful']}/{results['total']} files")
        if results['output_file']:
            print(f"   Results: {results['output_file']}")
    else:
        print("❌ Evaluation failed")
        sys.exit(1)
