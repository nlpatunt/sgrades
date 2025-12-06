#!/usr/bin/env python3
"""
Calculate standard deviation across 3 runs from _FULL.csv files
Handles both numeric and categorical predictions
Calculates OVERALL std across all datasets combined
Adds OVERALL AVERAGE row to the summary
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import re

def load_csv_safely(csv_path):
    """Load CSV with multiple encoding fallbacks"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(
                csv_path, 
                encoding=encoding,
                on_bad_lines='skip',
                encoding_errors='replace',
                engine='python'
            )
            return df
        except:
            continue
    
    return None

def extract_dataset_from_filename(filename):
    """Extract dataset name from filename"""
    name = filename.replace('.csv', '')
    
    # Handle D_ format
    if '_D_' in name:
        parts = name.split('_D_')[1]
        # Remove _3call_FULL, _3call, _FULL, etc
        dataset = re.sub(r'_(3call|FULL).*$', '', parts, flags=re.IGNORECASE)
        return f'D_{dataset}'
    
    return None

def is_numeric_column(series):
    """Check if a series contains numeric values"""
    try:
        pd.to_numeric(series, errors='raise')
        return True
    except (ValueError, TypeError):
        return False

def calculate_categorical_consistency(row):
    """
    Calculate agreement rate for categorical predictions
    Returns proportion of predictions that match the mode
    """
    values = [row['prediction_1'], row['prediction_2'], row['prediction_3']]
    # Remove None/NaN values
    values = [v for v in values if pd.notna(v)]
    
    if len(values) == 0:
        return np.nan
    
    # Find most common prediction
    from collections import Counter
    counts = Counter(values)
    most_common_count = counts.most_common(1)[0][1]
    
    # Agreement rate
    return most_common_count / len(values)

def calculate_std_from_full_files(full_files_folder, output_folder):
    """
    Calculate standard deviation/consistency across 3 runs from files containing 'FULL' in name
    
    Expected columns in FULL files:
    - essay_id (or id)
    - prediction_1
    - prediction_2  
    - prediction_3
    - final_prediction (average or mode)
    """
    print("\n" + "="*80)
    print("CALCULATING STANDARD DEVIATION/CONSISTENCY ACROSS 3 RUNS")
    print("="*80)
    print(f"Input folder: {full_files_folder}")
    print(f"Output folder: {output_folder}\n")
    
    # Get ALL CSV files first
    all_csv_files = sorted(list(Path(full_files_folder).glob('*.csv')))
    print(f"Total CSV files in folder: {len(all_csv_files)}")
    
    # Filter to only files with 'FULL' in the name (case-insensitive)
    full_files = [f for f in all_csv_files if 'FULL' in f.name or 'full' in f.name]
    
    # Show what was filtered out
    non_full_files = [f for f in all_csv_files if 'FULL' not in f.name and 'full' not in f.name]
    
    if non_full_files:
        print(f"Files WITHOUT 'FULL' in name (will be skipped): {len(non_full_files)}")
        for f in non_full_files[:5]:  # Show first 5 as examples
            print(f"   ⏭️  {f.name}")
        if len(non_full_files) > 5:
            print(f"   ... and {len(non_full_files) - 5} more")
        print()
    
    if not full_files:
        print("❌ No files with 'FULL' in name found")
        return None
    
    print(f"✅ Files WITH 'FULL' in name (will be processed): {len(full_files)}")
    for f in full_files:
        print(f"   📄 {f.name}")
    print()
    
    results = []
    detailed_stats = []
    
    # Collect ALL std values across all numeric datasets
    all_std_values = []
    all_cv_values = []
    
    # Collect ALL agreement rates across all categorical datasets
    all_agreement_rates = []
    
    for full_file in full_files:
        print(f"\n📋 Processing: {full_file.name}")
        
        # Extract dataset name
        dataset = extract_dataset_from_filename(full_file.name)
        if not dataset:
            print(f"   ⚠️  Could not extract dataset name")
            continue
        
        clean_dataset = dataset.replace('D_', '')
        print(f"   📊 Dataset: {clean_dataset}")
        
        # Load file
        df = load_csv_safely(str(full_file))
        if df is None:
            print(f"   ❌ Could not load file")
            continue
        
        print(f"   ✅ Loaded {len(df)} predictions")
        
        # Check for required columns
        id_col = None
        for col in ['essay_id', 'id', 'Essay_id', 'ID', 'essay_Id', 'Id']:
            if col in df.columns:
                id_col = col
                break
        
        if id_col is None:
            print(f"   ⚠️  No ID column found")
            id_col = df.columns[0]  # Use first column as ID
            print(f"   ℹ️  Using '{id_col}' as ID column")
        
        # Check for prediction columns
        pred_cols = ['prediction_1', 'prediction_2', 'prediction_3']
        missing_cols = [col for col in pred_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ❌ Missing columns: {missing_cols}")
            print(f"   Available columns: {df.columns.tolist()}")
            continue
        
        # Check if predictions are numeric or categorical
        is_numeric = is_numeric_column(df['prediction_1'])
        
        if is_numeric:
            print(f"   ℹ️  Prediction type: NUMERIC")
            
            # Convert to numeric
            for col in pred_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate standard deviation across the 3 predictions
            df['std_across_runs'] = df[pred_cols].std(axis=1)
            
            # Collect all std values for overall calculation
            all_std_values.extend(df['std_across_runs'].dropna().tolist())
            
            # Calculate statistics
            mean_std = df['std_across_runs'].mean()
            median_std = df['std_across_runs'].median()
            max_std = df['std_across_runs'].max()
            min_std = df['std_across_runs'].min()
            
            # Count predictions with high variance
            high_variance_count = (df['std_across_runs'] > 1.0).sum()
            high_variance_pct = (high_variance_count / len(df)) * 100
            
            # Calculate coefficient of variation
            df['mean_prediction'] = df[pred_cols].mean(axis=1)
            df['cv'] = df['std_across_runs'] / df['mean_prediction'].abs()
            df['cv'] = df['cv'].replace([np.inf, -np.inf], np.nan)
            mean_cv = df['cv'].mean()
            
            # Collect all CV values for overall calculation
            all_cv_values.extend(df['cv'].dropna().tolist())
            
            print(f"   📊 Mean std across runs: {mean_std:.4f}")
            print(f"   📊 Median std across runs: {median_std:.4f}")
            print(f"   📊 Max std across runs: {max_std:.4f}")
            print(f"   📊 Min std across runs: {min_std:.4f}")
            print(f"   📊 Predictions with std > 1.0: {high_variance_count} ({high_variance_pct:.1f}%)")
            print(f"   📊 Mean coefficient of variation: {mean_cv:.4f}")
            
            # Store summary results
            results.append({
                'dataset': clean_dataset,
                'filename': full_file.name,
                'prediction_type': 'numeric',
                'n_predictions': len(df),
                'mean_std': float(mean_std),
                'median_std': float(median_std),
                'max_std': float(max_std),
                'min_std': float(min_std),
                'high_variance_count': int(high_variance_count),
                'high_variance_pct': float(high_variance_pct),
                'mean_cv': float(mean_cv),
                'perfect_agreement_pct': None,
                'mean_agreement': None
            })
            
            # Store detailed per-prediction data
            if 'final_prediction' not in df.columns:
                df['final_prediction'] = df[pred_cols].mean(axis=1)
            
            detailed_df = df[[id_col, 'prediction_1', 'prediction_2', 'prediction_3', 
                             'final_prediction', 'std_across_runs', 'mean_prediction', 'cv']].copy()
            detailed_df['dataset'] = clean_dataset
            detailed_stats.append(detailed_df)
            
        else:
            print(f"   ℹ️  Prediction type: CATEGORICAL")
            
            # Calculate agreement rate for each prediction
            df['agreement_rate'] = df.apply(calculate_categorical_consistency, axis=1)
            
            # Collect all agreement rates for overall calculation
            all_agreement_rates.extend(df['agreement_rate'].dropna().tolist())
            
            # Calculate statistics
            mean_agreement = df['agreement_rate'].mean()
            perfect_agreement_count = (df['agreement_rate'] == 1.0).sum()
            perfect_agreement_pct = (perfect_agreement_count / len(df)) * 100
            partial_agreement_count = ((df['agreement_rate'] >= 0.67) & (df['agreement_rate'] < 1.0)).sum()
            partial_agreement_pct = (partial_agreement_count / len(df)) * 100
            no_agreement_count = (df['agreement_rate'] < 0.67).sum()
            no_agreement_pct = (no_agreement_count / len(df)) * 100
            
            print(f"   📊 Mean agreement rate: {mean_agreement:.4f}")
            print(f"   📊 Perfect agreement (3/3): {perfect_agreement_count} ({perfect_agreement_pct:.1f}%)")
            print(f"   📊 Partial agreement (2/3): {partial_agreement_count} ({partial_agreement_pct:.1f}%)")
            print(f"   📊 No agreement (0/3 or 1/3): {no_agreement_count} ({no_agreement_pct:.1f}%)")
            
            # Store summary results
            results.append({
                'dataset': clean_dataset,
                'filename': full_file.name,
                'prediction_type': 'categorical',
                'n_predictions': len(df),
                'mean_std': None,
                'median_std': None,
                'max_std': None,
                'min_std': None,
                'high_variance_count': None,
                'high_variance_pct': None,
                'mean_cv': None,
                'perfect_agreement_pct': float(perfect_agreement_pct),
                'mean_agreement': float(mean_agreement)
            })
            
            # Store detailed per-prediction data
            if 'final_prediction' not in df.columns:
                # For categorical, final_prediction is the mode
                df['final_prediction'] = df[pred_cols].mode(axis=1)[0]
            
            detailed_df = df[[id_col, 'prediction_1', 'prediction_2', 'prediction_3', 
                             'final_prediction', 'agreement_rate']].copy()
            detailed_df['dataset'] = clean_dataset
            detailed_stats.append(detailed_df)
    
    # Save results
    if results:
        print("\n" + "="*80)
        print("SUMMARY OF ALL DATASETS")
        print("="*80 + "\n")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Calculate overall statistics separately for numeric and categorical
        numeric_results = summary_df[summary_df['prediction_type'] == 'numeric']
        categorical_results = summary_df[summary_df['prediction_type'] == 'categorical']
        
        # Calculate OVERALL statistics across all predictions
        overall_numeric_stats = {}
        overall_categorical_stats = {}
        
        if len(numeric_results) > 0:
            overall_std_mean = np.mean(all_std_values)
            overall_std_median = np.median(all_std_values)
            overall_std_max = np.max(all_std_values)
            overall_std_min = np.min(all_std_values)
            overall_high_variance_count = sum(1 for x in all_std_values if x > 1.0)
            overall_high_variance_pct = (overall_high_variance_count / len(all_std_values)) * 100
            overall_cv_mean = np.mean(all_cv_values) if all_cv_values else 0
            
            overall_numeric_stats = {
                'dataset': '🏆 OVERALL AVERAGE (Numeric)',
                'filename': f'{len(numeric_results)} datasets combined',
                'prediction_type': 'numeric',
                'n_predictions': len(all_std_values),
                'mean_std': float(overall_std_mean),
                'median_std': float(overall_std_median),
                'max_std': float(overall_std_max),
                'min_std': float(overall_std_min),
                'high_variance_count': int(overall_high_variance_count),
                'high_variance_pct': float(overall_high_variance_pct),
                'mean_cv': float(overall_cv_mean),
                'perfect_agreement_pct': None,
                'mean_agreement': None
            }
        
        if len(categorical_results) > 0:
            overall_agreement_mean = np.mean(all_agreement_rates)
            overall_perfect_agreement_count = sum(1 for x in all_agreement_rates if x == 1.0)
            overall_perfect_agreement_pct = (overall_perfect_agreement_count / len(all_agreement_rates)) * 100
            
            overall_categorical_stats = {
                'dataset': '🏆 OVERALL AVERAGE (Categorical)',
                'filename': f'{len(categorical_results)} datasets combined',
                'prediction_type': 'categorical',
                'n_predictions': len(all_agreement_rates),
                'mean_std': None,
                'median_std': None,
                'max_std': None,
                'min_std': None,
                'high_variance_count': None,
                'high_variance_pct': None,
                'mean_cv': None,
                'perfect_agreement_pct': float(overall_perfect_agreement_pct),
                'mean_agreement': float(overall_agreement_mean)
            }
        
        # Add overall rows to summary DataFrame
        if overall_numeric_stats:
            summary_df = pd.concat([summary_df, pd.DataFrame([overall_numeric_stats])], ignore_index=True)
        if overall_categorical_stats:
            summary_df = pd.concat([summary_df, pd.DataFrame([overall_categorical_stats])], ignore_index=True)
        
        # Print summary table (now includes overall rows)
        print(summary_df.to_string(index=False))
        print()
        
        print("="*80)
        print("OVERALL STATISTICS")
        print("="*80 + "\n")
        
        if len(numeric_results) > 0:
            print(f"NUMERIC DATASETS ({len(numeric_results)} datasets, {len(all_std_values)} total predictions):")
            print(f"  📈 OVERALL std (all predictions combined):")
            print(f"     Mean: {overall_std_mean:.4f}")
            print(f"     Median: {overall_std_median:.4f}")
            print(f"     Max: {overall_std_max:.4f}")
            print(f"     Min: {overall_std_min:.4f}")
            print(f"     High variance (>1.0): {overall_high_variance_count} ({overall_high_variance_pct:.1f}%)")
            print(f"     Overall CV: {overall_cv_mean:.4f}")
            print()
            print(f"  📊 Per-dataset averages:")
            print(f"     Average mean std: {numeric_results['mean_std'].mean():.4f}")
            print(f"     Average median std: {numeric_results['median_std'].mean():.4f}")
            print(f"     Max std observed: {numeric_results['max_std'].max():.4f}")
            print(f"     Average high variance %: {numeric_results['high_variance_pct'].mean():.1f}%")
            print(f"     Average CV: {numeric_results['mean_cv'].mean():.4f}")
            print()
        
        if len(categorical_results) > 0:
            print(f"CATEGORICAL DATASETS ({len(categorical_results)} datasets, {len(all_agreement_rates)} total predictions):")
            print(f"  📈 OVERALL agreement (all predictions combined):")
            print(f"     Mean agreement rate: {overall_agreement_mean:.4f}")
            print(f"     Perfect agreement (3/3): {overall_perfect_agreement_count} ({overall_perfect_agreement_pct:.1f}%)")
            print()
            print(f"  📊 Per-dataset averages:")
            print(f"     Average mean agreement: {categorical_results['mean_agreement'].mean():.4f}")
            print(f"     Average perfect agreement %: {categorical_results['perfect_agreement_pct'].mean():.1f}%")
            print()
        
        print(f"GRAND TOTAL: {len(results)} datasets analyzed")
        
        # Save to files
        timestamp = int(datetime.now().timestamp())
        os.makedirs(output_folder, exist_ok=True)
        
        # Save summary Excel (now includes overall rows)
        summary_excel = Path(output_folder) / f"std_summary_{timestamp}.xlsx"
        
        with pd.ExcelWriter(summary_excel, engine='openpyxl') as writer:
            # Summary by dataset (includes overall rows at the end)
            summary_df.to_excel(writer, sheet_name='Per-Dataset Summary', index=False)
            
            # Overall statistics sheet
            overall_stats_data = []
            
            if len(numeric_results) > 0:
                overall_stats_data.append({
                    'Category': 'Numeric Datasets',
                    'Metric': 'Number of datasets',
                    'Value': len(numeric_results)
                })
                overall_stats_data.append({
                    'Category': 'Numeric Datasets',
                    'Metric': 'Total predictions',
                    'Value': len(all_std_values)
                })
                overall_stats_data.append({
                    'Category': 'Numeric Datasets',
                    'Metric': 'OVERALL mean std (all predictions)',
                    'Value': overall_std_mean
                })
                overall_stats_data.append({
                    'Category': 'Numeric Datasets',
                    'Metric': 'OVERALL median std (all predictions)',
                    'Value': overall_std_median
                })
                overall_stats_data.append({
                    'Category': 'Numeric Datasets',
                    'Metric': 'OVERALL max std',
                    'Value': overall_std_max
                })
                overall_stats_data.append({
                    'Category': 'Numeric Datasets',
                    'Metric': 'OVERALL high variance % (>1.0)',
                    'Value': overall_high_variance_pct
                })
                overall_stats_data.append({
                    'Category': 'Numeric Datasets',
                    'Metric': 'OVERALL mean CV',
                    'Value': overall_cv_mean
                })
            
            if len(categorical_results) > 0:
                overall_stats_data.append({
                    'Category': 'Categorical Datasets',
                    'Metric': 'Number of datasets',
                    'Value': len(categorical_results)
                })
                overall_stats_data.append({
                    'Category': 'Categorical Datasets',
                    'Metric': 'Total predictions',
                    'Value': len(all_agreement_rates)
                })
                overall_stats_data.append({
                    'Category': 'Categorical Datasets',
                    'Metric': 'OVERALL mean agreement (all predictions)',
                    'Value': overall_agreement_mean
                })
                overall_stats_data.append({
                    'Category': 'Categorical Datasets',
                    'Metric': 'OVERALL perfect agreement %',
                    'Value': overall_perfect_agreement_pct
                })
            
            overall_stats_df = pd.DataFrame(overall_stats_data)
            overall_stats_df.to_excel(writer, sheet_name='Overall Statistics', index=False)
        
        print(f"\n✅ Summary Excel saved: {summary_excel}")
        print(f"   ✨ Includes overall average rows at the bottom")
        
        # Save detailed Excel with multiple sheets
        detailed_excel = Path(output_folder) / f"std_detailed_{timestamp}.xlsx"
        with pd.ExcelWriter(detailed_excel, engine='openpyxl') as writer:
            # Write summary sheet (includes overall rows)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Write each dataset to its own sheet (limit sheet name to 31 chars)
            for detail_df in detailed_stats:
                dataset_name = detail_df['dataset'].iloc[0]
                sheet_name = dataset_name[:31]  # Excel sheet name limit
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Detailed Excel saved: {detailed_excel}")
        print(f"   (Contains {len(detailed_stats) + 1} sheets: 1 summary + {len(detailed_stats)} datasets)")
        
        # Save JSON
        json_file = Path(output_folder) / f"std_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'date': datetime.now().isoformat(),
                    'input_folder': str(full_files_folder),
                    'n_datasets': len(results),
                    'n_numeric': len(numeric_results),
                    'n_categorical': len(categorical_results)
                },
                'summary_by_dataset': results,
                'overall_statistics': {
                    'numeric': {
                        'n_datasets': len(numeric_results),
                        'total_predictions': len(all_std_values),
                        'overall_mean_std': float(overall_std_mean) if len(numeric_results) > 0 else None,
                        'overall_median_std': float(overall_std_median) if len(numeric_results) > 0 else None,
                        'overall_max_std': float(overall_std_max) if len(numeric_results) > 0 else None,
                        'overall_high_variance_pct': float(overall_high_variance_pct) if len(numeric_results) > 0 else None,
                        'overall_mean_cv': float(overall_cv_mean) if len(numeric_results) > 0 else None,
                        'per_dataset_avg_mean_std': float(numeric_results['mean_std'].mean()) if len(numeric_results) > 0 else None,
                        'per_dataset_avg_median_std': float(numeric_results['median_std'].mean()) if len(numeric_results) > 0 else None,
                    },
                    'categorical': {
                        'n_datasets': len(categorical_results),
                        'total_predictions': len(all_agreement_rates),
                        'overall_mean_agreement': float(overall_agreement_mean) if len(categorical_results) > 0 else None,
                        'overall_perfect_agreement_pct': float(overall_perfect_agreement_pct) if len(categorical_results) > 0 else None,
                        'per_dataset_avg_mean_agreement': float(categorical_results['mean_agreement'].mean()) if len(categorical_results) > 0 else None,
                    }
                }
            }, f, indent=2)
        print(f"✅ JSON saved: {json_file}\n")
        
        return summary_df
    else:
        print("\n❌ No results to save")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        full_files_folder = sys.argv[1]
    else:
        # Default path - update this
        full_files_folder = "/home/ts1506.UNT/Desktop/Work/besisr-benchmark-site/mllm_evaluation/gemini_flash/inductive_deductive_3call_predictions/"
    
    if len(sys.argv) > 2:
        output_folder = sys.argv[2]
    else:
        output_folder = "/home/ts1506.UNT/Desktop/Work/besisr-benchmark-site/mllm_evaluation/gemini_flash/inductive_deductive_3call_predictions/std_analysis/"
    
    if not os.path.isdir(full_files_folder):
        print(f"❌ Folder not found: {full_files_folder}")
        sys.exit(1)
    
    # Run analysis
    results = calculate_std_from_full_files(full_files_folder, output_folder)
    
    if results is not None:
        print("✅ Standard deviation/consistency analysis complete!")
        print(f"   Analyzed {len(results)} datasets")
    else:
        print("❌ Analysis failed")
        sys.exit(1)