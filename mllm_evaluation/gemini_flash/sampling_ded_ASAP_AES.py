#!/usr/bin/env python3
"""
Multi-Seed Deductive Experiment for ASAP-AES
Loads seed 42 results from file + runs 2 new seeds (123, 456)
Calculates average metrics across all 3 runs
"""
import os
import json
import time
import sys
import re
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datasets import load_dataset
from dataset_ranges import get_score_range_for_dataset

os.environ["HF_TOKEN"] = "REMOVED_KEY"

# ===== CONFIGURATION =====
DATASET = "ASAP-AES"
APPROACH = "deductive"
MODEL_CODE = "google/gemini-2.5-flash"
MODEL_NAME = "gemini-2.5-flash"
NEW_SEEDS = [123, 456]  # Only new seeds (seed 42 loaded from file)

# Output directory
BASE_DIR = "multi_seed_deductive"
OUTPUT_DIR = os.path.join(BASE_DIR, f"{MODEL_NAME}_{DATASET}_{APPROACH}")
CSV_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Column mappings
ASAP_AES_COLUMNS = {
    "id": "essay_id",
    "text": "essay",
    "score": "domain1_score",
    "question": "prompt",
    "essay_set": "essay_set"
}

def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

def download_test_data():
    """Download ASAP-AES test data"""
    print(f"  Downloading test data from {DATASET}...")
    try:
        dataset = load_dataset(f"nlpatunt/{DATASET}", split="test", trust_remote_code=True)
        df = dataset.to_pandas()
        print(f"  ✓ Downloaded: {len(df)} rows from {DATASET}")
        
        return {
            "status": "success", 
            "test_data": df.copy(),
            "ground_truth": df
        }
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"status": "error", "error": str(e)}

def create_deductive_prompt(essay_text: str, question: str, score_range: tuple) -> str:
    """Create deductive reasoning prompt - EXACT format from your original"""
    
    examples = """
GENERAL RULE 1: Complete answers provide required number of details
APPLICATION: Student provides 3 pieces of information as requested
DEDUCTION: Meets requirement → Base score awarded

GENERAL RULE 2: Answers must explain reasoning, not just state results
APPLICATION: Student gives correct number but no explanation
DEDUCTION: Partial credit only (correct result but incomplete reasoning)

GENERAL RULE 3: Scientific answers must correctly apply relevant laws
APPLICATION: Student mentions Coulomb's Law but misapplies effective nuclear charge concept
DEDUCTION: Partial credit (understands some principles but has conceptual errors)
"""
    
    system_prompt = f"""You are an expert scorer using DEDUCTIVE REASONING.

DEDUCTIVE PROCESS:
1. Start with GENERAL scoring rules/criteria
2. Apply rules to THIS SPECIFIC essay
3. Derive score logically from rule application

EXAMPLES OF DEDUCTIVE SCORING:
{examples}

SCORING RANGE: {score_range[0]} to {score_range[1]}
TASK: Essay scoring"""

    user_prompt = f"""Apply deductive reasoning to score:

QUESTION/PROMPT:
{question}

ESSAY TO SCORE:
{essay_text}

Apply general scoring rules to this specific essay and derive the score.

Provide ONLY a single number between {score_range[0]} and {score_range[1]} with no explanation. Just the number."""

    return system_prompt, user_prompt

def validate_prediction(response_text: str, score_range: tuple) -> Dict[str, Any]:
    """Validate response is a valid score"""
    min_score, max_score = score_range
    text = response_text.strip()
    
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    
    if not matches:
        return {'valid': False, 'error': 'No numeric value found', 'response': text}
    
    try:
        predicted_score = float(matches[0])
        
        if predicted_score < min_score or predicted_score > max_score:
            return {
                'valid': False,
                'error': f'Score {predicted_score} outside range [{min_score}, {max_score}]',
                'response': text
            }
        
        return {'valid': True, 'prediction': predicted_score}
    except ValueError:
        return {'valid': False, 'error': f'Could not convert to number', 'response': text}

def get_prediction_with_retry(client, essay_text: str, question: str, 
                              score_range: tuple, max_retries: int = 3) -> Dict[str, Any]:
    """Get prediction with retry logic"""
    for attempt in range(max_retries):
        try:
            system_prompt, user_prompt = create_deductive_prompt(essay_text, question, score_range)
            
            response = client.chat.completions.create(
                model=MODEL_CODE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            validation = validate_prediction(response_text, score_range)
            
            if validation['valid']:
                return {
                    'success': True,
                    'prediction': validation['prediction'],
                    'tokens': tokens_used,
                    'attempts': attempt + 1
                }
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return {
        'success': False,
        'error': f"Validation failed after {max_retries} attempts",
        'attempts': max_retries
    }

def calculate_metrics(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
    """Calculate all metrics"""
    try:
        from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, accuracy_score
        from scipy.stats import pearsonr
        
        # Merge predictions with ground truth
        merged = predictions_df.merge(
            ground_truth_df[[ASAP_AES_COLUMNS["id"], ASAP_AES_COLUMNS["score"]]],
            left_on='id',
            right_on=ASAP_AES_COLUMNS["id"],
            how='inner'
        )
        
        y_true = merged[ASAP_AES_COLUMNS["score"]].values
        y_pred = merged['prediction'].values
        
        # QWK and regression metrics
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        pearson_corr, _ = pearsonr(y_true, y_pred)
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        # Classification metrics (rounded scores)
        y_true_rounded = np.round(y_true).astype(int)
        y_pred_rounded = np.round(y_pred).astype(int)
        
        f1 = f1_score(y_true_rounded, y_pred_rounded, average='weighted', zero_division=0)
        precision = precision_score(y_true_rounded, y_pred_rounded, average='weighted', zero_division=0)
        recall = recall_score(y_true_rounded, y_pred_rounded, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true_rounded, y_pred_rounded)
        
        return {
            'qwk': round(qwk, 15),
            'pearson': round(pearson_corr, 15),
            'f1': round(f1, 15),
            'precision': round(precision, 15),
            'recall': round(recall, 15),
            'accuracy': round(accuracy, 15),
            'mae': round(mae, 15),
            'rmse': round(rmse, 15),
            'num_predictions': len(y_pred)
        }
    except Exception as e:
        print(f"  ✗ Error calculating metrics: {e}")
        return None

def run_single_seed(api_key: str, seed: int, test_data_result: Dict) -> Dict:
    """Run experiment with a single seed"""
    client = get_client(api_key)
    
    print(f"\n{'='*70}")
    print(f"RUNNING SEED {seed}")
    print(f"{'='*70}")
    
    # Create seed-specific output directory
    seed_dir = os.path.join(CSV_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    test_df = test_data_result["test_data"]
    ground_truth_df = test_data_result["ground_truth"]
    
    # Shuffle with this seed (for consistency)
    test_df_shuffled = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"  Processing {len(test_df_shuffled)} essays with seed {seed}")
    
    # Initialize results
    predictions_list = []
    failed_count = 0
    
    # Process each test example
    for i, (_, row) in enumerate(test_df_shuffled.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_df_shuffled)} ({100*i/len(test_df_shuffled):.1f}%)")
        
        essay_id = str(row[ASAP_AES_COLUMNS["id"]])
        essay_text = row[ASAP_AES_COLUMNS["text"]]
        question = row.get(ASAP_AES_COLUMNS["question"], "")
        essay_set = int(row[ASAP_AES_COLUMNS["essay_set"]])
        
        # Get score range for this essay
        score_range = get_score_range_for_dataset(DATASET, essay_set)
        
        result = get_prediction_with_retry(client, essay_text, question, score_range)
        
        if result['success']:
            predictions_list.append({
                'id': essay_id,
                'prediction': result['prediction']
            })
        else:
            failed_count += 1
        
        time.sleep(0.5)  # Rate limiting
    
    print(f"\n✅ Seed {seed} Complete:")
    print(f"   Valid: {len(predictions_list)}/{len(test_df_shuffled)}")
    print(f"   Failed: {failed_count}/{len(test_df_shuffled)}")
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame(predictions_list)
    
    # Save predictions CSV
    csv_filename = f"{MODEL_NAME}_{DATASET}_{APPROACH}_seed{seed}.csv"
    csv_path = os.path.join(seed_dir, csv_filename)
    predictions_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved predictions to {csv_path}")
    
    # Calculate metrics
    print(f"\n📊 Calculating metrics for seed {seed}...")
    metrics = calculate_metrics(predictions_df, ground_truth_df)
    
    if metrics:
        print(f"\n{'='*70}")
        print(f"SEED {seed} RESULTS")
        print(f"{'='*70}")
        print(f"QWK:       {metrics['qwk']:.15f}")
        print(f"Pearson:   {metrics['pearson']:.15f}")
        print(f"F1:        {metrics['f1']:.15f}")
        print(f"Precision: {metrics['precision']:.15f}")
        print(f"Recall:    {metrics['recall']:.15f}")
        print(f"Accuracy:  {metrics['accuracy']:.15f}")
        print(f"MAE:       {metrics['mae']:.15f}")
        print(f"RMSE:      {metrics['rmse']:.15f}")
        print(f"{'='*70}")
        
        # Create seed result object
        seed_result = {
            'seed': seed,
            'model': MODEL_NAME,
            'dataset': DATASET,
            'approach': APPROACH,
            'metrics': metrics,
            'valid_predictions': len(predictions_list),
            'failed_predictions': failed_count,
            'total_essays': len(test_df_shuffled),
            'csv_file': csv_path
        }
        
        # Save individual seed results
        seed_json_path = os.path.join(seed_dir, f"seed_{seed}_results.json")
        with open(seed_json_path, 'w') as f:
            json.dump(seed_result, f, indent=2)
        print(f"  ✓ Saved seed results to {seed_json_path}")
        
        return seed_result
    else:
        print("⚠️  Could not calculate metrics")
        return None

def run_multi_seed_experiment(api_key: str, seed_42_path: str):
    """Main experiment: Load seed 42 + run 2 new seeds"""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("MULTI-SEED DEDUCTIVE EXPERIMENT")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET}")
    print(f"Approach: {APPROACH}")
    print(f"Seed 42: Loading from {seed_42_path}")
    print(f"New seeds: {NEW_SEEDS}")
    print(f"Total: 3 seeds")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    all_results = {
        'experiment': f'{DATASET} {MODEL_NAME} {APPROACH} Multi-Seed',
        'model': MODEL_NAME,
        'dataset': DATASET,
        'approach': APPROACH,
        'seeds': [42] + NEW_SEEDS,
        'start_timestamp': datetime.now().isoformat(),
        'seed_results': []
    }
    
    # STEP 1: Load seed 42 results
    print(f"\n{'='*70}")
    print("LOADING SEED 42 RESULTS FROM FILE")
    print(f"{'='*70}")
    
    seed_42_loaded = False
    try:
        with open(seed_42_path, 'r') as f:
            seed_42_data = json.load(f)
        
        # Extract metrics from seed 42 data
        if 'results' in seed_42_data:
            seed_42_result = {
                'seed': 42,
                'model': MODEL_NAME,
                'dataset': DATASET,
                'approach': APPROACH,
                'metrics': {
                    'qwk': seed_42_data['results']['avg_qwk'],
                    'pearson': seed_42_data['results']['avg_pearson'],
                    'f1': seed_42_data['results']['avg_f1'],
                    'precision': seed_42_data['results']['avg_precision'],
                    'recall': seed_42_data['results']['avg_recall'],
                    'accuracy': seed_42_data['results']['avg_accuracy'],
                    'mae': seed_42_data['results']['avg_mae'],
                    'rmse': seed_42_data['results']['avg_rmse']
                },
                'source': 'loaded_from_file'
            }
            all_results['seed_results'].append(seed_42_result)
            seed_42_loaded = True
            
            print(f"✅ Successfully loaded seed 42 results")
            print(f"  QWK:  {seed_42_result['metrics']['qwk']:.4f}")
            print(f"  MAE:  {seed_42_result['metrics']['mae']:.4f}")
            print(f"  RMSE: {seed_42_result['metrics']['rmse']:.4f}")
    except Exception as e:
        print(f"❌ Error loading seed 42: {e}")
    
    if not seed_42_loaded:
        print("\n❌ ERROR: Could not load seed 42 results!")
        print("This experiment requires seed 42 baseline.")
        print(f"Please provide valid JSON file at: {seed_42_path}")
        return None
    
    # STEP 2: Download test data
    print(f"\n📥 Downloading test data...")
    test_data_result = download_test_data()
    if test_data_result["status"] != "success":
        print("❌ Failed to download test data")
        return None
    
    # STEP 3: Run new seeds (123 and 456)
    print(f"\n{'='*70}")
    print(f"RUNNING NEW SEEDS: {NEW_SEEDS}")
    print(f"{'='*70}")
    
    for idx, seed in enumerate(NEW_SEEDS):
        print(f"\n[Seed {idx+1}/{len(NEW_SEEDS)}]")
        
        result = run_single_seed(api_key, seed, test_data_result)
        
        if result:
            result['source'] = 'newly_computed'
            all_results['seed_results'].append(result)
            
            # Save partial results after each seed
            partial_path = os.path.join(OUTPUT_DIR, "partial_results.json")
            with open(partial_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✓ Partial results saved")
        
        # Pause between seeds
        if idx < len(NEW_SEEDS) - 1:
            print(f"\n⏳ Waiting 10 seconds before next seed...")
            time.sleep(10)
    
    all_results['end_timestamp'] = datetime.now().isoformat()
    
    # STEP 4: Calculate statistics across all 3 seeds
    print(f"\n{'='*70}")
    print("CALCULATING STATISTICS ACROSS ALL 3 SEEDS")
    print(f"{'='*70}")
    
    if len(all_results['seed_results']) < 2:
        print(f"\n⚠️ Warning: Only {len(all_results['seed_results'])} seeds completed")
        print("Need at least 2 seeds for statistics")
        return all_results
    
    # Extract metric values from all seeds
    metric_names = ['qwk', 'pearson', 'f1', 'precision', 'recall', 'accuracy', 'mae', 'rmse']
    metric_values = {metric: [] for metric in metric_names}
    
    for seed_result in all_results['seed_results']:
        for metric in metric_names:
            metric_values[metric].append(seed_result['metrics'][metric])
    
    # Calculate mean and std for each metric
    aggregated_metrics = {}
    for metric in metric_names:
        values = metric_values[metric]
        aggregated_metrics[f'avg_{metric}'] = np.mean(values)
        aggregated_metrics[f'std_{metric}'] = np.std(values)
        aggregated_metrics[f'min_{metric}'] = np.min(values)
        aggregated_metrics[f'max_{metric}'] = np.max(values)
    
    all_results['aggregated_metrics'] = aggregated_metrics
    
    # Determine stability
    qwk_std = aggregated_metrics['std_qwk']
    if qwk_std < 0.01:
        stability = "VERY STABLE"
    elif qwk_std < 0.02:
        stability = "STABLE"
    elif qwk_std < 0.05:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"
    
    all_results['stability'] = stability
    
    # Print results
    print(f"\nIndividual Seed Results:")
    print("-" * 70)
    for seed_result in all_results['seed_results']:
        m = seed_result['metrics']
        source_tag = f" [{seed_result.get('source', 'unknown')}]"
        print(f"Seed {seed_result['seed']}{source_tag}:")
        print(f"  QWK={m['qwk']:.4f}, Pearson={m['pearson']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")
    
    print(f"\n{'='*70}")
    print("AVERAGED RESULTS ACROSS ALL 3 SEEDS")
    print(f"{'='*70}")
    print(f"Avg QWK:       {aggregated_metrics['avg_qwk']:.15f} ± {aggregated_metrics['std_qwk']:.15f}")
    print(f"Avg Pearson:   {aggregated_metrics['avg_pearson']:.15f} ± {aggregated_metrics['std_pearson']:.15f}")
    print(f"Avg F1:        {aggregated_metrics['avg_f1']:.15f} ± {aggregated_metrics['std_f1']:.15f}")
    print(f"Avg Precision: {aggregated_metrics['avg_precision']:.15f} ± {aggregated_metrics['std_precision']:.15f}")
    print(f"Avg Recall:    {aggregated_metrics['avg_recall']:.15f} ± {aggregated_metrics['std_recall']:.15f}")
    print(f"Avg Accuracy:  {aggregated_metrics['avg_accuracy']:.15f} ± {aggregated_metrics['std_accuracy']:.15f}")
    print(f"Avg MAE:       {aggregated_metrics['avg_mae']:.15f} ± {aggregated_metrics['std_mae']:.15f}")
    print(f"Avg RMSE:      {aggregated_metrics['avg_rmse']:.15f} ± {aggregated_metrics['std_rmse']:.15f}")
    
    print(f"\n{'='*70}")
    print(f"STABILITY ASSESSMENT: {stability}")
    print(f"{'='*70}")
    
    if qwk_std < 0.02:
        print("✓ Conclusion: Results are CONSISTENT across seeds.")
        print("  Your deductive approach is ROBUST!")
    else:
        print("⚠ Conclusion: Results VARY across seeds.")
        print("  Consider investigating why stability is lower.")
    
    # Save final aggregated results
    final_json_path = os.path.join(OUTPUT_DIR, "final_aggregated_results.json")
    with open(final_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved final results to {final_json_path}")
    
    # Create summary table (matching original format)
    summary_data = {
        'Dataset': [DATASET],
        'Filename': [f'{DATASET}.csv'],
        'Avg QWK': [aggregated_metrics['avg_qwk']],
        'Avg Pearson': [aggregated_metrics['avg_pearson']],
        'Avg F1': [aggregated_metrics['avg_f1']],
        'Avg Precision': [aggregated_metrics['avg_precision']],
        'Avg Recall': [aggregated_metrics['avg_recall']],
        'Avg Accuracy': [aggregated_metrics['avg_accuracy']],
        'Avg MAE': [aggregated_metrics['avg_mae']],
        'Avg RMSE': [aggregated_metrics['avg_rmse']],
        'MAE %': ['N/A']
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_table.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ Saved summary table to {summary_csv_path}")
    
    # Create detailed text summary
    summary_txt_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MULTI-SEED DEDUCTIVE EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Dataset: {DATASET}\n")
        f.write(f"Approach: {APPROACH}\n")
        f.write(f"Seeds: {[42] + NEW_SEEDS}\n")
        f.write(f"Seed 42: Loaded from file\n")
        f.write(f"Seeds {NEW_SEEDS}: Newly computed\n\n")
        
        f.write("-"*70 + "\n")
        f.write("INDIVIDUAL SEED RESULTS\n")
        f.write("-"*70 + "\n")
        for seed_result in all_results['seed_results']:
            m = seed_result['metrics']
            f.write(f"\nSeed {seed_result['seed']} [{seed_result.get('source', 'unknown')}]:\n")
            f.write(f"  QWK:       {m['qwk']:.15f}\n")
            f.write(f"  Pearson:   {m['pearson']:.15f}\n")
            f.write(f"  F1:        {m['f1']:.15f}\n")
            f.write(f"  Precision: {m['precision']:.15f}\n")
            f.write(f"  Recall:    {m['recall']:.15f}\n")
            f.write(f"  Accuracy:  {m['accuracy']:.15f}\n")
            f.write(f"  MAE:       {m['mae']:.15f}\n")
            f.write(f"  RMSE:      {m['rmse']:.15f}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("AVERAGED RESULTS (Mean ± Std Dev)\n")
        f.write("-"*70 + "\n")
        f.write(f"Avg QWK:       {aggregated_metrics['avg_qwk']:.15f} ± {aggregated_metrics['std_qwk']:.15f}\n")
        f.write(f"Avg Pearson:   {aggregated_metrics['avg_pearson']:.15f} ± {aggregated_metrics['std_pearson']:.15f}\n")
        f.write(f"Avg F1:        {aggregated_metrics['avg_f1']:.15f} ± {aggregated_metrics['std_f1']:.15f}\n")
        f.write(f"Avg Precision: {aggregated_metrics['avg_precision']:.15f} ± {aggregated_metrics['std_precision']:.15f}\n")
        f.write(f"Avg Recall:    {aggregated_metrics['avg_recall']:.15f} ± {aggregated_metrics['std_recall']:.15f}\n")
        f.write(f"Avg Accuracy:  {aggregated_metrics['avg_accuracy']:.15f} ± {aggregated_metrics['std_accuracy']:.15f}\n")
        f.write(f"Avg MAE:       {aggregated_metrics['avg_mae']:.15f} ± {aggregated_metrics['std_mae']:.15f}\n")
        f.write(f"Avg RMSE:      {aggregated_metrics['avg_rmse']:.15f} ± {aggregated_metrics['std_rmse']:.15f}\n\n")
        f.write(f"Stability Assessment: {stability}\n")
    
    print(f"✓ Saved detailed summary to {summary_txt_path}")
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Final results: {final_json_path}")
    print(f"✓ Summary table: {summary_csv_path}")
    print(f"✓ Detailed summary: {summary_txt_path}")
    print(f"✓ CSV predictions: {CSV_DIR}/")
    print(f"{'='*70}\n")
    
    return all_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("ERROR: Missing API key")
        print("="*70)
        print("\nUsage:")
        print("  python multi_seed_deductive_load_seed42.py API_KEY [SEED_42_JSON_PATH]")
        print("\nExample:")
        print("  python multi_seed_deductive_load_seed42.py sk-or-v1-xxx...")
        print("  python multi_seed_deductive_load_seed42.py sk-or-v1-xxx... ./seed_42_results.json")
        print("\n" + "="*70)
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Default to seed_42_results.json in current directory
    seed_42_path = sys.argv[2] if len(sys.argv) > 2 else "./seed_42_json.json"
    
    if not os.path.exists(seed_42_path):
        print(f"\n❌ ERROR: Seed 42 JSON file not found: {seed_42_path}")
        print("\nExpected file: seed_42_json.json")
        print("Or provide path as second argument")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("MULTI-SEED DEDUCTIVE EXPERIMENT")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Approach: {APPROACH}")
    print(f"Seed 42: Loading from {seed_42_path}")
    print(f"New seeds: {NEW_SEEDS}")
    print(f"Total: 3 seeds")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)
    
    confirm = input("\nProceed with experiment? (y/n): ").strip().lower()
    if confirm == 'y':
        result = run_multi_seed_experiment(api_key, seed_42_path)
        if result:
            print("\n✅ Experiment completed successfully!")
        else:
            print("\n❌ Experiment failed!")
            sys.exit(1)
    else:
        print("\n❌ Experiment cancelled")
        sys.exit(0)