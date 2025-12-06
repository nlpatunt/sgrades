#!/usr/bin/env python3
"""
Multi-Run Randomization Experiment for ASAP-AES ONLY
Tests if random sampling affects QWK performance
3 runs: 1 pre-calculated (seed 42) + 2 new (seeds 123, 456)
Inductive + Deductive approach
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from dataset_ranges import get_score_range_for_dataset
except ImportError:
    print("Error: Could not import get_score_range_for_dataset")
    sys.exit(1)

os.environ["HF_TOKEN"] = "REMOVED_KEY"

# ===== CONFIGURATION - ASAP-AES ONLY =====
DATASET_NAME = "ASAP-AES"
NUM_EXAMPLES = 5  # Number of training examples to sample per run
NEW_SEEDS = [123, 456]  # Only new seeds (seed 42 pre-calculated)
MODEL_CODE = "openai/gpt-4o-mini"
MODEL_NAME = "gpt-4o-mini"
TEST_SAMPLE_SIZE = None  # Set to None for full test set, or number for testing (e.g., 2)

BASE_DIR = "Randomization/gpt4mini_asap_aes_3_runs"
CSV_DIR = os.path.join(BASE_DIR, "predictions")

COLUMNS = {
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

def download_training_data():
    """Download ASAP-AES training data"""
    print(f"  Downloading training data for {DATASET_NAME}...")
    try:
        dataset = load_dataset(f"nlpatunt/{DATASET_NAME}", split="train", trust_remote_code=True)
        df = dataset.to_pandas()
        print(f"  ✓ Loaded {len(df)} training examples")
        return {"status": "success", "dataset": df}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"status": "error", "error": str(e)}

def download_test_data():
    """Download ASAP-AES test data"""
    print(f"  Downloading test data for {DATASET_NAME}...")
    try:
        dataset = load_dataset(f"nlpatunt/{DATASET_NAME}", split="test", trust_remote_code=True)
        df = dataset.to_pandas()
        
        # Apply test sample size if set
        if TEST_SAMPLE_SIZE is not None:
            df = df.head(TEST_SAMPLE_SIZE)
            print(f"    ⚠ TESTING MODE: Limited to {len(df)} test examples")
        else:
            print(f"    ✓ Downloaded: {len(df)} rows (full test set)")
        
        return {
            "status": "success", 
            "test_data": df,
            "ground_truth": df
        }
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {"status": "error", "error": str(e)}

def sample_training_examples(train_df: pd.DataFrame, num_examples: int, random_seed: int) -> tuple:
    """Sample training examples with given random seed"""
    sampled = train_df.sample(n=min(num_examples, len(train_df)), random_state=random_seed)
    print(f"  ✓ Sampled {len(sampled)} training examples (seed={random_seed})")
    
    examples = []
    train_ids = []
    
    for _, row in sampled.iterrows():
        essay_id = str(row[COLUMNS["id"]])
        essay_text = row[COLUMNS["text"]]
        score = row[COLUMNS["score"]]
        question = row.get(COLUMNS["question"], "")
        
        examples.append({
            "id": essay_id,
            "text": essay_text,
            "score": score,
            "question": question
        })
        train_ids.append(essay_id)
    
    return examples, train_ids

def create_inductive_deductive_prompt(essay_text: str, question: str,
                                     training_examples: List[Dict], 
                                     score_range: tuple) -> str:
    """Create combined inductive + deductive prompt - EXACT format from seed 42"""
    min_score, max_score = score_range
    
    # Build examples string
    examples_str = ""
    for idx, example in enumerate(training_examples, 1):
        examples_str += f"""Example {idx}:
Question: {example.get('question', '')}
Essay: {example['text'][:500]}...
Score: {example['score']}

"""
    
    system_prompt = f"""You are an expert scorer using INDUCTIVE then DEDUCTIVE REASONING.

PHASE 1 - INDUCTIVE REASONING:
{examples_str}

Learn scoring patterns from these examples.

PHASE 2 - DEDUCTIVE REASONING:
Apply general scoring rules:
- Completeness and accuracy
- Depth of reasoning
- Clear communication

SCORING RANGE: {min_score} to {max_score}
Use both learned patterns AND general principles."""

    user_prompt = f"""Classify this answer using INDUCTIVE patterns then DEDUCTIVE inference:

QUESTION: {question}
ESSAY: {essay_text}
STOP. Do not write steps. Do not write explanations. Do not write reasoning.
Provide ONLY a number between {min_score} and {max_score}. Just the number. Nothing Else"""

    # Combine system and user prompts into single message
    full_prompt = f"""{system_prompt}

{user_prompt}"""

    return full_prompt

def validate_prediction(response_text: str, score_range: tuple) -> Dict[str, Any]:
    """Validate response is a valid score"""
    min_score, max_score = score_range
    text = response_text.strip()
    
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    
    if not matches:
        return {'valid': False, 'error': 'No numeric value found'}
    
    try:
        predicted_score = float(matches[0])
        
        if predicted_score < min_score or predicted_score > max_score:
            return {
                'valid': False,
                'error': f'Score {predicted_score} outside range [{min_score}, {max_score}]'
            }
        
        return {'valid': True, 'prediction': predicted_score}
    except ValueError:
        return {'valid': False, 'error': f'Could not convert to number'}

def get_prediction_with_retry(client, essay_text: str, question: str, 
                              training_examples: List[Dict], score_range: tuple,
                              max_retries: int = 3) -> Dict[str, Any]:
    """Get prediction with retry logic"""
    for attempt in range(max_retries):
        try:
            prompt = create_inductive_deductive_prompt(
                essay_text, question, training_examples, score_range
            )
            
            response = client.chat.completions.create(
                model=MODEL_CODE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            response_text = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            
            validation = validate_prediction(response_text, score_range)
            
            if validation['valid']:
                return {
                    'success': True,
                    'prediction': validation['prediction'],
                    'tokens': tokens_used
                }
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return {'success': False, 'error': 'Validation failed'}

def save_predictions_as_csv(test_df: pd.DataFrame, predictions_map: Dict, run_number: int, seed: int):
    """Save predictions to CSV"""
    os.makedirs(CSV_DIR, exist_ok=True)
    
    output_df = test_df.copy()
    
    for idx in output_df.index:
        row_id = str(output_df.loc[idx, COLUMNS["id"]])
        if row_id in predictions_map:
            output_df.loc[idx, COLUMNS["score"]] = predictions_map[row_id]
    
    csv_filename = f"run{run_number}_seed{seed}_predictions.csv"
    csv_path = os.path.join(CSV_DIR, csv_filename)
    
    output_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    
    return csv_path

def calculate_qwk_for_run(csv_path: str, ground_truth_df: pd.DataFrame):
    """Calculate QWK for a single run"""
    try:
        from sklearn.metrics import cohen_kappa_score
        
        pred_df = pd.read_csv(csv_path)
        
        merged = pred_df.merge(
            ground_truth_df[[COLUMNS["id"], COLUMNS["score"]]], 
            on=COLUMNS["id"], 
            how='inner',
            suffixes=('_pred', '_true')
        )
        
        y_pred = merged[f'{COLUMNS["score"]}_pred'].values
        y_true = merged[f'{COLUMNS["score"]}_true'].values
        
        valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        
        if len(y_pred) == 0:
            return None
        
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        return {
            'qwk': round(qwk, 4),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'num_predictions': len(y_pred)
        }
    except Exception as e:
        print(f"  ✗ Error calculating QWK: {e}")
        return None

def run_single_experiment(api_key, run_number: int, random_seed: int, 
                         train_df: pd.DataFrame, test_data_result: dict):
    """Run a single experiment with given random seed"""
    client = get_client(api_key)
    
    print(f"\n{'='*70}")
    print(f"RUN {run_number} - Random Seed: {random_seed}")
    print(f"{'='*70}")
    
    training_examples, train_ids = sample_training_examples(train_df, NUM_EXAMPLES, random_seed)
    test_df = test_data_result["test_data"]
    
    run_result = {
        'run_number': run_number,
        'random_seed': random_seed,
        'dataset': DATASET_NAME,
        'training_examples_used': len(training_examples),
        'test_examples_total': len(test_df),
        'stats': {'valid': 0, 'invalid': 0},
        'start_time': datetime.now().isoformat()
    }
    
    predictions_map = {}
    
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 100 == 0 or i == len(test_df):
            print(f"  Progress: {i}/{len(test_df)}")
        
        essay_id = str(row[COLUMNS["id"]])
        essay_text = row[COLUMNS["text"]]
        question = row.get(COLUMNS["question"], "")
        essay_set = int(row[COLUMNS["essay_set"]])
        
        score_range = get_score_range_for_dataset(DATASET_NAME, essay_set)
        
        result = get_prediction_with_retry(
            client, essay_text, question, training_examples, score_range
        )
        
        if result['success']:
            run_result['stats']['valid'] += 1
            predictions_map[essay_id] = result['prediction']
        else:
            run_result['stats']['invalid'] += 1
        
        time.sleep(1)
    
    csv_path = save_predictions_as_csv(test_df, predictions_map, run_number, random_seed)
    run_result['csv_output'] = csv_path
    run_result['end_time'] = datetime.now().isoformat()
    
    print(f"\n  Calculating QWK for Run {run_number}...")
    qwk_result = calculate_qwk_for_run(csv_path, test_data_result["ground_truth"])
    
    if qwk_result:
        run_result['metrics'] = qwk_result
        print(f"  ✓ Run {run_number} Complete:")
        print(f"     Valid: {run_result['stats']['valid']}")
        print(f"     Invalid: {run_result['stats']['invalid']}")
        print(f"     QWK: {qwk_result['qwk']:.4f}")
        print(f"     MAE: {qwk_result['mae']:.4f}")
        print(f"     RMSE: {qwk_result['rmse']:.4f}")
    
    return run_result

def run_multi_experiment(api_key, seed_42_json_path):
    """Main function to run experiments"""
    
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"✓ Output directory: {BASE_DIR}/")
    
    print("\n" + "="*70)
    print(f"RANDOMIZATION EXPERIMENT: {DATASET_NAME} ONLY")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Approach: Inductive + Deductive")
    print(f"New runs: 2 (seeds 123, 456)")
    print(f"Plus pre-calculated: seed 42")
    print(f"Total runs: 3")
    print(f"Training examples: {NUM_EXAMPLES} per run")
    print("="*70)
    
    print("\n📥 Downloading data...")
    train_result = download_training_data()
    if train_result["status"] != "success":
        print("❌ Failed to download training data")
        return None
    
    test_data_result = download_test_data()
    if test_data_result["status"] != "success":
        print("❌ Failed to download test data")
        return None
    
    train_df = train_result["dataset"]
    
    all_results = {
        'experiment_type': 'randomization_asap_aes',
        'model_code': MODEL_CODE,
        'model_name': MODEL_NAME,
        'dataset': DATASET_NAME,
        'new_seeds': NEW_SEEDS,
        'total_runs': 3,
        'num_training_examples': NUM_EXAMPLES,
        'start_timestamp': datetime.now().isoformat(),
        'runs': []
    }
    
    # Load seed 42 results - REQUIRED
    print("\n" + "="*70)
    print("LOADING PRE-CALCULATED SEED 42 RESULTS")
    print("="*70)
    
    seed_42_loaded = False
    if seed_42_json_path and os.path.exists(seed_42_json_path):
        try:
            with open(seed_42_json_path, 'r') as f:
                seed_42_data = json.load(f)
            
            if 'runs' in seed_42_data and len(seed_42_data['runs']) > 0:
                run_42 = seed_42_data['runs'][0].copy()
                run_42['run_number'] = 1
                run_42['source'] = 'pre-calculated'
                all_results['runs'].append(run_42)
                seed_42_loaded = True
                print("✓ Loaded seed 42 successfully:")
                if 'metrics' in run_42:
                    m = run_42['metrics']
                    print(f"  QWK: {m['qwk']:.4f}, MAE: {m['mae']:.4f}, RMSE: {m['rmse']:.4f}")
                else:
                    print("  ⚠ Warning: No metrics found in seed 42 data")
        except Exception as e:
            print(f"❌ Error loading seed 42: {e}")
    
    if not seed_42_loaded:
        print("\n❌ ERROR: Could not load seed 42 results!")
        print("This experiment requires seed 42 baseline for comparison.")
        print("\nPlease provide valid seed_42_results.json file")
        return None
    
    # Run new experiments
    print("\n" + "="*70)
    print("RUNNING NEW EXPERIMENTS (SEEDS 123, 456)")
    print("="*70)
    
    for run_idx, seed in enumerate(NEW_SEEDS):
        run_number = run_idx + 2  # Runs 2 and 3
        
        run_result = run_single_experiment(api_key, run_number, seed, train_df, test_data_result)
        
        if run_result:
            all_results['runs'].append(run_result)
            
            # Save partial results after each run
            temp_filename = os.path.join(BASE_DIR, "partial_results.json")
            with open(temp_filename, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✓ Partial results saved: {temp_filename}")
        
        if run_idx < len(NEW_SEEDS) - 1:
            print(f"\n⏳ Waiting 10 seconds before next run...\n")
            time.sleep(10)
    
    all_results['end_timestamp'] = datetime.now().isoformat()
    
    # Calculate statistics
    print("\n" + "="*70)
    print("CALCULATING STATISTICS ACROSS ALL 3 RUNS")
    print("="*70)
    
    qwk_values = []
    mae_values = []
    rmse_values = []
    
    for run in all_results['runs']:
        if 'metrics' in run:
            qwk_values.append(run['metrics']['qwk'])
            mae_values.append(run['metrics']['mae'])
            rmse_values.append(run['metrics']['rmse'])
    
    if len(qwk_values) >= 2:  # At least 2 runs completed
        statistics = {
            'qwk': {
                'mean': round(float(np.mean(qwk_values)), 4),
                'std': round(float(np.std(qwk_values)), 4),
                'min': round(float(np.min(qwk_values)), 4),
                'max': round(float(np.max(qwk_values)), 4),
                'values': qwk_values,
                'num_runs': len(qwk_values)
            },
            'mae': {
                'mean': round(float(np.mean(mae_values)), 4),
                'std': round(float(np.std(mae_values)), 4)
            },
            'rmse': {
                'mean': round(float(np.mean(rmse_values)), 4),
                'std': round(float(np.std(rmse_values)), 4)
            }
        }
        
        all_results['statistics'] = statistics
        
        qwk_std = statistics['qwk']['std']
        if qwk_std < 0.01:
            stability = "VERY STABLE"
        elif qwk_std < 0.02:
            stability = "STABLE"
        elif qwk_std < 0.05:
            stability = "MODERATE"
        else:
            stability = "UNSTABLE"
        
        print(f"\nIndividual Run Results:")
        print("-" * 70)
        for run in all_results['runs']:
            if 'metrics' in run:
                m = run['metrics']
                source = f" [{run.get('source', 'new')}]"
                print(f"Run {run['run_number']} (seed={run['random_seed']}){source}: QWK={m['qwk']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")
        
        print("\n" + "="*70)
        print("STATISTICAL SUMMARY")
        print("="*70)
        print(f"\nQWK Statistics (across {len(qwk_values)} runs):")
        print(f"  Mean:     {statistics['qwk']['mean']:.4f}")
        print(f"  Std Dev:  {statistics['qwk']['std']:.4f}")
        print(f"  Min:      {statistics['qwk']['min']:.4f}")
        print(f"  Max:      {statistics['qwk']['max']:.4f}")
        print(f"  Range:    {statistics['qwk']['max'] - statistics['qwk']['min']:.4f}")
        
        print(f"\nMAE Statistics:")
        print(f"  Mean:     {statistics['mae']['mean']:.4f}")
        print(f"  Std Dev:  {statistics['mae']['std']:.4f}")
        
        print(f"\nRMSE Statistics:")
        print(f"  Mean:     {statistics['rmse']['mean']:.4f}")
        print(f"  Std Dev:  {statistics['rmse']['std']:.4f}")
        
        print(f"\n{'='*70}")
        print(f"STABILITY ASSESSMENT: {stability}")
        print(f"{'='*70}")
        
        if qwk_std < 0.02:
            print("✓ Conclusion: Random sampling has MINIMAL effect on QWK.")
            print("  Your method is ROBUST and reliable across different training samples!")
        else:
            print("⚠ Conclusion: Random sampling AFFECTS results significantly.")
            print("  Consider: More training examples or better sampling strategy.")
    else:
        print(f"\n⚠ Warning: Only {len(qwk_values)} runs completed. Need at least 2 for statistics.")
    
    # Save final results
    final_filename = os.path.join(BASE_DIR, f"final_results_{int(time.time())}.json")
    with open(final_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary table
    summary_file = os.path.join(BASE_DIR, "summary_table.txt")
    with open(summary_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("ASAP-AES RANDOMIZATION EXPERIMENT - SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Approach: Inductive + Deductive\n")
        f.write(f"Training Examples: {NUM_EXAMPLES} per run\n")
        f.write(f"Total Runs: {len(all_results['runs'])}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("INDIVIDUAL RUN RESULTS\n")
        f.write("-"*70 + "\n")
        for run in all_results['runs']:
            if 'metrics' in run:
                m = run['metrics']
                source = run.get('source', 'new')
                f.write(f"Run {run['run_number']} (seed={run['random_seed']}) [{source}]\n")
                f.write(f"  QWK:  {m['qwk']:.4f}\n")
                f.write(f"  MAE:  {m['mae']:.4f}\n")
                f.write(f"  RMSE: {m['rmse']:.4f}\n\n")
        
        if 'statistics' in all_results:
            stats = all_results['statistics']
            f.write("-"*70 + "\n")
            f.write("STATISTICAL SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"QWK:  Mean={stats['qwk']['mean']:.4f}, Std={stats['qwk']['std']:.4f}\n")
            f.write(f"MAE:  Mean={stats['mae']['mean']:.4f}, Std={stats['mae']['std']:.4f}\n")
            f.write(f"RMSE: Mean={stats['rmse']['mean']:.4f}, Std={stats['rmse']['std']:.4f}\n\n")
            f.write(f"Stability: {stability}\n")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"✓ Final results: {final_filename}")
    print(f"✓ Summary table: {summary_file}")
    print(f"✓ CSV files: {CSV_DIR}/")
    print("="*70)
    
    return all_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("ERROR: Missing API key")
        print("="*70)
        print("\nUsage:")
        print("  python run_asap_aes_randomization.py API_KEY [SEED_42_JSON_PATH]")
        print("\nExample:")
        print("  python run_asap_aes_randomization.py sk-or-v1-xxx...")
        print("  python run_asap_aes_randomization.py sk-or-v1-xxx... ./seed_42_json.json")
        print("\n" + "="*70)
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    # Default to seed_42_json.json in current directory
    seed_42_path = sys.argv[2] if len(sys.argv) > 2 else "./seed_42_json.json"
    
    if not os.path.exists(seed_42_path):
        print(f"\n❌ ERROR: Seed 42 JSON file not found: {seed_42_path}")
        print("\nExpected file: seed_42_json.json in current directory")
        print("Or provide path as second argument:")
        print("  python run_asap_aes_randomization.py API_KEY path/to/seed_42_json.json")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("ASAP-AES RANDOMIZATION EXPERIMENT")
    print("="*70)
    print(f"Dataset: {DATASET_NAME} ONLY")
    print(f"Seed 42: Will be loaded from {seed_42_path}")
    print(f"New runs: 2 (seeds 123, 456)")
    print(f"Total: 3 runs")
    print(f"Output: {BASE_DIR}/")
    print("="*70)
    
    confirm = input("\nProceed with experiment? (y/n): ").strip().lower()
    if confirm == 'y':
        result = run_multi_experiment(api_key, seed_42_path)
        if result:
            print("\n✅ Experiment completed successfully!")
        else:
            print("\n❌ Experiment failed!")
            sys.exit(1)
    else:
        print("\n❌ Experiment cancelled by user")
        sys.exit(0)