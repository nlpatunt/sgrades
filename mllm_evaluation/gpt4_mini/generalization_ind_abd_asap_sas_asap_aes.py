#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiment - Short Answer
Training examples: ASAP-AES
Test dataset: ASAP-SAS (LIMITED TO 1200 SAMPLES)
Approach: Inductive + Abductive
Model: GPT-4o-mini
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
TRAIN_DATASET = "ASAP-AES"  # Where training examples come from
TEST_DATASET = "ASAP-SAS"  # What we're testing on
NUM_EXAMPLES = 5  # Number of training examples to sample
RANDOM_SEED = 42

# ============================================================================
# NEW: TEST DATASET SIZE LIMIT CONFIGURATION
# ============================================================================
TEST_DATASET_SIZE_LIMIT = 1200  # Limit ASAP-SAS test data to 1200 samples
                                # Set to None to disable limiting
# ============================================================================

MODEL_CODE = "openai/gpt-4o-mini"
MODEL_NAME = "gpt-4o-mini"


# Output directory
BASE_DIR = "generalization"
EXPERIMENT_NAME = f"{MODEL_NAME}_train_{TRAIN_DATASET}_test_{TEST_DATASET}_ind_abd"
OUTPUT_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)

# Column mappings for both datasets
ASAP_AES_COLUMNS = {
    "id": "essay_id",
    "text": "essay",
    "score": "domain1_score",
    "question": "prompt",
    "essay_set": "essay_set"
}

ASAP_SAS_COLUMNS = {
    "id": "Id",
    "text": "essay_text",
    "score": "Score1",
    "question": "prompt"
}

def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

def download_training_data():
    """Download ASAP-AES training data"""
    print(f"  Downloading training data from {TRAIN_DATASET}...")
    try:
        dataset = load_dataset(f"nlpatunt/{TRAIN_DATASET}", split="train", trust_remote_code=True)
        df = dataset.to_pandas()
        print(f"  ✓ Loaded {len(df)} training examples from {TRAIN_DATASET}")
        return {"status": "success", "dataset": df}
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"status": "error", "error": str(e)}

def download_test_data():
    """Download ASAP-SAS test data with optional size limit"""
    print(f"  Downloading test data from {TEST_DATASET}...")
    try:
        dataset = load_dataset(f"nlpatunt/{TEST_DATASET}", data_files="test.csv", trust_remote_code=True)["train"]
        df = dataset.to_pandas()
        
        original_size = len(df)
        print(f"    ✓ Downloaded: {original_size} rows from {TEST_DATASET}")
        
        # ====================================================================
        # APPLY SIZE LIMIT FOR ASAP-SAS TEST DATASET
        # ====================================================================
        if TEST_DATASET in ["ASAP-SAS", "D_ASAP-SAS"] and TEST_DATASET_SIZE_LIMIT:
            if original_size > TEST_DATASET_SIZE_LIMIT:
                print(f"    ⚠️  Applying size limit to {TEST_DATASET}")
                print(f"       Original size: {original_size}")
                print(f"       Target size: {TEST_DATASET_SIZE_LIMIT}")
                
                # Random sampling with fixed seed for reproducibility
                df = df.sample(n=TEST_DATASET_SIZE_LIMIT, random_state=RANDOM_SEED).reset_index(drop=True)
                
                print(f"       ✅ Sampled {len(df)} rows (seed={RANDOM_SEED})")
                print(f"       📊 Reduction: {original_size - TEST_DATASET_SIZE_LIMIT} samples ({100*(original_size-TEST_DATASET_SIZE_LIMIT)/original_size:.1f}%)")
            else:
                print(f"       ℹ️  Dataset size ({original_size}) already below limit ({TEST_DATASET_SIZE_LIMIT})")
        
        print(f"    🎯 Processing {len(df)} samples from {TEST_DATASET}")
        # ====================================================================
        
        # Keep ground truth separate for evaluation
        test_df_for_prediction = df.drop(columns=[ASAP_SAS_COLUMNS["score"]], errors='ignore')
        
        return {
            "status": "success", 
            "test_data": test_df_for_prediction,
            "ground_truth": df
        }
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {"status": "error", "error": str(e)}

def sample_training_examples(train_df: pd.DataFrame, num_examples: int) -> tuple:
    """Sample training examples from asap_aes"""
    sampled = train_df.sample(n=min(num_examples, len(train_df)), random_state=RANDOM_SEED)
    
    print(f"  ✓ Sampled {len(sampled)} training examples from {TRAIN_DATASET} (seed={RANDOM_SEED})")
    
    examples = []
    train_ids = []
    
    for _, row in sampled.iterrows():
        essay_id = str(row[ASAP_AES_COLUMNS["id"]])
        essay_text = row[ASAP_AES_COLUMNS["text"]]
        score = row[ASAP_AES_COLUMNS["score"]]
        question = row.get(ASAP_AES_COLUMNS["question"], "")
        
        # Get ASAP_AES_COLUMNS score range from dataset_ranges.py
        asap_aes_range = get_score_range_for_dataset(TRAIN_DATASET, 1)
        
        examples.append({
            "id": essay_id,
            "text": essay_text,
            "score": score,
            "question": question,
            "score_range": asap_aes_range
        })
        train_ids.append(essay_id)
    
    return examples, train_ids

def create_cross_dataset_ind_abd_prompt(essay_text: str, question: str,
                                        training_examples: List[Dict], 
                                        target_score_range: tuple) -> str:
    """
    Create Inductive + Abductive prompt for cross-dataset generalization
    Training: ASAP-AES examples
    Testing: ASAP-SAS essays
    """
    target_min, target_max = target_score_range
    
    # Get asap_aes range from dataset_ranges.py
    asap_aes_range = get_score_range_for_dataset(TRAIN_DATASET, 1)
    source_min, source_max = asap_aes_range
    
    # Build examples string (NO TRUNCATION)
    examples_str = ""
    for i, ex in enumerate(training_examples, 1):
        examples_str += f"\nEXAMPLE {i} (from {TRAIN_DATASET}, score range {source_min}-{source_max}):\n"
        ex_q = ex.get("question", "")
        if ex_q:
            examples_str += f"Question: {ex_q}\n"
        examples_str += f"Student Answer: {ex.get('text', '')}\n"
        examples_str += f"Score: {ex['score']} (on {source_min}-{source_max} scale)\n"
    
    prompt = f"""You are an expert short answer grader performing CROSS-DATASET GENERALIZATION using INDUCTIVE then ABDUCTIVE REASONING.

**IMPORTANT CONTEXT:**
- Training examples below are from {TRAIN_DATASET} dataset (score range: {source_min}-{source_max})
- Target answer to score is from {TEST_DATASET} dataset (score range: {target_min}-{target_max})
- You must adapt the scoring scale from the training dataset to the test dataset

**PHASE 1 - INDUCTIVE REASONING (Learn from examples FIRST):**
Study these training examples:
{examples_str}

What patterns do you observe?
- What distinguishes high-scoring from low-scoring responses?
- What quality indicators appear consistently?
- How does content completeness relate to scores?

**PHASE 2 - ABDUCTIVE REASONING (Infer best explanation):**
For the target answer below, generate hypotheses:
- Hypothesis 1: High quality (complete, accurate, clear)
- Hypothesis 2: Medium quality (partial understanding, some gaps)
- Hypothesis 3: Low quality (incomplete, unclear, incorrect)

Which hypothesis best explains the target answer?

**CRITICAL SCALE ADAPTATION:**
- Source scale: {source_min}-{source_max}
- Target scale: {target_min}-{target_max}
- Map the quality level to the target scale appropriately

Now evaluate this answer from {TEST_DATASET}:

QUESTION: {question}
STUDENT ANSWER: {essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.
Provide ONLY a number between {target_min} and {target_max}. Just the number. Nothing else."""

    return prompt

def validate_prediction(response_text: str, score_range: tuple) -> Dict[str, Any]:
    min_score, max_score = score_range
    text = response_text.strip()
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    if not matches:
        return {'valid': False, 'error': 'No numeric value found', 'response': text}
    try:
        predicted = float(matches[0])
        if predicted < min_score or predicted > max_score:
            return {'valid': False, 'error': f'Score {predicted} outside range [{min_score}, {max_score}]', 'response': text}
        return {'valid': True, 'prediction': predicted}
    except ValueError:
        return {'valid': False, 'error': 'Could not convert to number', 'response': text}

def get_prediction_with_retry(client, essay_text, question, training_examples, score_range, max_retries=3):
    for attempt in range(max_retries):
        try:
            prompt = create_cross_dataset_ind_abd_prompt(essay_text, question, training_examples, score_range)
            response = client.chat.completions.create(
                model=MODEL_CODE,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            resp_text = response.choices[0].message.content.strip()
            validation = validate_prediction(resp_text, score_range)
            
            if validation['valid']:
                return {
                    'success': True,
                    'prediction': validation['prediction'],
                    'attempts': attempt + 1
                }
            else:
                print(f"    ⚠️ Attempt {attempt + 1}: {validation['error']}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            print(f"    ✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return {'success': False, 'error': 'Max retries exceeded', 'attempts': max_retries}

def save_predictions_as_csv(test_df, predictions_map):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = test_df.copy()
    
    for idx in df.index:
        row_id = str(df.loc[idx, ASAP_SAS_COLUMNS["id"]])
        if row_id in predictions_map:
            df.loc[idx, ASAP_SAS_COLUMNS["score"]] = predictions_map[row_id]
    
    csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved predictions: {csv_path}")
    return csv_path

def calculate_qwk(csv_path, ground_truth_df):
    from sklearn.metrics import cohen_kappa_score
    
    try:
        pred_df = pd.read_csv(csv_path)
        merged = pred_df.merge(
            ground_truth_df[[ASAP_SAS_COLUMNS["id"], ASAP_SAS_COLUMNS["score"]]],
            on=ASAP_SAS_COLUMNS["id"], how='inner', suffixes=('_pred', '_true')
        )
        
        y_pred = merged[f"{ASAP_SAS_COLUMNS['score']}_pred"].values
        y_true = merged[f"{ASAP_SAS_COLUMNS['score']}_true"].values
        
        mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        
        if len(y_pred) == 0:
            print("  ✗ No valid predictions for QWK calculation")
            return None
        
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        return {
            'qwk': round(qwk, 4),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'num_predictions': int(len(y_pred))
        }
    except Exception as e:
        print(f"  ✗ Error calculating QWK: {e}")
        return None

def run_generalization_experiment(api_key: str):
    client = get_client(api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}/")

    print("\n" + "="*70)
    print(f"CROSS-DATASET GENERALIZATION - INDUCTIVE + ABDUCTIVE")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Training dataset: {TRAIN_DATASET}")
    print(f"Test dataset: {TEST_DATASET}")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Download data
    print(f"\n📥 Downloading data...")
    train_result = download_training_data()
    if train_result["status"] != "success":
        print("❌ Failed to download training data")
        return None

    test_result = download_test_data()
    if test_result["status"] != "success":
        print("❌ Failed to download test data")
        return None

    train_df = train_result["dataset"]
    test_df = test_result["test_data"]
    ground_truth_df = test_result["ground_truth"]

    # Sample examples
    training_examples, train_ids = sample_training_examples(train_df, NUM_EXAMPLES)
    
    # Get target score range
    target_range = get_score_range_for_dataset(TEST_DATASET, 1)

    # Initialize results
    results = {
        'experiment_type': 'cross_dataset_generalization_ind_abd',
        'model_code': MODEL_CODE,
        'model_name': MODEL_NAME,
        'train_dataset': TRAIN_DATASET,
        'test_dataset': TEST_DATASET,
        'reasoning_approach': 'inductive_abductive',
        'num_training_examples': NUM_EXAMPLES,
        'random_seed': RANDOM_SEED,
        'training_ids': train_ids,
        'start_timestamp': datetime.now().isoformat(),
        'test_examples': int(len(test_df)),
        'predictions': [],
        'failed_predictions': [],
        'stats': {'valid': 0, 'invalid': 0}
    }

    predictions_map = {}

    # Process test set
    print(f"\n🔄 Processing {len(test_df)} test answers from {TEST_DATASET}...")
    print(f"   Using {NUM_EXAMPLES} training examples from {TRAIN_DATASET}")
    print(f"   Reasoning: Inductive + Abductive")

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_df)} ({100*i/len(test_df):.1f}%)")

        essay_id = str(row[ASAP_SAS_COLUMNS["id"]])
        essay_text = row[ASAP_SAS_COLUMNS["text"]]
        question = row.get(ASAP_SAS_COLUMNS["question"], "")

        result = get_prediction_with_retry(
            client, essay_text, question, training_examples, target_range
        )

        if result['success']:
            pred_entry = {
                'id': essay_id,
                'prediction': result['prediction'],
                'attempts': result.get('attempts', 1)
            }
            results['predictions'].append(pred_entry)
            results['stats']['valid'] += 1
            predictions_map[essay_id] = float(result['prediction'])
        else:
            fail_entry = {
                'id': essay_id,
                'error': result['error'],
                'attempts': result.get('attempts', 3)
            }
            results['failed_predictions'].append(fail_entry)
            results['stats']['invalid'] += 1

        time.sleep(2.0)

    results['end_timestamp'] = datetime.now().isoformat()

    print(f"\n✅ Prediction Complete:")
    print(f"   Valid: {results['stats']['valid']}")
    print(f"   Invalid: {results['stats']['invalid']}")

    # Save predictions
    print(f"\n💾 Saving predictions...")
    csv_path = save_predictions_as_csv(test_df, predictions_map)
    results['csv_output'] = csv_path

    # Calculate metrics
    print(f"\n📊 Calculating QWK...")
    metrics = calculate_qwk(csv_path, ground_truth_df)
    if metrics:
        results['metrics'] = metrics
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"QWK:  {metrics['qwk']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Predictions: {metrics['num_predictions']}")
        print(f"{'='*70}")
    else:
        print("⚠️  Could not calculate metrics")

    # Save results
    results_filename = os.path.join(OUTPUT_DIR, f"results_{int(time.time())}.json")
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary_filename = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_filename, "w") as f:
        f.write("="*70 + "\n")
        f.write("CROSS-DATASET GENERALIZATION - INDUCTIVE + ABDUCTIVE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training Dataset: {TRAIN_DATASET}\n")
        f.write(f"Test Dataset: {TEST_DATASET}\n")
        f.write(f"Training Examples: {NUM_EXAMPLES}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Score Range (target {TEST_DATASET}): {target_range}\n\n")
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        if metrics:
            f.write(f"QWK:  {metrics['qwk']:.4f}\n")
            f.write(f"MAE:  {metrics['mae']:.4f}\n")
            f.write(f"RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"Valid Predictions Used: {metrics['num_predictions']}\n")
        else:
            f.write("Metrics: Could not be calculated\n")

        f.write(f"\nValid Predictions: {results['stats']['valid']}\n")
        f.write(f"Failed Predictions: {results['stats']['invalid']}\n")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Results JSON: {results_filename}")
    print(f"✅ Summary: {summary_filename}")
    print(f"✅ Predictions CSV: {csv_path}")
    print(f"{'='*70}\n")

    return results

if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key provided")
        print("\nUsage: python script.py YOUR_API_KEY")
        sys.exit(1)

    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION - INDUCTIVE + ABDUCTIVE")
    print("="*70)
    print(f"Training from: {TRAIN_DATASET}")
    print(f"Testing on: {TEST_DATASET}")
    print(f"Model: {MODEL_NAME}")
    print(f"Approach: Inductive + Abductive")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)

    confirm = input("\nProceed with generalization experiment? (y/n): ").strip().lower()
    if confirm == 'y':
        _ = run_generalization_experiment(api_key)
    else:
        print("Cancelled")