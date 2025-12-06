#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiment - Short Answer
Training examples: ASAP-SAS
Test dataset: CSEE
Reasoning: Inductive + Deductive
Model: LLaMA 4 Scout (OpenRouter)
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

# IMPORTANT: this helper must define get_score_range_for_dataset(dataset_name:str, essay_set:int)->(min,max)
from dataset_ranges import get_score_range_for_dataset

os.environ["HF_TOKEN"] = "REMOVED_KEY"

TRAIN_DATASET = "ASAP-SAS"   # Training examples are sampled from here
TEST_DATASET  = "CSEE"       # Model is evaluated on this dataset
NUM_EXAMPLES  = 5            # n-shot examples
RANDOM_SEED   = 42
TEST_DATASET_SIZE_LIMIT = 1000 
MODEL_CODE = "meta-llama/llama-4-scout"
MODEL_NAME = "llama-4-scout"

BASE_DIR       = "generalization"
EXPERIMENT_NAME = f"{MODEL_NAME}_train_{TRAIN_DATASET}_test_{TEST_DATASET}_ind_ded"
OUTPUT_DIR      = os.path.join(BASE_DIR, EXPERIMENT_NAME)
PRED_DIR        = OUTPUT_DIR  # keep same layout as your prior scripts

# Column mappings
ASAP_SAS_COLUMNS = {
    "id": "Id",
    "text": "essay_text",
    "score": "Score1",
    "question": "prompt"
}

# From your screenshot of CSEE (index, essay_id, prompt_id, prompt, essay, overall_score, ...)
CSEE_COLUMNS = {
    "id": "index",
    "text": "essay",
    "score": "overall_score",
    "question": "prompt"
}

# Rate-limiting
SLEEP_BETWEEN_CALLS_SEC = 2.0
MAX_RETRIES = 3

# ======================== CLIENT ========================
def get_client(api_key: str):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
def download_training_data():
    """Download training data for cross-dataset experiments, with dataset-specific handling."""
    print(f"  Downloading training data from {TRAIN_DATASET}...")
    try:
        normalized_name = TRAIN_DATASET.strip()

        if normalized_name == "Mohlar":
            # Explicitly load only train.csv
            dataset = load_dataset("nlpatunt/Mohlar", data_files="train.csv")["train"]

        elif normalized_name == "ASAP-SAS":
            # Explicitly load only train.csv (avoids validation.csv column mismatch)
            dataset = load_dataset("nlpatunt/ASAP-SAS", data_files="train.csv")["train"]

        else:
            # Default generic loader
            dataset = load_dataset(f"nlpatunt/{TRAIN_DATASET}", split="train", trust_remote_code=True)

        df = dataset.to_pandas()

        # Drop stray 'Unnamed' columns automatically
        unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
        if unnamed_cols:
            print(f"  ⚠️ Found and dropped columns: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols, errors='ignore')

        print(f"  ✓ Loaded {len(df)} training examples from {TRAIN_DATASET}")
        return {"status": "success", "dataset": df}

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {"status": "error", "error": str(e)}


def download_test_data():
    """Download CSEE test data with optional size limit"""
    print(f"  Downloading test data from {TEST_DATASET}...")
    try:
        ds = load_dataset(f"nlpatunt/{TEST_DATASET}", split="test", trust_remote_code=True)
        df = ds.to_pandas()
        
        original_size = len(df)
        print(f"    ✓ Downloaded: {original_size} rows from {TEST_DATASET}")
        
        # ============================================================
        # APPLY SIZE LIMIT FOR CSEE (NEW CODE)
        # ============================================================
        if TEST_DATASET in ["CSEE", "D_CSEE"] and TEST_DATASET_SIZE_LIMIT:
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
        # ============================================================

        # Keep ground truth; create copy for predictions without score column
        test_df_for_prediction = df.drop(columns=[CSEE_COLUMNS["score"]], errors='ignore')
        return {
            "status": "success",
            "test_data": test_df_for_prediction,
            "ground_truth": df
        }
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {"status": "error", "error": str(e)}
    
def sample_training_examples(train_df: pd.DataFrame, num_examples: int) -> tuple:
    """Sample training examples from ASAP-SAS"""
    sampled = train_df.sample(n=min(num_examples, len(train_df)), random_state=RANDOM_SEED)
    print(f"  ✓ Sampled {len(sampled)} training examples from {TRAIN_DATASET} (seed={RANDOM_SEED})")

    examples = []
    train_ids = []

    # Validate columns exist
    for k, col in ASAP_SAS_COLUMNS.items():
        if col not in train_df.columns:
            raise KeyError(f"[ASAP-SAS] Expected column '{col}' not found in training dataframe.")

    for _, row in sampled.iterrows():
        ex_id = str(row[ASAP_SAS_COLUMNS["id"]])
        ex_text = row[ASAP_SAS_COLUMNS["text"]]
        ex_score = row[ASAP_SAS_COLUMNS["score"]]
        ex_question = row.get(ASAP_SAS_COLUMNS["question"], "")

        examples.append({
            "id": ex_id,
            "text": ex_text,
            "score": ex_score,
            "question": ex_question
        })
        train_ids.append(ex_id)

    return examples, train_ids

# ================== PROMPT (Inductive → Deductive) ==================
def create_inductive_deductive_prompt(essay_text: str, question: str,
                                      training_examples: List[Dict],
                                      dataset_name: str,
                                      score_range: tuple) -> Dict[str, str]:
    """
    Ind→Ded prompt: Learn from examples FIRST, then apply general principles.
    """
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    is_3way = '3way' in normalized_name.lower()
    is_2way = '2way' in normalized_name.lower()

    # Build examples string (NO TRUNCATION)
    examples_str = ""
    for i, ex in enumerate(training_examples, 1):
        examples_str += f"\nEXAMPLE {i}:\n"
        ex_q = ex.get("question", "")
        if ex_q:
            examples_str += f"Question: {ex_q}\n"
        ex_text = ex.get("text", "")
        examples_str += f"Student Answer: {ex_text}\n"
        examples_str += f"Score: {ex['score']}\n"

    # Inductive → Deductive ordering
    system_prompt = f"""You are an expert scorer using INDUCTIVE and then DEDUCTIVE REASONING.

PHASE 1 - INDUCTIVE REASONING (Learn from examples FIRST):
Analyze these training examples (from {TRAIN_DATASET}):
{examples_str}

What scoring patterns emerge from these examples?
- What distinguishes high-scoring from low-scoring responses?
- What quality indicators appear consistently?
- What patterns separate excellent from poor responses?

PHASE 2 - DEDUCTIVE REASONING (Then apply general principles):
Apply universal scoring principles:
- Completeness and accuracy requirements
- Depth and correctness of reasoning
- Clear, coherent communication
- Evidence/justification where appropriate

SCORING RANGE: {score_range}
COMBINED: Learn patterns from examples first, then validate with general principles. Output only a number in range."""

    user_prompt = f"""Classify this answer using INDUCTIVE patterns then DEDUCTIVE principles:

QUESTION: {question}
STUDENT ANSWER: {essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.
Provide ONLY a number between {score_range[0]} and {score_range[1]}. Just the number. Nothing else"""

    return {"system": system_prompt, "user": user_prompt}

# ================== VALIDATION ==================
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

# ================== INFERENCE ==================
def get_prediction_with_retry(client,
                              essay_text: str,
                              question: str,
                              training_examples: List[Dict],
                              score_range: tuple,
                              max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """
    Attempts to get a valid numeric prediction with adaptive retries.
    """
    last_response_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            # Generate prompt
            prompts = create_inductive_deductive_prompt(
                essay_text, question, training_examples, TEST_DATASET, score_range
            )

            # Call model
            response = client.chat.completions.create(
                model=MODEL_CODE,
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]}
                ],
                temperature=0.0
            )

            response_text = response.choices[0].message.content.strip()
            last_response_text = response_text

            # Count tokens
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0

            # Validate
            validation = validate_prediction(response_text, score_range)
            if validation['valid']:
                return {
                    'success': True,
                    'prediction': validation['prediction'],
                    'tokens': tokens,
                    'attempts': attempt
                }

            # If invalid, continue to next retry

        except Exception as e:
            print(f"  ⚠️ Attempt {attempt} raised exception: {e}")
            if attempt == max_retries:
                return {
                    'success': False,
                    'error': str(e),
                    'attempts': attempt,
                    'last_response': last_response_text
                }

    # All retries exhausted
    return {
        'success': False,
        'error': 'Max retries exceeded (all invalid or errored)',
        'attempts': max_retries,
        'last_response': last_response_text
    }

# ================== SAVING ==================
def save_predictions_as_csv(test_df: pd.DataFrame, predictions_map: Dict[str, float]) -> str:
    """
    Merge predictions into test_df, write CSV.
    """
    csv_df = test_df.copy()

    # Map IDs
    csv_df[CSEE_COLUMNS["id"]] = csv_df[CSEE_COLUMNS["id"]].astype(str)
    pred_series = csv_df[CSEE_COLUMNS["id"]].map(predictions_map)

    csv_df["Score"] = pred_series

    out_filename = os.path.join(PRED_DIR, "predictions.csv")
    csv_df.to_csv(out_filename, index=False)
    print(f"  ✓ Predictions saved: {out_filename}")
    return out_filename

# ================== EVALUATION ==================
def calculate_qwk(predictions_csv: str, ground_truth_df: pd.DataFrame) -> Dict:
    """
    Compute QWK using predictions CSV and ground truth.
    """
    try:
        from sklearn.metrics import cohen_kappa_score

        # Load predictions
        pred_df = pd.read_csv(predictions_csv)

        # Ensure ID columns are strings
        pred_df[CSEE_COLUMNS["id"]] = pred_df[CSEE_COLUMNS["id"]].astype(str)
        ground_truth_df[CSEE_COLUMNS["id"]] = ground_truth_df[CSEE_COLUMNS["id"]].astype(str)

        # Merge on ID
        merged = pd.merge(
            ground_truth_df[[CSEE_COLUMNS["id"], CSEE_COLUMNS["score"]]],
            pred_df[[CSEE_COLUMNS["id"], "Score"]],
            on=CSEE_COLUMNS["id"],
            how="inner"
        )

        y_true = merged[CSEE_COLUMNS["score"]].values
        y_pred = merged["Score"].values

        # Filter out NaNs
        valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        if len(y_true) == 0:
            print("  ⚠️ No valid predictions to evaluate")
            return None

        # Round predictions
        y_pred_rounded = np.round(y_pred).astype(int)

        qwk = cohen_kappa_score(y_true, y_pred_rounded, weights='quadratic')
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

# ================== MAIN ==================
def run_generalization_experiment(api_key: str):
    client = get_client(api_key)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}/")

    print("\n" + "="*70)
    print(f"CROSS-DATASET GENERALIZATION EXPERIMENT - INDUCTIVE + DEDUCTIVE")
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

    # Sample examples from ASAP-SAS
    training_examples, train_ids = sample_training_examples(train_df, NUM_EXAMPLES)

    # Get CSEE score range (essay_set=1 convention)
    csee_range = get_score_range_for_dataset(TEST_DATASET, 1)

    # Initialize results structure
    results = {
        'experiment_type': 'cross_dataset_generalization_ind_ded',
        'model_code': MODEL_CODE,
        'model_name': MODEL_NAME,
        'train_dataset': TRAIN_DATASET,
        'test_dataset': TEST_DATASET,
        'reasoning_approach': 'inductive_deductive',
        'num_training_examples': NUM_EXAMPLES,
        'random_seed': RANDOM_SEED,
        'training_ids': train_ids,
        'start_timestamp': datetime.now().isoformat(),
        'test_examples': int(len(test_df)),
        'predictions': [],
        'failed_predictions': [],
        'stats': {'valid': 0, 'invalid': 0}
    }

    predictions_map: Dict[str, float] = {}

    # Iterate test set
    print(f"\n🔄 Processing {len(test_df)} test answers from {TEST_DATASET}...")
    print(f"   Using {NUM_EXAMPLES} training examples from {TRAIN_DATASET}")
    print(f"   Reasoning: Inductive + Deductive")

    # Validate required columns in test df
    for k, col in CSEE_COLUMNS.items():
        if k != "score" and col not in test_df.columns:
            raise KeyError(f"[CSEE] Expected column '{col}' not found in test dataframe.")

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_df)} ({100*i/len(test_df):.1f}%)")

        essay_id = str(row[CSEE_COLUMNS["id"]])
        essay_text = row[CSEE_COLUMNS["text"]]
        question = row.get(CSEE_COLUMNS["question"], "")

        result = get_prediction_with_retry(
            client, essay_text, question, training_examples, csee_range
        )

        if result['success']:
            pred_entry = {
                'id': essay_id,
                'prediction': result['prediction'],
                'tokens': result.get('tokens'),
                'attempts': result.get('attempts', 1)
            }
            results['predictions'].append(pred_entry)
            results['stats']['valid'] += 1
            predictions_map[essay_id] = float(result['prediction'])
        else:
            fail_entry = {
                'id': essay_id,
                'error': result['error'],
                'attempts': result.get('attempts', MAX_RETRIES)
            }
            if 'last_response' in result:
                fail_entry['last_response'] = str(result['last_response'])[:200]
            results['failed_predictions'].append(fail_entry)
            results['stats']['invalid'] += 1

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        # Optional: write partial progress
        if i % 250 == 0:
            tmp_json = os.path.join(OUTPUT_DIR, "partial_results.json")
            with open(tmp_json, "w") as f:
                json.dump(results, f, indent=2)

    results['end_timestamp'] = datetime.now().isoformat()

    print(f"\n✅ Prediction Complete:")
    print(f"   Valid: {results['stats']['valid']}")
    print(f"   Invalid: {results['stats']['invalid']}")

    # Save predictions CSV
    print(f"\n💾 Saving predictions...")
    csv_path = save_predictions_as_csv(test_df, predictions_map)
    results['csv_output'] = csv_path

    # Metrics
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

    # Save results JSON
    results_filename = os.path.join(OUTPUT_DIR, f"results_{int(time.time())}.json")
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2)

    # Write summary.txt
    summary_filename = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_filename, "w") as f:
        f.write("="*70 + "\n")
        f.write("CROSS-DATASET GENERALIZATION - INDUCTIVE + DEDUCTIVE\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Training Dataset: {TRAIN_DATASET}\n")
        f.write(f"Test Dataset: {TEST_DATASET}\n")
        f.write(f"Training Examples: {NUM_EXAMPLES}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n")
        f.write(f"Score Range (target {TEST_DATASET}): {csee_range}\n\n")
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

        f.write("\n" + "-"*70 + "\n")
        f.write(f"TRAINING EXAMPLES USED (from {TRAIN_DATASET})\n")
        f.write("-"*70 + "\n")
        for tid in results['training_ids']:
            f.write(f"  {tid}\n")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Results JSON: {results_filename}")
    print(f"✅ Summary: {summary_filename}")
    print(f"✅ Predictions CSV: {csv_path}")
    print(f"{'='*70}\n")

    return results

# ================== ENTRYPOINT ==================
if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key provided")
        print("\nUsage: python generalization_ind_ded_ASAPSAS_to_CSEE.py YOUR_API_KEY")
        sys.exit(1)

    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION - INDUCTIVE + DEDUCTIVE")
    print("="*70)
    print(f"Training from: {TRAIN_DATASET}")
    print(f"Testing on: {TEST_DATASET}")
    print(f"Model: {MODEL_NAME}")
    print(f"Approach: Inductive + Deductive")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)

    confirm = input("\nProceed with generalization experiment? (y/n): ").strip().lower()
    if confirm == 'y':
        _ = run_generalization_experiment(api_key)
    else:
        print("Cancelled")