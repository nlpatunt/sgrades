#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiment - Short Answer
Training examples: ASAP-SAS
Test dataset: CSEE (LIMITED TO 1000 SAMPLES)
Reasoning: Inductive only
Model: Gemini 2.5 Flash (via OpenRouter)
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

# ======================== CONFIG ========================
os.environ.setdefault("HF_TOKEN", "REMOVED_KEY")

TRAIN_DATASET = "ASAP-SAS"
TEST_DATASET = "CSEE"
NUM_EXAMPLES = 5
RANDOM_SEED = 42

# ============================================================================
# NEW: TEST DATASET SIZE LIMIT CONFIGURATION
# ============================================================================
TEST_DATASET_SIZE_LIMIT = 1000  # Limit CSEE test data to 1000 samples
                                # Set to None to disable limiting
# ============================================================================

MODEL_CODE = "google/gemini-2.5-flash"
MODEL_NAME = "gemini-2.5-flash"

BASE_DIR = "generalization"
EXPERIMENT_NAME = f"{MODEL_NAME}_train_{TRAIN_DATASET}_test_{TEST_DATASET}_ind"
OUTPUT_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)
PRED_DIR = OUTPUT_DIR

ASAP_SAS_COLUMNS = {
    "id": "Id",
    "text": "essay_text",
    "score": "Score1",
    "question": "prompt"
}

CSEE_COLUMNS = {
    "id": "essay_id",
    "text": "essay",
    "score": "overall_score",
    "question": "prompt"
}

SLEEP_BETWEEN_CALLS_SEC = 2.0
MAX_RETRIES = 3

# ======================== CLIENT ========================
def get_client(api_key: str):
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# ======================== DATA ==========================
def download_training_data():
    """Download ASAP-SAS training data, ignoring unnamed columns."""
    print(f"  Downloading training data from {TRAIN_DATASET}...")
    try:
        dataset = load_dataset("nlpatunt/ASAP-SAS", data_files="train.csv")["train"]
        df = dataset.to_pandas()
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
        dataset = load_dataset(f"nlpatunt/{TEST_DATASET}", split="test", trust_remote_code=True)
        df = dataset.to_pandas()
        
        # Drop unnamed columns
        unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
        if unnamed_cols:
            print(f"  ⚠️ Found and dropped columns: {unnamed_cols}")
            df = df.drop(columns=unnamed_cols, errors='ignore')
        
        original_size = len(df)
        print(f"    ✓ Downloaded: {original_size} rows from {TEST_DATASET}")
        
        # ====================================================================
        # APPLY SIZE LIMIT FOR CSEE TEST DATASET
        # ====================================================================
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
        # ====================================================================
        
        test_df_for_prediction = df.drop(columns=[CSEE_COLUMNS["score"]], errors='ignore')
        return {"status": "success", "test_data": test_df_for_prediction, "ground_truth": df}
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return {"status": "error", "error": str(e)}

def sample_training_examples(train_df: pd.DataFrame, num_examples: int) -> tuple:
    """Sample training examples."""
    sampled = train_df.sample(n=min(num_examples, len(train_df)), random_state=RANDOM_SEED)
    print(f"  ✓ Sampled {len(sampled)} training examples from {TRAIN_DATASET} (seed={RANDOM_SEED})")

    examples, ids = [], []
    for _, row in sampled.iterrows():
        ex_id = str(row[ASAP_SAS_COLUMNS["id"]])
        ex_text = row[ASAP_SAS_COLUMNS["text"]]
        ex_score = row[ASAP_SAS_COLUMNS["score"]]
        ex_question = row.get(ASAP_SAS_COLUMNS["question"], "")
        examples.append({"id": ex_id, "text": ex_text, "score": ex_score, "question": ex_question})
        ids.append(ex_id)
    return examples, ids

# ================== PROMPT (Inductive Only) ==================
def create_inductive_prompt(essay_text: str, question: str,
                            training_examples: List[Dict],
                            dataset_name: str,
                            score_range: tuple) -> Dict[str, str]:
    """Create inductive-only prompt."""
    examples_str = ""
    for i, ex in enumerate(training_examples, 1):
        examples_str += f"\nEXAMPLE {i}:\n"
        if ex.get("question"): examples_str += f"Question: {ex['question']}\n"
        examples_str += f"Student Answer: {ex['text']}\n"
        examples_str += f"Score: {ex['score']}\n"

    system_prompt = f"""You are an expert short-answer grader using INDUCTIVE REASONING ONLY.

INDUCTIVE REASONING:
- Learn patterns from provided examples.
- Infer how score correlates with completeness, correctness, and clarity.
- Avoid applying external rules or prior knowledge — rely only on example patterns.

Training examples (from {TRAIN_DATASET}):
{examples_str}

Your task: Evaluate new student answers from {TEST_DATASET} using these learned patterns."""

    user_prompt = f"""Question: {question}
Student Answer: {essay_text}

Provide ONLY a numeric score between {score_range[0]} and {score_range[1]} based on pattern similarity to the examples.
Do NOT explain or describe your reasoning — respond with only the number."""

    return {"system": system_prompt, "user": user_prompt}

# ================== VALIDATION ==================
def validate_prediction(response_text: str, score_range: tuple) -> Dict[str, Any]:
    min_s, max_s = score_range
    matches = re.findall(r'[-+]?\d*\.?\d+', response_text)
    if not matches:
        return {'valid': False, 'error': 'No numeric value found', 'response': response_text}
    try:
        val = float(matches[0])
        if val < min_s or val > max_s:
            return {'valid': False, 'error': f'Score {val} outside range [{min_s}, {max_s}]', 'response': response_text}
        return {'valid': True, 'prediction': val}
    except ValueError:
        return {'valid': False, 'error': 'Invalid number', 'response': response_text}

# ================== INFERENCE ==================
def get_prediction_with_retry(client, essay_text, question, training_examples, score_range, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            prompt = create_inductive_prompt(essay_text, question, training_examples, TEST_DATASET, score_range)
            response = client.chat.completions.create(
                model=MODEL_CODE,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                temperature=0.0,
                max_tokens=10
            )
            resp_text = response.choices[0].message.content.strip()
            validation = validate_prediction(resp_text, score_range)
            if validation["valid"]:
                return {"success": True, "prediction": validation["prediction"]}
            else:
                print(f"    ⚠️ Attempt {attempt + 1}: {validation['error']}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            print(f"    ✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    return {"success": False, "error": "Validation failed"}

# ================== SAVE & METRICS ==================
def save_predictions_as_csv(test_df, preds):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = test_df.copy()
    for idx in df.index:
        rid = str(df.loc[idx, CSEE_COLUMNS["id"]])
        if rid in preds:
            df.loc[idx, CSEE_COLUMNS["score"]] = preds[rid]
    path = os.path.join(OUTPUT_DIR, "predictions.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved predictions: {path}")
    return path

def calculate_qwk(csv_path, ground_truth_df):
    from sklearn.metrics import cohen_kappa_score
    pred_df = pd.read_csv(csv_path)
    merged = pred_df.merge(
        ground_truth_df[[CSEE_COLUMNS["id"], CSEE_COLUMNS["score"]]],
        on=CSEE_COLUMNS["id"], how="inner", suffixes=("_pred", "_true")
    )
    y_pred = merged[f"{CSEE_COLUMNS['score']}_pred"].values
    y_true = merged[f"{CSEE_COLUMNS['score']}_true"].values
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    y_pred, y_true = y_pred[mask], y_true[mask]
    if len(y_pred) == 0:
        print("  ❌ No valid predictions for QWK calculation")
        return None
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return {"qwk": round(qwk, 4), "mae": round(mae, 4), "rmse": round(rmse, 4), "num_predictions": len(y_pred)}

# ================== MAIN ==================
def run_generalization_experiment(api_key):
    client = get_client(api_key)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}/")

    print("\n" + "="*70)
    print(f"CROSS-DATASET GENERALIZATION - INDUCTIVE ONLY")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Training dataset: {TRAIN_DATASET}")
    print(f"Test dataset: {TEST_DATASET}")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    train_result = download_training_data()
    if train_result["status"] != "success":
        print("❌ Failed to download training data"); return
    test_result = download_test_data()
    if test_result["status"] != "success":
        print("❌ Failed to download test data"); return

    train_df = train_result["dataset"]
    test_df = test_result["test_data"]
    ground_truth_df = test_result["ground_truth"]

    training_examples, train_ids = sample_training_examples(train_df, NUM_EXAMPLES)
    score_range = get_score_range_for_dataset(TEST_DATASET, 1)

    results = {
        "experiment_type": "cross_dataset_generalization_inductive",
        "model_code": MODEL_CODE,
        "model_name": MODEL_NAME,
        "train_dataset": TRAIN_DATASET,
        "test_dataset": TEST_DATASET,
        "reasoning_approach": "inductive",
        "num_training_examples": NUM_EXAMPLES,
        "random_seed": RANDOM_SEED,
        "training_ids": train_ids,
        "start_timestamp": datetime.now().isoformat(),
        "test_examples": len(test_df),
        "predictions": [],
        "failed_predictions": [],
        "stats": {"valid": 0, "invalid": 0}
    }

    preds = {}
    print(f"\n🔄 Predicting {len(test_df)} test answers from {TEST_DATASET}...")

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_df)} ({100*i/len(test_df):.1f}%)")
        essay_id = str(row[CSEE_COLUMNS["id"]])
        essay_text = row[CSEE_COLUMNS["text"]]
        question = row.get(CSEE_COLUMNS["question"], "")
        result = get_prediction_with_retry(client, essay_text, question, training_examples, score_range)
        if result["success"]:
            preds[essay_id] = result["prediction"]
            results["predictions"].append({"id": essay_id, "prediction": result["prediction"]})
            results["stats"]["valid"] += 1
        else:
            results["failed_predictions"].append({"id": essay_id, "error": result["error"]})
            results["stats"]["invalid"] += 1
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    print(f"\n✅ Completed: {results['stats']['valid']} valid, {results['stats']['invalid']} invalid")
    csv_path = save_predictions_as_csv(test_df, preds)
    results["csv_output"] = csv_path

    metrics = calculate_qwk(csv_path, ground_truth_df)
    if metrics:
        results["metrics"] = metrics
        print(f"\nQWK: {metrics['qwk']:.4f} | MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f}")
    else:
        print("⚠️ Could not calculate metrics")

    results_filename = os.path.join(OUTPUT_DIR, f"results_{int(time.time())}.json")
    with open(results_filename, "w") as f: json.dump(results, f, indent=2)
    print(f"Results saved: {results_filename}")

    summary_file = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("CROSS-DATASET GENERALIZATION - INDUCTIVE ONLY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\nTraining Dataset: {TRAIN_DATASET}\nTest Dataset: {TEST_DATASET}\n")
        if metrics:
            f.write(f"QWK: {metrics['qwk']:.4f}\nMAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}\n")
        f.write(f"\nValid: {results['stats']['valid']} | Invalid: {results['stats']['invalid']}\n")
    print(f"Summary saved: {summary_file}")

    return results

# ================== ENTRYPOINT ==================
if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key provided")
        print("\nUsage: python generalization_ind_ASAPSAS_to_CSEE_gemini.py YOUR_API_KEY")
        sys.exit(1)

    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION - INDUCTIVE ONLY")
    print("="*70)
    print(f"Training from: {TRAIN_DATASET}")
    print(f"Testing on: {TEST_DATASET}")
    print(f"Model: {MODEL_NAME}")
    print(f"Approach: Inductive Only")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Output: {OUTPUT_DIR}/")
    print("="*70)

    confirm = input("\nProceed with generalization experiment? (y/n): ").strip().lower()
    if confirm == "y":
        _ = run_generalization_experiment(api_key)
    else:
        print("Cancelled.")