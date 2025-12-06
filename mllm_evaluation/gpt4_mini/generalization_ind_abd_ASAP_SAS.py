#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiment - Short Answer
Training examples: CSEE
Test dataset: ASAP-SAS
Approach: Inductive + Abductive
Model: Llama 4 Scout
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
TRAIN_DATASET = "CSEE"  # Where training examples come from
TEST_DATASET = "ASAP-SAS"  # What we're testing on
NUM_EXAMPLES = 5  # Number of training examples to sample
RANDOM_SEED = 42
TEST_DATASET_SIZE_LIMIT = 1200
MODEL_CODE = "openai/gpt-4o-mini"
MODEL_NAME = "gpt-4o-mini"


# Output directory
BASE_DIR = "generalization"
EXPERIMENT_NAME = f"{MODEL_NAME}_train_{TRAIN_DATASET}_test_{TEST_DATASET}_ind_abd"
OUTPUT_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)

# Column mappings for both datasets
CSEE_COLUMNS = {
    "id": "index",
    "text": "essay",
    "score": "overall_score",
    "question": "prompt"
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
    """Download CSEE training data"""
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
        # APPLY SIZE LIMIT FOR TEST DATASET
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
    """Sample training examples from CSEE"""
    sampled = train_df.sample(n=min(num_examples, len(train_df)), random_state=RANDOM_SEED)
    
    print(f"  ✓ Sampled {len(sampled)} training examples from {TRAIN_DATASET} (seed={RANDOM_SEED})")
    
    examples = []
    train_ids = []
    
    for _, row in sampled.iterrows():
        essay_id = str(row[CSEE_COLUMNS["id"]])
        essay_text = row[CSEE_COLUMNS["text"]]
        score = row[CSEE_COLUMNS["score"]]
        question = row.get(CSEE_COLUMNS["question"], "")
        
        # Get CSEE score range from dataset_ranges.py
        csee_range = get_score_range_for_dataset(TRAIN_DATASET, 1)
        
        examples.append({
            "id": essay_id,
            "text": essay_text,
            "score": score,
            "question": question,
            "score_range": csee_range
        })
        train_ids.append(essay_id)
    
    return examples, train_ids

def create_cross_dataset_ind_abd_prompt(essay_text: str, question: str,
                                        training_examples: List[Dict], 
                                        target_score_range: tuple) -> str:
    """
    Create Inductive + Abductive prompt for cross-dataset generalization
    Training: CSEE examples
    Testing: ASAP-SAS essays
    """
    target_min, target_max = target_score_range
    
    # Get CSEE range from dataset_ranges.py
    csee_range = get_score_range_for_dataset(TRAIN_DATASET, 1)
    source_min, source_max = csee_range
    
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
- You must learn patterns from {TRAIN_DATASET} examples but apply them to {TEST_DATASET} scoring scale

**PHASE 1 - INDUCTIVE REASONING (Learn from {TRAIN_DATASET} examples FIRST):**
Look at these {TRAIN_DATASET} training examples and identify patterns:
{examples_str}

From these {TRAIN_DATASET} examples, identify patterns in:
- What characterizes high-scoring vs low-scoring answers
- How content accuracy and completeness affect scores
- Patterns in scientific/factual reasoning
- Common strengths and weaknesses

**PHASE 2 - ABDUCTIVE REASONING (Then infer best explanation for {TEST_DATASET} answer):**
Now infer the most likely explanation for this {TEST_DATASET} answer's quality:
- OBSERVATION: What does this specific {TEST_DATASET} answer demonstrate?
- POSSIBLE EXPLANATIONS: 
  * Matches pattern of high-quality answers from {TRAIN_DATASET}
  * Shows characteristics of medium-quality answers from {TRAIN_DATASET}
  * Exhibits traits of low-quality answers from {TRAIN_DATASET}
- BEST EXPLANATION: Which quality level best accounts for this observation?

**PHASE 3 - CROSS-DATASET TRANSFER:**
Now score this {TEST_DATASET} answer:
- Use quality patterns learned from {TRAIN_DATASET} (range: {source_min}-{source_max})
- Infer the best explanation for this answer's quality (abductive)
- BUT score using the {TEST_DATASET} range: {target_min} to {target_max}

**TARGET ANSWER TO SCORE ({TEST_DATASET}):**
Question: {question}
Student Answer: {essay_text}

**SCORING INSTRUCTIONS:**
- Score range for this {TEST_DATASET} answer: {target_min} to {target_max}
- {TRAIN_DATASET} training examples used range: {source_min} to {source_max}
- FIRST learn patterns from {TRAIN_DATASET} examples (inductive)
- THEN infer the best quality explanation for this answer (abductive)
- FINALLY adapt to {TEST_DATASET} scale ({target_min}-{target_max})

STOP. Do not write steps. Do not write explanations. Do not write reasoning.
Provide ONLY a number between {target_min} and {target_max}. Just the number. Nothing else."""

    return prompt

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
                              training_examples: List[Dict], score_range: tuple,
                              max_retries: int = 3) -> Dict[str, Any]:
    """Get prediction with retry logic"""
    for attempt in range(max_retries):
        try:
            prompt = create_cross_dataset_ind_abd_prompt(
                essay_text, question, training_examples, score_range
            )
            
            response = client.chat.completions.create(
                model=MODEL_CODE,
                messages=[{"role": "user", "content": prompt}],
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
                print(f"    ⚠️  Attempt {attempt + 1}: {validation['error']}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            print(f"    ✗ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return {
        'success': False,
        'error': f"Validation failed after {max_retries} attempts",
        'last_response': response_text if 'response_text' in locals() else 'No response',
        'attempts': max_retries
    }

def save_predictions_as_csv(test_df: pd.DataFrame, predictions_map: Dict):
    """Save predictions to CSV"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_df = test_df.copy()
    
    # Add predictions
    for idx in output_df.index:
        row_id = str(output_df.loc[idx, ASAP_SAS_COLUMNS["id"]])
        if row_id in predictions_map:
            output_df.loc[idx, ASAP_SAS_COLUMNS["score"]] = predictions_map[row_id]
    
    csv_filename = f"predictions.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    
    output_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved predictions: {csv_path}")
    
    return csv_path

def calculate_qwk(csv_path: str, ground_truth_df: pd.DataFrame):
    """Calculate QWK for predictions"""
    try:
        from sklearn.metrics import cohen_kappa_score
        
        # Load predictions
        pred_df = pd.read_csv(csv_path)
        
        # Merge with ground truth
        merged = pred_df.merge(
            ground_truth_df[[ASAP_SAS_COLUMNS["id"], ASAP_SAS_COLUMNS["score"]]], 
            on=ASAP_SAS_COLUMNS["id"], 
            how='inner',
            suffixes=('_pred', '_true')
        )
        
        y_pred = merged[f'{ASAP_SAS_COLUMNS["score"]}_pred'].values
        y_true = merged[f'{ASAP_SAS_COLUMNS["score"]}_true'].values
        
        # Remove NaN values
        valid_mask = ~(np.isnan(y_pred) | np.isnan(y_true))
        y_pred = y_pred[valid_mask]
        y_true = y_true[valid_mask]
        
        if len(y_pred) == 0:
            print("  ❌ No valid predictions for QWK calculation")
            return None
        
        # Calculate metrics
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

def run_generalization_experiment(api_key):
    """Main experiment function"""
    client = get_client(api_key)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}/")
    
    print("\n" + "="*70)
    print(f"CROSS-DATASET GENERALIZATION EXPERIMENT - SHORT ANSWER")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Approach: Inductive + Abductive")
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
    
    # Sample training examples from CSEE
    training_examples, train_ids = sample_training_examples(train_df, NUM_EXAMPLES)
    
    # Get ASAP-SAS score range
    asap_sas_range = get_score_range_for_dataset(TEST_DATASET, 1)
    
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
        'test_examples': len(test_df),
        'predictions': [],
        'failed_predictions': [],
        'stats': {'valid': 0, 'invalid': 0}
    }
    
    predictions_map = {}
    
    print(f"\n🔄 Processing {len(test_df)} test answers from {TEST_DATASET}...")
    print(f"   Using {NUM_EXAMPLES} training examples from {TRAIN_DATASET}")
    print(f"   Reasoning: Inductive + Abductive")
    
    # Process each test example
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_df)} ({100*i/len(test_df):.1f}%)")
        
        essay_id = str(row[ASAP_SAS_COLUMNS["id"]])
        essay_text = row[ASAP_SAS_COLUMNS["text"]]
        question = row.get(ASAP_SAS_COLUMNS["question"], "")
        
        result = get_prediction_with_retry(
            client, essay_text, question, training_examples, asap_sas_range
        )
        
        if result['success']:
            pred_entry = {
                'id': essay_id,
                'prediction': result['prediction'],
                'tokens': result['tokens'],
                'attempts': result['attempts']
            }
            
            results['predictions'].append(pred_entry)
            results['stats']['valid'] += 1
            predictions_map[essay_id] = result['prediction']
        else:
            fail_entry = {
                'id': essay_id,
                'error': result['error'],
                'attempts': result['attempts']
            }
            if 'last_response' in result:
                fail_entry['last_response'] = result['last_response'][:100]
            
            results['failed_predictions'].append(fail_entry)
            results['stats']['invalid'] += 1
        
        time.sleep(2)  # Rate limiting
    
    results['end_timestamp'] = datetime.now().isoformat()
    
    print(f"\n✅ Prediction Complete:")
    print(f"   Valid: {results['stats']['valid']}")
    print(f"   Invalid: {results['stats']['invalid']}")
    
    # Save predictions to CSV
    print(f"\n💾 Saving predictions...")
    csv_path = save_predictions_as_csv(test_df, predictions_map)
    results['csv_output'] = csv_path
    
    # Calculate QWK
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
    
    # Create summary file
    summary_filename = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_filename, "w") as f:
        f.write("="*70 + "\n")
        f.write("CROSS-DATASET GENERALIZATION - SHORT ANSWER\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Approach: Inductive + Abductive\n")
        f.write(f"Training Dataset: {TRAIN_DATASET}\n")
        f.write(f"Test Dataset: {TEST_DATASET}\n")
        f.write(f"Training Examples: {NUM_EXAMPLES}\n")
        f.write(f"Random Seed: {RANDOM_SEED}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        if metrics:
            f.write(f"QWK:  {metrics['qwk']:.4f}\n")
            f.write(f"MAE:  {metrics['mae']:.4f}\n")
            f.write(f"RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"Valid Predictions: {metrics['num_predictions']}\n")
        else:
            f.write("Metrics: Could not be calculated\n")
        
        f.write(f"\nValid Predictions: {results['stats']['valid']}\n")
        f.write(f"Failed Predictions: {results['stats']['invalid']}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write(f"TRAINING EXAMPLES USED (from {TRAIN_DATASET})\n")
        f.write("-"*70 + "\n")
        for train_id in train_ids:
            f.write(f"  {train_id}\n")
    
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
        print("\nUsage: python run_generalization_csee_asap_sas.py YOUR_API_KEY")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION - SHORT ANSWER")
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
        result = run_generalization_experiment(api_key)
        if result and 'metrics' in result:
            print("\n✅ Experiment completed successfully!")
            print(f"   Final QWK: {result['metrics']['qwk']:.4f}")
        else:
            print("\n⚠️  Experiment completed with issues")
    else:
        print("Cancelled")