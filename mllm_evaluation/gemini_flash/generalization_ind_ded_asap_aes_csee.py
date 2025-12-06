#!/usr/bin/env python3
"""
Cross-Dataset Generalization Experiment
Training examples: CSEE
Test dataset: ASAP-AES
Approach: Inductive + Deductive
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
TEST_DATASET = "ASAP-AES"  # What we're testing on
NUM_EXAMPLES = 5  # Number of training examples to sample
RANDOM_SEED = 42

MODEL_CODE = "google/gemini-2.5-flash"
MODEL_NAME = "gemini-2.5-flash"

# Output directory
BASE_DIR = "generalization"
EXPERIMENT_NAME = f"{MODEL_NAME}_train_{TRAIN_DATASET}_test_{TEST_DATASET}"
OUTPUT_DIR = os.path.join(BASE_DIR, EXPERIMENT_NAME)

# Column mappings for both datasets
CSEE_COLUMNS = {
    "id": "index",
    "text": "essay",
    "score": "overall_score",
    "question": "prompt"
}

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
    """Download ASAP-AES test data"""
    print(f"  Downloading test data from {TEST_DATASET}...")
    try:
        dataset = load_dataset(f"nlpatunt/{TEST_DATASET}", split="test", trust_remote_code=True)
        df = dataset.to_pandas()
        
        print(f"    ✓ Downloaded: {len(df)} rows from {TEST_DATASET}")
        
        # Keep ground truth separate for evaluation
        test_df_for_prediction = df.drop(columns=[ASAP_AES_COLUMNS["score"]], errors='ignore')
        
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
        # CSEE doesn't have essay_set column, so use default essay_set=1
        csee_range = get_score_range_for_dataset(TRAIN_DATASET, 1)
        
        examples.append({
            "id": essay_id,
            "text": essay_text,
            "score": score,
            "question": question,
            "score_range": csee_range  # Store the range from dataset_ranges.py
        })
        train_ids.append(essay_id)
    
    return examples, train_ids

def create_cross_dataset_prompt(essay_text: str, question: str,
                                training_examples: List[Dict], 
                                target_score_range: tuple,
                                essay_set: int,
                                retry_context: Dict = None) -> str:
    """
    Create prompt that clearly specifies:
    1. Training examples are from CSEE (with CSEE score range from dataset_ranges.py)
    2. Target essay is from ASAP-AES (with ASAP-AES score range from dataset_ranges.py)
    3. Model must adapt patterns across different scales
    4. Adaptive: Gets progressively more explicit with each retry
    """
    target_min, target_max = target_score_range
    
    # Get CSEE range from dataset_ranges.py
    csee_range = get_score_range_for_dataset(TRAIN_DATASET, 1)
    
    # Handle if CSEE is a string
    if isinstance(csee_range, str):
        csee_range = (0, 3)
    
    source_min, source_max = csee_range
    
    # Determine retry level for adaptive prompting
    attempt_num = retry_context.get('attempt', 1) if retry_context else 1
    previous_score = retry_context.get('previous_score') if retry_context else None
    previous_error = retry_context.get('previous_error') if retry_context else None
    
    prompt = f"""You are an expert essay grader performing CROSS-DATASET GENERALIZATION.

**IMPORTANT CONTEXT:**
- The training examples below are from CSEE dataset (score range: {source_min}-{source_max})
- The target essay to score is from ASAP-AES dataset (score range: {target_min}-{target_max})
- You must learn patterns from CSEE examples but apply them to ASAP-AES scoring scale

**PHASE 1: INDUCTIVE REASONING (Learn from CSEE Examples)**
Analyze these CSEE examples to identify quality patterns:

"""
    
    for idx, example in enumerate(training_examples, 1):
        ex_question = example.get('question', '')
        prompt += f"""Example {idx} (from CSEE, range {source_min}-{source_max}):
Question: {ex_question}
Response: {example['text'][:500]}...
Score: {example['score']} (on {source_min}-{source_max} scale)

"""
    
    prompt += f"""From these CSEE examples, identify:
- What distinguishes high-scoring from low-scoring responses
- Key quality indicators (content, organization, depth, clarity)
- Common patterns in excellent vs. poor responses

**PHASE 2: DEDUCTIVE REASONING (Apply General Principles)**
Apply universal academic grading principles:
- Content accuracy and completeness
- Organization and logical structure
- Use of evidence and examples
- Language quality and clarity
- Depth of analysis and critical thinking

**PHASE 3: CROSS-DATASET TRANSFER**
Now score this ASAP-AES essay:
- Use the quality patterns learned from CSEE (range: {source_min}-{source_max})
- Apply general grading principles
- BUT score using the ASAP-AES range: {target_min} to {target_max}

**TARGET ESSAY TO SCORE (ASAP-AES, Essay Set {essay_set}):**
Question: {question}
Student Response: {essay_text}

"""
    
    # ADAPTIVE PROMPTING: Get more explicit with each retry
    if attempt_num == 1:
        # First attempt: Standard instructions
        prompt += f"""**SCORING INSTRUCTIONS:**
- Score range for this ASAP-AES essay: {target_min} to {target_max}
- CSEE training examples used range: {source_min} to {source_max}
- Adapt the quality patterns from CSEE scale to ASAP-AES scale
- Provide ONLY the numeric score (e.g., "{target_min}", "{target_max}", or a value between)
- No explanation needed - just the number"""

    elif attempt_num == 2:
        # Second attempt: Add warning about previous failure
        prompt += f"""⚠️ **RETRY ATTEMPT {attempt_num}** ⚠️
Your previous score ({previous_score}) was INVALID: {previous_error}

**CRITICAL REQUIREMENTS:**
- MINIMUM valid score: {target_min} (anything below is INVALID)
- MAXIMUM valid score: {target_max} (anything above is INVALID)
- You MUST score between {target_min} and {target_max} (inclusive)
- You are scoring ASAP-AES (range {target_min}-{target_max}), NOT CSEE (range {source_min}-{source_max})

Provide ONLY a numeric score between {target_min} and {target_max}."""

    elif attempt_num == 3:
        # Third attempt: Add scale mapping examples
        mid_point = (target_min + target_max) / 2
        prompt += f"""🚨 **RETRY ATTEMPT {attempt_num} - SCALE MAPPING HELP** 🚨
Your previous score ({previous_score}) was INVALID: {previous_error}

**YOU ARE CONFUSING THE TWO SCALES!**
- CSEE uses range {source_min}-{source_max} (training examples)
- ASAP-AES uses range {target_min}-{target_max} (what you need to score)

**Scale Mapping Guide:**
- Low quality → Score near {target_min} (NOT {source_min})
- Medium quality → Score near {mid_point:.1f}
- High quality → Score near {target_max} (NOT {source_max})

**Your score MUST be a number between {target_min} and {target_max}.**
Provide ONLY that number."""

    else:
        # Attempt 4+: Maximum explicitness
        prompt += f"""🚨🚨 **FINAL RETRY ATTEMPT {attempt_num}** 🚨🚨
You have failed {attempt_num - 1} times. Your last score was {previous_score}.

**THE PROBLEM:** You predicted {previous_score} but the valid range is [{target_min}, {target_max}]

**WHAT YOU MUST DO:**
1. Look at the essay quality
2. If quality is LOW → use a score close to {target_min}
3. If quality is MEDIUM → use a score around {(target_min + target_max) / 2:.1f}
4. If quality is HIGH → use a score close to {target_max}

**DO NOT:**
- Use scores below {target_min}
- Use scores above {target_max}
- Use scores from the CSEE range ({source_min}-{source_max})

**OUTPUT:** One number between {target_min} and {target_max}. Nothing else."""

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
                              essay_set: int, max_retries: int = 5) -> Dict[str, Any]:
    """Get prediction with adaptive retry prompts - NO CLAMPING"""
    previous_score = None
    previous_error = None
    
    for attempt in range(max_retries):
        try:
            # Create retry context for adaptive prompting
            retry_context = {
                'attempt': attempt + 1,
                'previous_score': previous_score,
                'previous_error': previous_error
            }
            
            prompt = create_cross_dataset_prompt(
                essay_text, question, training_examples, score_range, essay_set, retry_context
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
                # Store error for next retry's adaptive prompt
                previous_error = validation['error']
                
                # Extract the predicted score even if invalid
                matches = re.findall(r'[-+]?\d*\.?\d+', response_text)
                if matches:
                    previous_score = float(matches[0])
                else:
                    previous_score = "non-numeric"
                
                print(f"    ⚠️  Attempt {attempt + 1}/{max_retries}: {validation['error']}")
                
                if attempt < max_retries - 1:
                    time.sleep(1)
        except Exception as e:
            print(f"    ✗ Attempt {attempt + 1}/{max_retries} API error: {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    # NO CLAMPING - Let the failure be a failure
    return {
        'success': False,
        'error': f"Failed after {max_retries} attempts with adaptive prompts",
        'last_response': response_text if 'response_text' in locals() else 'No response',
        'last_predicted_score': previous_score,
        'last_error': previous_error,
        'attempts': max_retries
    }

def save_predictions_as_csv(test_df: pd.DataFrame, predictions_map: Dict):
    """Save predictions to CSV"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_df = test_df.copy()
    
    # Add predictions
    for idx in output_df.index:
        row_id = str(output_df.loc[idx, ASAP_AES_COLUMNS["id"]])
        if row_id in predictions_map:
            output_df.loc[idx, ASAP_AES_COLUMNS["score"]] = predictions_map[row_id]
    
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
            ground_truth_df[[ASAP_AES_COLUMNS["id"], ASAP_AES_COLUMNS["score"]]], 
            on=ASAP_AES_COLUMNS["id"], 
            how='inner',
            suffixes=('_pred', '_true')
        )
        
        y_pred = merged[f'{ASAP_AES_COLUMNS["score"]}_pred'].values
        y_true = merged[f'{ASAP_AES_COLUMNS["score"]}_true'].values
        
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
    print(f"CROSS-DATASET GENERALIZATION WITH ADAPTIVE RETRY")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Approach: Inductive + Deductive with Adaptive Retry Prompts")
    print(f"Training dataset: {TRAIN_DATASET}")
    print(f"Test dataset: {TEST_DATASET}")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Max retries: 5 (with progressively explicit prompts)")
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
    
    # Initialize results
    results = {
        'experiment_type': 'cross_dataset_generalization_adaptive_retry',
        'model_code': MODEL_CODE,
        'model_name': MODEL_NAME,
        'train_dataset': TRAIN_DATASET,
        'test_dataset': TEST_DATASET,
        'num_training_examples': NUM_EXAMPLES,
        'random_seed': RANDOM_SEED,
        'training_ids': train_ids,
        'start_timestamp': datetime.now().isoformat(),
        'test_examples': len(test_df),
        'predictions': [],
        'failed_predictions': [],
        'stats': {
            'valid': 0, 
            'invalid': 0,
            'failed_by_essay_set': {},  # Track failures per essay set
            'common_failure_reasons': {}  # Track common error patterns
        }
    }
    
    predictions_map = {}
    
    print(f"\n🔄 Processing {len(test_df)} test essays from {TEST_DATASET}...")
    print(f"   Using {NUM_EXAMPLES} training examples from {TRAIN_DATASET}")
    
    # Process each test example
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_df)} ({100*i/len(test_df):.1f}%)")
        
        essay_id = str(row[ASAP_AES_COLUMNS["id"]])
        essay_text = row[ASAP_AES_COLUMNS["text"]]
        question = row.get(ASAP_AES_COLUMNS["question"], "")
        essay_set = int(row[ASAP_AES_COLUMNS["essay_set"]])
        
        # Get ASAP-AES score range for this essay
        score_range = get_score_range_for_dataset(TEST_DATASET, essay_set)
        
        result = get_prediction_with_retry(
            client, essay_text, question, training_examples, score_range, essay_set
        )
        
        if result['success']:
            pred_entry = {
                'id': essay_id,
                'prediction': result['prediction'],
                'tokens': result['tokens'],
                'essay_set': essay_set,
                'attempts': result['attempts']
            }
            
            results['predictions'].append(pred_entry)
            results['stats']['valid'] += 1
            predictions_map[essay_id] = result['prediction']
        else:
            # Detailed failure tracking
            fail_entry = {
                'id': essay_id,
                'error': result['error'],
                'attempts': result['attempts'],
                'essay_set': essay_set,
                'score_range': score_range,
                'last_predicted_score': result.get('last_predicted_score'),
                'last_error': result.get('last_error')
            }
            if 'last_response' in result:
                fail_entry['last_response'] = result['last_response'][:200]
            
            results['failed_predictions'].append(fail_entry)
            results['stats']['invalid'] += 1
            
            # Track failures by essay set
            essay_set_key = f"set_{essay_set}"
            if essay_set_key not in results['stats']['failed_by_essay_set']:
                results['stats']['failed_by_essay_set'][essay_set_key] = 0
            results['stats']['failed_by_essay_set'][essay_set_key] += 1
            
            # Track common failure patterns
            if result.get('last_error'):
                error_key = result['last_error'][:50]  # First 50 chars of error
                if error_key not in results['stats']['common_failure_reasons']:
                    results['stats']['common_failure_reasons'][error_key] = 0
                results['stats']['common_failure_reasons'][error_key] += 1
        
        time.sleep(2)  # Rate limiting
    
    results['end_timestamp'] = datetime.now().isoformat()
    
    print(f"\n✅ Prediction Complete:")
    print(f"   Valid: {results['stats']['valid']}/{len(test_df)} ({100*results['stats']['valid']/len(test_df):.1f}%)")
    print(f"   Invalid: {results['stats']['invalid']}/{len(test_df)} ({100*results['stats']['invalid']/len(test_df):.1f}%)")
    
    if results['stats']['invalid'] > 0:
        print(f"\n❌ Failure Analysis:")
        print(f"   Failed after 5 adaptive retry attempts each")
        
        # Show failures by essay set
        if results['stats']['failed_by_essay_set']:
            print(f"\n   Failures by Essay Set:")
            for essay_set_key in sorted(results['stats']['failed_by_essay_set'].keys()):
                count = results['stats']['failed_by_essay_set'][essay_set_key]
                print(f"     {essay_set_key}: {count} failures")
        
        # Show common failure reasons
        if results['stats']['common_failure_reasons']:
            print(f"\n   Common Failure Patterns:")
            sorted_reasons = sorted(results['stats']['common_failure_reasons'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for reason, count in sorted_reasons:
                print(f"     [{count}x] {reason}")
    
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
        f.write("CROSS-DATASET GENERALIZATION EXPERIMENT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Approach: Inductive + Deductive\n")
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
        
        f.write(f"\nValid Predictions: {results['stats']['valid']}/{len(test_df)} ({100*results['stats']['valid']/len(test_df):.1f}%)\n")
        f.write(f"Failed Predictions: {results['stats']['invalid']}/{len(test_df)} ({100*results['stats']['invalid']/len(test_df):.1f}%)\n")
        
        # Add failure analysis if there are failures
        if results['stats']['invalid'] > 0:
            f.write("\n" + "-"*70 + "\n")
            f.write("FAILURE ANALYSIS\n")
            f.write("-"*70 + "\n")
            f.write(f"All failures occurred after 5 adaptive retry attempts\n\n")
            
            # Failures by essay set
            if results['stats']['failed_by_essay_set']:
                f.write("Failures by Essay Set:\n")
                for essay_set_key in sorted(results['stats']['failed_by_essay_set'].keys()):
                    count = results['stats']['failed_by_essay_set'][essay_set_key]
                    f.write(f"  {essay_set_key}: {count} failures\n")
            
            # Common failure patterns
            if results['stats']['common_failure_reasons']:
                f.write("\nMost Common Failure Patterns:\n")
                sorted_reasons = sorted(results['stats']['common_failure_reasons'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]
                for reason, count in sorted_reasons:
                    f.write(f"  [{count}x] {reason}\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("TRAINING EXAMPLES USED (from CSEE)\n")
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
        print("\nUsage: python run_generalization_experiment.py YOUR_API_KEY")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("CROSS-DATASET GENERALIZATION WITH ADAPTIVE RETRY")
    print("="*70)
    print(f"Training from: {TRAIN_DATASET}")
    print(f"Testing on: {TEST_DATASET}")
    print(f"Model: {MODEL_NAME}")
    print(f"Approach: Inductive + Deductive with Adaptive Retry")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Max retries: 5 (progressively more explicit)")
    print(f"No score clamping - failures tracked for analysis")
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