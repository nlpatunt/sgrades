#!/usr/bin/env python3
"""
Few-Shot Inductive with 3-Call Aggregation - CORRECTED VERSION
- Training data: HuggingFace WITHOUT D_ prefix (WITH labels)
- Test data: Local CSV WITH D_ prefix (WITHOUT labels, 21% sample)
"""

import os
import json
import time
import sys
import re
import pandas as pd
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_ranges import get_score_range_for_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_EXAMPLES = 5
RANDOM_SEED = 42
NUM_CALLS_PER_DATAPOINT = 3
NUM_TEST_SAMPLES = None  # Set to None for full dataset

MODEL_CODE = "meta-llama/llama-4-scout"
MODEL_NAME = "llama-4-scout"

# Paths
SAM_DATASETS_DIR = "/home/ts1506.UNT/Desktop/Work/besisr-benchmark-site/mllm_evaluation/sam_datasets"
OUTPUT_DIR = "inductive_3call_predictions"

os.environ["HF_TOKEN"] = "REMOVED_KEY"

# ============================================================================
# DATASETS
# ============================================================================
DATASETS = [
    # 'D_ASAP-AES', 'D_ASAP2', 'D_ASAP_plus_plus', 'D_persuade_2',
    # 'D_Ielts_Writing_Dataset', 'D_Ielts_Writing_Task_2_Dataset',
    # 'D_Regrading_Dataset_J2C', 'D_ASAP-SAS', 'D_CSEE', 'D_Mohlar',
    # 'D_BEEtlE_2way', 'D_BEEtlE_3way',  # Separate 2way/3way
    # 'D_SciEntSBank_2way', 'D_SciEntSBank_3way',  # Separate 2way/3way
    # 'D_OS_Dataset_q1', 'D_OS_Dataset_q2', 'D_OS_Dataset_q3',
    # 'D_OS_Dataset_q4', 'D_OS_Dataset_q5',
    # 'D_Rice_Chem_Q1', 'D_Rice_Chem_Q2', 'D_Rice_Chem_Q3', 'D_Rice_Chem_Q4'
    'D_SciEntSBank_2way', 'D_SciEntSBank_3way'
]

def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    """Get column names - using NORMALIZED name (without D_)"""
    norm = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
    configs = {
        "ASAP-AES": {"id": "essay_id", "text": "essay", "score": "domain1_score", "question": "prompt"},
        "ASAP2": {"id": "essay_id", "text": "full_text", "score": "score", "question": "assignment"},
        "ASAP-SAS": {"id": "Id", "text": "essay_text", "score": "Score1", "question": "prompt"},
        "ASAP_plus_plus": {"id": "essay_id", "text": "essay", "score": "overall_score", "question": "prompt"},
        "BEEtlE_2way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text"},
        "BEEtlE_3way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text"},
        "SciEntSBank_2way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text"},
        "SciEntSBank_3way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text"},
        "CSEE": {"id": "index", "text": "essay", "score": "overall_score", "question": "prompt"},
        "Mohlar": {"id": "ID", "text": "student_answer", "score": "grade", "question": "Question"},
        "Ielts_Writing_Dataset": {"id": "ID", "text": "Essay", "score": "Overall_Score", "question": "Question"},
        "Ielts_Writing_Task_2_Dataset": {"id": "ID", "text": "essay", "score": "band_score", "question": "prompt"},
        "persuade_2": {"id": "essay_id_comp", "text": "full_text", "score": "holistic_essay_score", "question": "assignment"},
        "Regrading_Dataset_J2C": {"id": "ID", "text": "student_answer", "score": "grade", "question": "Question"},
        "OS_Dataset_q1": {"id": "ID", "text": "answer", "score": "score_1", "question": "question"},
        "OS_Dataset_q2": {"id": "ID", "text": "answer", "score": "score_1", "question": "question"},
        "OS_Dataset_q3": {"id": "ID", "text": "answer", "score": "score_1", "question": "question"},
        "OS_Dataset_q4": {"id": "ID", "text": "answer", "score": "score_1", "question": "question"},
        "OS_Dataset_q5": {"id": "ID", "text": "answer", "score": "score_1", "question": "question"},
        "Rice_Chem_Q1": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt"},
        "Rice_Chem_Q2": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt"},
        "Rice_Chem_Q3": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt"},
        "Rice_Chem_Q4": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt"},
    }
    return configs.get(norm, {"id": "id", "text": "text", "score": "score", "question": "question"})

# ============================================================================
# LOAD TRAINING DATA FROM HUGGINGFACE (WITHOUT D_ PREFIX, WITH LABELS)
# ============================================================================

def load_training_data(dataset_name: str) -> pd.DataFrame:
    """
    Load training data from HuggingFace (WITHOUT D_ prefix, WITH labels)
    Uses your old working logic
    """
    from datasets import load_dataset
    
    # Remove D_ prefix
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
    print(f"  Loading training data for {normalized_name}...")
    
    try:
        if normalized_name in ["BEEtlE_2way", "BEEtlE_3way"]:
            suffix = "2way" if "2way" in normalized_name else "3way"
            dataset = load_dataset("nlpatunt/BEEtlE", data_files=f"train_{suffix}.csv", trust_remote_code=True)["train"]
        
        elif normalized_name in ["SciEntSBank_2way", "SciEntSBank_3way"]:
            suffix = "2way" if "2way" in normalized_name else "3way"
            dataset = load_dataset("nlpatunt/SciEntSBank", data_files=f"train_{suffix}.csv", trust_remote_code=True)["train"]
        
        elif normalized_name in ["Rice_Chem_Q1", "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"]:
            q_num = normalized_name.split("_")[-1]  # Q1, Q2, Q3, Q4
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files=f"{q_num}/train.csv", trust_remote_code=True)["train"]
        
        elif normalized_name.startswith("OS_Dataset"):

            q_num = normalized_name.split("_q")[-1]  # "1", "2", "3", "4", "5"
            dataset = load_dataset("nlpatunt/OS_Dataset", data_files=f"q{q_num}/train.csv", trust_remote_code=True)["train"]
            df = dataset.to_pandas()
            df = df[df['question_id'] == q_num].reset_index(drop=True)
            print(f"  ✓ Loaded {len(df)} training examples (filtered q{q_num})")
            return df
        
        elif normalized_name == "persuade_2":
            dataset = load_dataset("nlpatunt/persuade_2", data_files="train.csv", trust_remote_code=True)["train"]
        
        elif normalized_name == "Mohlar":
            dataset = load_dataset("nlpatunt/Mohlar", data_files="train.csv", trust_remote_code=True)["train"]
        
        elif normalized_name == "ASAP-SAS":
            dataset = load_dataset("nlpatunt/ASAP-SAS", data_files="train.csv", trust_remote_code=True)["train"]
        
        else:
            # Default: try loading normally
            try:
                dataset = load_dataset(f"nlpatunt/{normalized_name}", split="train", trust_remote_code=True)
            except:
                dataset = load_dataset(f"nlpatunt/{normalized_name}", trust_remote_code=True)
                if hasattr(dataset, 'keys'):
                    first_split = list(dataset.keys())[0]
                    dataset = dataset[first_split]
        
        df = dataset.to_pandas()
        print(f"  ✓ Loaded {len(df)} training examples")
        return df
        
    except Exception as e:
        print(f"  ✗ Error loading training data: {e}")
        raise

# ============================================================================
# LOAD TEST DATA FROM LOCAL CSV (WITH D_ PREFIX, WITHOUT LABELS)
# ============================================================================

def load_test_data_from_csv(dataset_name: str) -> pd.DataFrame:
    """
    Load test data from local CSV (WITH D_ prefix)
    dataset_name: D_ASAP-AES
    File: D_ASAP-AES_test_21pct.csv
    """
    csv_filename = f"{dataset_name}.csv"
    csv_path = os.path.join(SAM_DATASETS_DIR, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # TEST MODE: Limit samples if specified
    if NUM_TEST_SAMPLES is not None:
        df = df.head(NUM_TEST_SAMPLES)
        print(f"  ✓ TEST MODE: Using only {len(df)} samples from {csv_filename}")
    else:
        print(f"  ✓ Loaded {len(df)} test samples from {csv_filename}")
    
    return df

# ============================================================================
# FEW-SHOT SAMPLING
# ============================================================================

def sample_training_examples(train_df: pd.DataFrame, dataset_name: str, n: int = 5):
    """Sample N training examples"""
    cols = get_dataset_columns(dataset_name)
    
    # Handle missing score column gracefully
    if cols["score"] not in train_df.columns:
        print(f"  ⚠️ Warning: Score column '{cols['score']}' not found")
        print(f"  Available columns: {list(train_df.columns)}")
        # Try alternative score columns
        alternatives = ['score', 'Score', 'grade', 'label']
        for alt in alternatives:
            if alt in train_df.columns:
                print(f"  Using alternative: '{alt}'")
                cols["score"] = alt
                break
    
    train_clean = train_df.dropna(subset=[cols["score"]])
    
    if len(train_clean) < n:
        n = len(train_clean)
    
    sampled = train_clean.sample(n=n, random_state=RANDOM_SEED)
    
    examples = []
    for _, row in sampled.iterrows():
        examples.append({
            "question": row.get(cols["question"], ""),
            "text": row.get(cols["text"], ""),
            "score": row.get(cols["score"], "")
        })
    
    return examples

# ============================================================================
# PROMPT CREATION (EXACT COPY FROM YOUR ORIGINAL)
# ============================================================================

def create_fewshot_inductive_prompt(essay_text: str, question: str, 
                                     training_examples: List[Dict], 
                                     dataset_name: str,
                                     score_range: tuple) -> Dict[str, str]:
    
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    is_3way = '3way' in normalized_name.lower()
    is_2way = '2way' in normalized_name.lower()
    
    # Build examples string
    examples_str = ""
    for i, ex in enumerate(training_examples, 1):
        examples_str += f"\nEXAMPLE {i}:\n"
        if ex["question"]:
            examples_str += f"Question: {ex['question']}\n"
        examples_str += f"Student Answer: {ex['text']}\n"
        examples_str += f"Score: {ex['score']}\n"
    
    if is_3way:
        system_prompt = f"""You are an expert evaluator using INDUCTIVE REASONING for answer classification.

INDUCTIVE PROCESS:
1. Learn classification patterns from the examples below
2. Identify what distinguishes correct, contradictory, and incorrect answers
3. Apply these learned patterns to classify the new answer

CLASSIFICATION EXAMPLES FROM TRAINING DATA:
{examples_str}

From these examples, identify patterns in:
- CORRECT: Accurate answer with sound scientific reasoning
- CONTRADICTORY: Answer that directly contradicts established scientific principles
- INCORRECT: Wrong answer based on misunderstanding or incomplete knowledge

TASK: Classify the student answer based on the patterns you learned."""

        user_prompt = f"""Based on the patterns you learned, classify this student answer:

QUESTION:
{question}

STUDENT ANSWER:
{essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.

Your ENTIRE response must be EXACTLY ONE of these three words:
correct
contradictory
incorrect

Nothing else. Just the word."""

    elif is_2way:
        system_prompt = f"""You are an expert evaluator using INDUCTIVE REASONING for answer classification.

INDUCTIVE PROCESS:
1. Learn classification patterns from the examples below
2. Identify what distinguishes correct from incorrect answers
3. Apply these learned patterns to classify the new answer

CLASSIFICATION EXAMPLES FROM TRAINING DATA:
{examples_str}

From these examples, identify patterns in:
- What makes an answer CORRECT vs INCORRECT
- How completeness and accuracy affect classification

TASK: Classify the student answer based on the patterns you learned."""

        user_prompt = f"""Based on the patterns you learned, classify this student answer:

QUESTION:
{question}

STUDENT ANSWER:
{essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.

Your ENTIRE response must be EXACTLY ONE of these two words:
correct
incorrect

Nothing else. Just the word."""

    else:
        system_prompt = f"""You are an expert essay scorer using INDUCTIVE REASONING.

INDUCTIVE PROCESS:
1. Learn scoring patterns from the examples below
2. Identify scoring criteria from the example patterns
3. Apply these learned patterns to score the new essay

SCORING EXAMPLES FROM TRAINING DATA:
{examples_str}

From these examples, identify patterns in:
- What makes a high score vs low score
- How content quality affects scoring
- What level of development is expected

SCORING RANGE: {score_range}
TASK: Score the essay based on the patterns you learned."""

        user_prompt = f"""Based on the patterns you learned from the examples above, score this essay:

QUESTION/PROMPT:
{question}

ESSAY TO SCORE:
{essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.

Your ENTIRE response must be EXACTLY one number between {score_range[0]} and {score_range[1]}.

Nothing else. Just the number."""

    return {"system": system_prompt, "user": user_prompt}

# ============================================================================
# API & VALIDATION
# ============================================================================

def get_client(api_key):
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def call_api(client, model_code: str, messages: list):
    """Single API call"""
    try:
        response = client.chat.completions.create(
            model=model_code,
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
        return {
            'success': True,
            'text': response.choices[0].message.content.strip(),
            'tokens': response.usage.total_tokens if response.usage else 0
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def validate_prediction(text: str, dataset_name: str, score_range: tuple):
    """Validate and extract prediction"""
    text_clean = text.strip().lower()
    
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    is_3way = '3way' in normalized_name.lower()
    is_2way = '2way' in normalized_name.lower()
    
    if is_3way:
        # Check 'incorrect' BEFORE 'correct' (substring issue)
        if 'incorrect' in text_clean:
            return {'valid': True, 'value': 'incorrect'}
        elif 'contradictory' in text_clean:
            return {'valid': True, 'value': 'contradictory'}
        elif 'correct' in text_clean:
            return {'valid': True, 'value': 'correct'}
        else:
            return {'valid': False, 'error': 'Invalid 3-way'}
    
    elif is_2way:
        if 'incorrect' in text_clean:
            return {'valid': True, 'value': 'incorrect'}
        elif 'correct' in text_clean:
            return {'valid': True, 'value': 'correct'}
        else:
            return {'valid': False, 'error': 'Invalid 2-way'}
    
    else:
        # Numeric
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text_clean)
        if not numbers:
            return {'valid': False, 'error': 'No number found'}
        
        try:
            score = float(numbers[0])
            if score_range[0] <= score <= score_range[1]:
                return {'valid': True, 'value': score}
            else:
                return {'valid': False, 'error': 'Out of range'}
        except:
            return {'valid': False, 'error': 'Parse error'}

# ============================================================================
# 3-CALL AGGREGATION
# ============================================================================

def aggregate_predictions(predictions: List, dataset_name: str):
    """Aggregate 3 predictions"""
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    is_classification = '2way' in normalized_name.lower() or '3way' in normalized_name.lower()
    
    if is_classification:
        counter = Counter(predictions)
        if len(set(predictions)) == 3:
            return 'uncertain'
        else:
            return counter.most_common(1)[0][0]
    else:
        return sum(predictions) / len(predictions)

def get_3call_prediction(client, model_code, essay_text, question,
                         training_examples, dataset_name, score_range):
    """Call model 3 times and aggregate"""
    predictions = []
    raw_responses = []
    total_tokens = 0
    
    prompt_data = create_fewshot_inductive_prompt(
        essay_text, question, training_examples, dataset_name, score_range
    )
    
    messages = [
        {"role": "system", "content": prompt_data["system"]},
        {"role": "user", "content": prompt_data["user"]}
    ]
    
    # Make 3 calls
    for call_num in range(NUM_CALLS_PER_DATAPOINT):
        result = call_api(client, model_code, messages)
        
        if not result['success']:
            print(f"        Call {call_num+1}/3: API failed")
            continue
        
        validation = validate_prediction(result['text'], dataset_name, score_range)
        
        if validation['valid']:
            predictions.append(validation['value'])
            raw_responses.append(result['text'])
            total_tokens += result.get('tokens', 0)
        else:
            print(f"        Call {call_num+1}/3: Invalid - {validation['error']}")
        
        time.sleep(1)
    
    if len(predictions) == 0:
        return {'success': False, 'error': 'All 3 calls failed'}
    
    final_prediction = aggregate_predictions(predictions, dataset_name)
    
    return {
        'success': True,
        'prediction': final_prediction,
        'individual_predictions': predictions,
        'raw_responses': raw_responses,
        'num_valid_calls': len(predictions),
        'tokens': total_tokens
    }

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation(api_key):
    """Main evaluation loop"""
    client = get_client(api_key)
  
    print("="*70)
    print(f"INDUCTIVE 3-CALL AGGREGATION - {MODEL_NAME}")
    print("="*70)
    print(f"Calls per data point: {NUM_CALLS_PER_DATAPOINT}")
    print(f"Training examples: {NUM_EXAMPLES}")
    print(f"Test samples per dataset: {NUM_TEST_SAMPLES if NUM_TEST_SAMPLES else 'ALL'}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {
        'model': MODEL_NAME,
        'method': 'fewshot_inductive_3call',
        'calls_per_point': NUM_CALLS_PER_DATAPOINT,
        'datasets': []
    }
    
    for dataset_name in DATASETS:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        csv_filename = f"{dataset_name}.csv"
        csv_path = os.path.join(SAM_DATASETS_DIR, csv_filename)
        try:
            # Load training data
            train_df = load_training_data(dataset_name)
            training_examples = sample_training_examples(train_df, dataset_name, NUM_EXAMPLES)
            print(f"  ✓ Sampled {len(training_examples)} training examples")
            
            # Load test data from CSV
            test_df = load_test_data_from_csv(dataset_name)
            cols = get_dataset_columns(dataset_name)
            
            predictions_map = {}
            failed = []
            
            # Process each test sample
            for i, (_, row) in enumerate(test_df.iterrows(), 1):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(test_df)}")
                
                essay_id = str(row[cols["id"]])
                essay_text = row[cols["text"]]
                question = row.get(cols["question"], "")
                essay_set = 1
                if "essay_set" in row and pd.notna(row["essay_set"]):
                    essay_set = int(row["essay_set"])

                # dataset_name can have D_ prefix, function handles both
                score_range = get_score_range_for_dataset(dataset_name, essay_set)
                
                result = get_3call_prediction(
                    client, MODEL_CODE, essay_text, question,
                    training_examples, dataset_name, score_range
                )
                
                if result['success']:
                    predictions_map[essay_id] = {
                        'final': result['prediction'],
                        'individual': result['individual_predictions'],
                        'num_valid': result['num_valid_calls'],
                        'raw': result['raw_responses']
                    }
                else:
                    failed.append({'id': essay_id, 'error': result['error']})
                
                time.sleep(2)
            # ============================================================
            # SAVE FILE 1: FULL (with all 3 predictions + aggregated)
            # ============================================================
            full_df = test_df.copy()
            full_df['prediction_1'] = None
            full_df['prediction_2'] = None
            full_df['prediction_3'] = None
            full_df['final_prediction'] = None
            full_df['num_valid_calls'] = None
            
            for idx in full_df.index:
                row_id = str(full_df.loc[idx, cols["id"]])
                if row_id in predictions_map:
                    pred_data = predictions_map[row_id]
                    individual = pred_data['individual']
                    
                    # Fill in individual predictions
                    if len(individual) > 0:
                        full_df.loc[idx, 'prediction_1'] = individual[0]
                    if len(individual) > 1:
                        full_df.loc[idx, 'prediction_2'] = individual[1]
                    if len(individual) > 2:
                        full_df.loc[idx, 'prediction_3'] = individual[2]
                    
                    # Fill in aggregated
                    full_df.loc[idx, 'final_prediction'] = pred_data['final']
                    full_df.loc[idx, 'num_valid_calls'] = pred_data['num_valid']
            
            full_csv = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{dataset_name}_3call_FULL.csv")
            full_df.to_csv(full_csv, index=False)
            print(f"  ✓ Saved FULL: {full_csv}")
            

            submission_df = test_df.copy()  # Keep ALL columns
            
            # Fill the score column with final predictions
            for idx in submission_df.index:
                row_id = str(submission_df.loc[idx, cols["id"]])
                if row_id in predictions_map:
                    submission_df.loc[idx, cols["score"]] = predictions_map[row_id]['final']
            
            submission_csv = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{dataset_name}_3call.csv")
            submission_df.to_csv(submission_csv, index=False)
            print(f"  ✓ Saved SUBMISSION: {submission_csv}")
            
            print(f"  ✓ Valid: {len(predictions_map)} | Failed: {len(failed)}")
            
            results['datasets'].append({
                'name': dataset_name,
                'valid': len(predictions_map),
                'failed': len(failed),
                'csv': csv_path
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save JSON summary
    json_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_3call_summary.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"Saved: {json_path}")
    print("="*70)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key")
        sys.exit(1)
    
    confirm = input("\nRun 3-call aggregation? (y/n): ").strip().lower()
    if confirm == 'y':
        run_evaluation(api_key)
    else:
        print("Cancelled")