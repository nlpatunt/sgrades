#!/usr/bin/env python3
"""
Deductive Reasoning with 3-Call Aggregation
- No training data needed (applies general rules)
- Test data: Local CSV (sam_datasets/)
- Runs each data point 3 times
- Aggregates: mean (regression) or majority vote (classification)
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
RANDOM_SEED = 42
NUM_CALLS_PER_DATAPOINT = 3
NUM_TEST_SAMPLES = None  # Set to None for full dataset

MODEL_CODE = "google/gemini-2.5-flash"
MODEL_NAME = "gemini-2.5-flash"

# Paths
SAM_DATASETS_DIR = "/home/ts1506.UNT/Desktop/Work/besisr-benchmark-site/mllm_evaluation/sam_datasets"
OUTPUT_DIR = "deductive_3call_predictions"

os.environ["HF_TOKEN"] = "REMOVED_KEY"

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
    """Get column names"""
    norm = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
    configs = {
        "ASAP-AES": {"id": "essay_id", "text": "essay", "score": "domain1_score", "question": "prompt", "essay_set": "essay_set"},
        "ASAP2": {"id": "essay_id", "text": "full_text", "score": "score", "question": "assignment", "essay_set": None},
        "ASAP-SAS": {"id": "Id", "text": "essay_text", "score": "Score1", "question": "prompt", "essay_set": None},
        "ASAP_plus_plus": {"id": "essay_id", "text": "essay", "score": "overall_score", "question": "prompt", "essay_set": "essay_set"},
        "BEEtlE_2way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text", "essay_set": None},
        "BEEtlE_3way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text", "essay_set": None},
        "SciEntSBank_2way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text", "essay_set": None},
        "SciEntSBank_3way": {"id": "ID", "text": "student_answer", "score": "label", "question": "question_text", "essay_set": None},
        "CSEE": {"id": "index", "text": "essay", "score": "overall_score", "question": "prompt", "essay_set": None},
        "Mohlar": {"id": "ID", "text": "student_answer", "score": "grade", "question": "Question", "essay_set": None},
        "Ielts_Writing_Dataset": {"id": "ID", "text": "Essay", "score": "Overall_Score", "question": "Question", "essay_set": None},
        "Ielts_Writing_Task_2_Dataset": {"id": "ID", "text": "essay", "score": "band_score", "question": "prompt", "essay_set": None},
        "persuade_2": {"id": "essay_id_comp", "text": "full_text", "score": "holistic_essay_score", "question": "assignment", "essay_set": None},
        "Regrading_Dataset_J2C": {"id": "ID", "text": "student_answer", "score": "grade", "question": "Question", "essay_set": None},
        "OS_Dataset_q1": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "OS_Dataset_q2": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "OS_Dataset_q3": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "OS_Dataset_q4": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "OS_Dataset_q5": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "Rice_Chem_Q1": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
        "Rice_Chem_Q2": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
        "Rice_Chem_Q3": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
        "Rice_Chem_Q4": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
    }
    return configs.get(norm, {"id": "id", "text": "text", "score": "score", "question": "question", "essay_set": None})

# ============================================================================
# LOAD TEST DATA FROM LOCAL CSV
# ============================================================================

def load_test_data_from_csv(dataset_name: str) -> pd.DataFrame:
    """Load test data from local CSV"""
    csv_filename = f"{dataset_name}.csv"
    csv_path = os.path.join(SAM_DATASETS_DIR, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if NUM_TEST_SAMPLES is not None:
        df = df.head(NUM_TEST_SAMPLES)
        print(f"  ✓ TEST MODE: Using only {len(df)} samples from {csv_filename}")
    else:
        print(f"  ✓ Loaded {len(df)} test samples from {csv_filename}")
    
    return df

# ============================================================================
# DEDUCTIVE PROMPT CREATION
# ============================================================================

def create_deductive_prompt(essay_text: str, dataset_info: dict) -> dict:
    """Create deductive reasoning prompt - applies general rules to specific cases"""
    
    dataset_name = dataset_info.get('name', 'D_ASAP-AES')
    essay_set = dataset_info.get('essay_set', 1)
    score_range = get_score_range_for_dataset(dataset_name, essay_set)
    question = dataset_info.get('question', '')
    
    is_3way = '3way' in dataset_name.lower()
    is_2way = '2way' in dataset_name.lower()
    
    if is_3way:
        examples = """
GENERAL PRINCIPLE 1: Different substances have different physical properties (solubility, density, etc.)
APPLICATION: Student claims all solids must have the same solubility
DEDUCTION: This contradicts the general principle → Classification: contradictory

GENERAL PRINCIPLE 2: Voltage measures electrical potential difference between two points
APPLICATION: Student says "terminals are the same" for 0V reading
DEDUCTION: This misunderstands the principle (0V = same potential, not same terminals) → Classification: contradictory

GENERAL PRINCIPLE 3: Zero voltage indicates equal electrical potential
APPLICATION: Student says "same electrical state"
DEDUCTION: This correctly applies the principle → Classification: correct
"""
        
        system_prompt = f"""You are an expert evaluator using DEDUCTIVE REASONING.

DEDUCTIVE PROCESS:
1. Start with GENERAL scientific principles/laws
2. Apply the principle to THIS SPECIFIC answer
3. Logically derive the classification

EXAMPLES OF DEDUCTIVE REASONING:
{examples}

Apply this process: Principle → Application → Conclusion

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Classify this answer:

QUESTION: {question}
ANSWER: {essay_text}

STOP. Do not write steps. Do not write explanations.

Your ENTIRE response must be EXACTLY ONE of these three words:
correct
contradictory
incorrect

Nothing else. Just the word."""

    elif is_2way:
        examples = """
GENERAL PRINCIPLE 1: Switches control components only within their electrical circuit path
APPLICATION: Student says switch must be in path with bulb
DEDUCTION: This correctly applies the principle → Classification: correct

GENERAL PRINCIPLE 2: Territorial animals require separate spaces to reduce aggression
APPLICATION: Student recommends adding shelter
DEDUCTION: This correctly applies the principle → Classification: correct

GENERAL PRINCIPLE 3: Different chemicals have different solubilities (substance-specific property)
APPLICATION: Student says different solubilities mean measurements are wrong
DEDUCTION: This misapplies the principle (different solubilities are expected) → Classification: incorrect
"""
        
        system_prompt = f"""You are an expert evaluator using DEDUCTIVE REASONING.

DEDUCTIVE PROCESS:
1. Start with GENERAL scientific principles
2. Apply to THIS SPECIFIC answer
3. Derive classification logically

EXAMPLES OF DEDUCTIVE REASONING:
{examples}

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Classify this answer:

QUESTION: {question}
ANSWER: {essay_text}

STOP. Do not write steps. Do not write explanations.

Your ENTIRE response must be EXACTLY ONE of these two words:
correct
incorrect

Nothing else. Just the word."""

    else:
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

SCORING RANGE: {score_range}
TASK: {dataset_info.get('description', 'Essay scoring')}"""

        user_prompt = f"""Apply deductive reasoning to score:

QUESTION/PROMPT:
{question}

ESSAY TO SCORE:
{essay_text}

Apply general scoring rules to this specific essay and derive the score.

Provide ONLY a single number between {score_range[0]} and {score_range[1]} with no explanation. Just the number."""

    return {
        "system": system_prompt,
        "user": user_prompt
    }

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
                         dataset_name, essay_set):
    """Call model 3 times and aggregate"""
    predictions = []
    raw_responses = []
    total_tokens = 0
    
    dataset_info = {
        'name': dataset_name,
        'essay_set': essay_set,
        'question': question,
        'description': 'Evaluate student response'
    }
    
    prompt_data = create_deductive_prompt(essay_text, dataset_info)
    
    messages = [
        {"role": "system", "content": prompt_data["system"]},
        {"role": "user", "content": prompt_data["user"]}
    ]
    
    score_range = get_score_range_for_dataset(dataset_name, essay_set)
    
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
    print(f"DEDUCTIVE 3-CALL AGGREGATION - {MODEL_NAME}")
    print("="*70)
    print(f"Calls per data point: {NUM_CALLS_PER_DATAPOINT}")
    print(f"Reasoning: DEDUCTIVE (general rules → specific case)")
    print(f"Test samples per dataset: {NUM_TEST_SAMPLES if NUM_TEST_SAMPLES else 'ALL'}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {
        'model': MODEL_NAME,
        'method': 'deductive_3call',
        'calls_per_point': NUM_CALLS_PER_DATAPOINT,
        'datasets': []
    }
    
    for dataset_name in DATASETS:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        
        try:
            # Load test data from CSV (NO training data needed)
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
                
                # Extract essay_set if exists
                essay_set = 1
                if cols.get("essay_set") and cols["essay_set"] in row and pd.notna(row[cols["essay_set"]]):
                    essay_set = int(row[cols["essay_set"]])
                
                result = get_3call_prediction(
                    client, MODEL_CODE, essay_text, question,
                    dataset_name, essay_set
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
                    
                    if len(individual) > 0:
                        full_df.loc[idx, 'prediction_1'] = individual[0]
                    if len(individual) > 1:
                        full_df.loc[idx, 'prediction_2'] = individual[1]
                    if len(individual) > 2:
                        full_df.loc[idx, 'prediction_3'] = individual[2]
                    
                    full_df.loc[idx, 'final_prediction'] = pred_data['final']
                    full_df.loc[idx, 'num_valid_calls'] = pred_data['num_valid']
            
            full_csv = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{dataset_name}_3call_FULL.csv")
            full_df.to_csv(full_csv, index=False)
            print(f"  ✓ Saved FULL: {full_csv}")
            
            # ============================================================
            # SAVE FILE 2: SUBMISSION (ALL columns + final score)
            # ============================================================
            submission_df = test_df.copy()
            
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
                'csv': submission_csv
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
    
    confirm = input("\nRun deductive 3-call aggregation? (y/n): ").strip().lower()
    if confirm == 'y':
        run_evaluation(api_key)
    else:
        print("Cancelled")