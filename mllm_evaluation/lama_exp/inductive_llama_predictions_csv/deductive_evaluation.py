#!/usr/bin/env python3
"""
Zero-Shot Deductive Evaluation
Uses general principles and logical deduction (no training examples)
"""
import os
import json
import time
import sys
import re
from openai import OpenAI
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from dataset_ranges import get_score_range_for_dataset

# Set HuggingFace token
os.environ["HF_TOKEN"] = "REMOVED_KEY"

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_ESSAYS = None  # None = all test essays

MODEL_CODE = "meta-llama/llama-4-scout"
MODEL_NAME = "llama-4-scout"

# Output directory for CSV files
OUTPUT_DIR = f"deductive_{MODEL_NAME}_predictions_csv"

# ============================================================================
# FUNCTIONS
# ============================================================================

def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

def download_test_data(dataset_name: str, num_essays: int = None) -> Dict[str, Any]:
    """Download test data from HuggingFace"""
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
    print(f"  Downloading test data for {normalized_name}...")
    
    try:
        if normalized_name in ["BEEtlE_2way", "BEEtlE_3way"]:
            import requests
            from io import StringIO
            
            suffix = "2way" if "2way" in normalized_name else "3way"
            url = f"https://huggingface.co/datasets/nlpatunt/D_BEEtlE/resolve/main/test_{suffix}.csv"
            print(f"    Direct download from: {url}")
            
            hf_token = os.getenv("HF_TOKEN")
            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            
            response = requests.get(url, headers=headers, timeout=60)
            
            if response.status_code == 401:
                raise Exception("HuggingFace authentication required")
            
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            print(f"    ✓ Downloaded via direct URL")
            
        elif normalized_name in ["SciEntSBank_2way", "SciEntSBank_3way"]:
            import requests
            from io import StringIO
            
            suffix = "2way" if "2way" in normalized_name else "3way"
            url = f"https://huggingface.co/datasets/nlpatunt/D_SciEntSBank/resolve/main/test_{suffix}.csv"
            print(f"    Direct download from: {url}")
            
            hf_token = os.getenv("HF_TOKEN")
            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            
            response = requests.get(url, headers=headers, timeout=60)
            
            if response.status_code == 401:
                raise Exception("HuggingFace authentication required")
            
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            print(f"    ✓ Downloaded via direct URL")
            
        else:
            from datasets import load_dataset
            
            if normalized_name in ["Rice_Chem_Q1", "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"]:
                q_num = normalized_name.split("_")[-1]
                dataset = load_dataset("nlpatunt/Rice_Chem", data_files=f"{q_num}/test.csv")["train"]
            elif normalized_name.startswith("grade_like_a_human_dataset_os_q"):
                q_num = normalized_name.split("_q")[-1]
                dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", name=f"q{q_num}", split="test")
            elif normalized_name == "persuade_2":
                dataset = load_dataset("nlpatunt/persuade_2", data_files="test.csv")["train"]
            elif normalized_name == "Mohlar":
                dataset = load_dataset("nlpatunt/Mohlar", data_files="test.csv")["train"]
            elif normalized_name == "ASAP-SAS":
                dataset = load_dataset("nlpatunt/ASAP-SAS", data_files="test.csv")["train"]
            else:
                dataset = load_dataset(f"nlpatunt/{normalized_name}", split="test", trust_remote_code=True)
            
            df = dataset.to_pandas()
        
        print(f"    ✓ Downloaded: {len(df)} rows")
        print(f"    ✓ Columns: {list(df.columns)}")
        
        # Check if label column has any actual values (ground truth)
        if 'label' in df.columns:
            non_empty_labels = df['label'].notna().sum()
            if non_empty_labels > 0:
                print(f"    ⚠️  WARNING: Label column has {non_empty_labels} ground truth values!")
                print(f"    ⚠️  Dropping label column to avoid data leakage")
                df = df.drop(columns=['label'])
            else:
                print(f"    ✓ Label column is empty (will be filled with predictions)")
        
        # Also check for other score columns that might have ground truth
        cols_to_drop = [col for col in ['score', 'Score'] if col in df.columns and df[col].notna().sum() > 0]
        if cols_to_drop:
            print(f"    ⚠️  Dropping columns with ground truth: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Show first row
        cols = get_dataset_columns(dataset_name)
        if len(df) > 0:
            first_id = df[cols["id"]].iloc[0]
            first_q = str(df[cols["question"]].iloc[0])[:80]
            first_a = str(df[cols["text"]].iloc[0])[:80]
            print(f"    First test ID: {first_id}")
            print(f"    First Q: {first_q}...")
            print(f"    First A: {first_a}...")
        
        # Sample if needed
        if num_essays and num_essays < len(df):
            df = df.sample(n=num_essays, random_state=42)
            print(f"    ✓ Sampled {len(df)} examples")
        
        print(f"    ✓ Final test data: {len(df)} examples")
        return {"status": "success", "dataset": df}
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    """Get column names for a dataset"""
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
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
        "grade_like_a_human_dataset_os_q1": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "grade_like_a_human_dataset_os_q2": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "grade_like_a_human_dataset_os_q3": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "grade_like_a_human_dataset_os_q4": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "grade_like_a_human_dataset_os_q5": {"id": "ID", "text": "answer", "score": "score_1", "question": "question", "essay_set": None},
        "Rice_Chem_Q1": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
        "Rice_Chem_Q2": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
        "Rice_Chem_Q3": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
        "Rice_Chem_Q4": {"id": "sis_id", "text": "student_response", "score": "Score", "question": "Prompt", "essay_set": None},
    }
    
    return configs.get(normalized_name, {"id": "ID", "text": "text", "score": "score", "question": "question", "essay_set": None})

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

def validate_prediction(prediction_text: str, dataset_name: str, essay_set: int = 1) -> Dict[str, Any]:
    """Validate and extract prediction"""
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    is_3way = '3way' in normalized_name.lower()
    is_2way = '2way' in normalized_name.lower()
    prediction_clean = prediction_text.strip().lower()
    
    if is_3way:
        for label in ['correct', 'contradictory', 'incorrect']:
            if label in prediction_clean:
                return {'valid': True, 'extracted': label, 'error': None, 'raw': prediction_text}
        return {'valid': False, 'extracted': None, 'error': f'Invalid 3way: {prediction_text[:50]}', 'raw': prediction_text}
    
    elif is_2way:
        if 'incorrect' in prediction_clean:
            return {'valid': True, 'extracted': 'incorrect', 'error': None, 'raw': prediction_text}
        elif 'correct' in prediction_clean:
            return {'valid': True, 'extracted': 'correct', 'error': None, 'raw': prediction_text}
        return {'valid': False, 'extracted': None, 'error': f'Invalid 2way: {prediction_text[:50]}', 'raw': prediction_text}
    
    else:
        score_range = get_score_range_for_dataset(dataset_name, essay_set)
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', prediction_clean)
        
        if not numbers:
            return {'valid': False, 'extracted': None, 'error': f'No number: {prediction_text[:50]}', 'raw': prediction_text}
        
        try:
            score = float(numbers[0])
            if score < score_range[0] or score > score_range[1]:
                return {'valid': False, 'extracted': score, 'error': f'Out of range {score_range}', 'raw': prediction_text}
            return {'valid': True, 'extracted': score, 'error': None, 'raw': prediction_text}
        except ValueError:
            return {'valid': False, 'extracted': None, 'error': 'Parse error', 'raw': prediction_text}

def call_openrouter_api(client, model_code: str, messages: list, max_retries: int = 3):
    """Call API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_code,
                messages=messages,
                max_tokens=50,
                temperature=0.1,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "S-GRADES Deductive"
                }
            )
            return {
                'response': response,
                'tokens': {
                    'prompt': response.usage.prompt_tokens if response.usage else 0,
                    'completion': response.usage.completion_tokens if response.usage else 0,
                    'total': (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
                },
                'success': True
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {'response': None, 'tokens': {}, 'success': False, 'error': str(e)}
            time.sleep(2 ** attempt)

def get_prediction_with_retry(client, model_code, essay_text, question, 
                              dataset_name, score_range, essay_set=1, max_retries=5):
    """Get validated prediction with retry"""
    for attempt in range(max_retries):
        dataset_info = {
            'name': dataset_name,
            'essay_set': essay_set,
            'question': question,
            'description': f'{dataset_name} evaluation'
        }
        
        prompt_data = create_deductive_prompt(essay_text, dataset_info)
        
        messages = [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user", "content": prompt_data["user"]}
        ]
        
        api_result = call_openrouter_api(client, model_code, messages)
        
        if not api_result['success']:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {'success': False, 'error': f"API failed after {max_retries} attempts", 'attempts': max_retries}
        
        response_text = api_result['response'].choices[0].message.content.strip()
        validation = validate_prediction(response_text, dataset_name, essay_set)
        
        if validation['valid']:
            return {
                'success': True,
                'prediction': validation['extracted'],
                'raw_response': validation.get('raw', response_text),
                'tokens': api_result['tokens'],
                'attempts': attempt + 1
            }
        else:
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return {
        'success': False,
        'error': f"Validation failed after {max_retries} attempts: {validation['error']}",
        'last_response': response_text,
        'attempts': max_retries
    }

def get_all_datasets():
    """Get list of all D_ datasets"""
    # TEST MODE: Only SciEntSBank (since you converted those)
    datasets = [
        "D_SciEntSBank_2way",
        "D_SciEntSBank_3way"
    ]
    return datasets

def save_predictions_as_csv(test_df: pd.DataFrame, predictions_map: Dict, 
                            dataset_name: str, score_column: str):
    """Save predictions in CSV format matching original test data structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_df = test_df.copy()
    
    for idx in output_df.index:
        row_id = str(output_df.loc[idx, get_dataset_columns(dataset_name)["id"]])
        if row_id in predictions_map:
            output_df.loc[idx, score_column] = predictions_map[row_id]
    
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    csv_filename = f"{MODEL_NAME}_{normalized_name}_deductive.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    
    output_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved predictions: {csv_path}")
    
    return csv_path

def run_deductive_evaluation(api_key):
    """Main evaluation function"""
    client = get_client(api_key)
    
    print("="*70)
    print(f"ZERO-SHOT DEDUCTIVE EVALUATION - {MODEL_NAME}")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_datasets = get_all_datasets()
    
    results = {
        'model_code': MODEL_CODE,
        'model_name': MODEL_NAME,
        'reasoning_approach': 'deductive',
        'timestamp': datetime.now().isoformat(),
        'datasets': []
    }
    
    for dataset_idx, dataset_name in enumerate(all_datasets, 1):
        print(f"\n{'='*70}")
        print(f"Dataset {dataset_idx}/{len(all_datasets)}: {dataset_name}")
        print(f"{'='*70}")
        
        test_result = download_test_data(dataset_name, NUM_ESSAYS)
        if test_result["status"] != "success":
            print(f"  ✗ Skipping - no test data")
            continue
        
        test_df = test_result["dataset"]
        cols = get_dataset_columns(dataset_name)

        dataset_result = {
            'dataset_name': dataset_name,
            'predictions': [],
            'failed_predictions': [],
            'stats': {'valid': 0, 'invalid': 0}
        }
        
        predictions_map = {}
        
        for i, (_, row) in enumerate(test_df.iterrows(), 1):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_df)}")
            
            essay_id = str(row[cols["id"]])
            essay_text = row[cols["text"]]
            question = row.get(cols["question"], "")
            
            essay_set = 1
            if cols.get("essay_set") and cols["essay_set"] in row:
                essay_set = int(row[cols["essay_set"]])
            
            score_range = get_score_range_for_dataset(dataset_name, essay_set)
            
            result = get_prediction_with_retry(
                client, MODEL_CODE, essay_text, question,
                dataset_name, score_range, essay_set
            )
            
            if result['success']:
                pred_entry = {
                    'id': essay_id,
                    'prediction': result['prediction'],
                    'tokens': result['tokens']
                }
                if cols.get("essay_set"):
                    pred_entry['essay_set'] = essay_set
                
                dataset_result['predictions'].append(pred_entry)
                dataset_result['stats']['valid'] += 1
                predictions_map[essay_id] = result['prediction']
            else:
                fail_entry = {
                    'id': essay_id,
                    'error': result['error'],
                    'attempts': result['attempts']
                }
                if cols.get("essay_set"):
                    fail_entry['essay_set'] = essay_set
                if 'last_response' in result:
                    fail_entry['last_response'] = result['last_response'][:100]
                
                dataset_result['failed_predictions'].append(fail_entry)
                dataset_result['stats']['invalid'] += 1
            
            time.sleep(2)
        
        csv_path = save_predictions_as_csv(test_df, predictions_map, dataset_name, cols["score"])
        dataset_result['csv_output'] = csv_path
        
        results['datasets'].append(dataset_result)
        print(f"  ✓ Valid: {dataset_result['stats']['valid']} | ✗ Invalid: {dataset_result['stats']['invalid']}")
        
        temp_filename = f"deductive_{MODEL_NAME}_partial.json"
        with open(temp_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        time.sleep(5)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    
    filename = f"deductive_{MODEL_NAME}_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved JSON summary: {filename}")
    print(f"CSV files saved in: {OUTPUT_DIR}/")
    return results

if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key")
        sys.exit(1)
    
    confirm = input("\nProceed with deductive evaluation? (y/n): ").strip().lower()
    if confirm == 'y':
        run_deductive_evaluation(api_key)
    else:
        print("Cancelled")