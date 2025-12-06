

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
MODEL_CODE = "openai/gpt-4o-mini"
MODEL_NAME = "gpt-4o-mini"


OUTPUT_DIR = f"abductive_{MODEL_NAME}_predictions_csv"

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

def load_checkpoint(model_name):
    """Load previous checkpoint if exists"""
    checkpoint_file = f"abductive_{model_name}_checkpoint.json"
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Found checkpoint: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            completed_datasets = set(d['dataset_name'] for d in checkpoint.get('datasets', []))
            print(f"   ✓ Previously completed: {len(completed_datasets)} datasets")
            print(f"   ✓ Datasets: {list(completed_datasets)}")
            
            resume = input("\n   Resume from checkpoint? (y/n): ").strip().lower()
            if resume == 'y':
                return checkpoint, completed_datasets
        except Exception as e:
            print(f"   ✗ Error loading checkpoint: {e}")
    
    return None, set()

def save_checkpoint(results, model_name):
    """Save checkpoint after each dataset"""
    checkpoint_file = f"abductive_{model_name}_checkpoint.json"
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  💾 Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        print(f"  ⚠️  Checkpoint save failed: {e}")

def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    """Get column names for a dataset"""
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
    configs = {
        "ASAP-AES": {"id": "essay_id", "text": "essay", "score": "domain1_score", "question": "prompt", "essay_set": "essay_set"},
        "ASAP2": {"id": "essay_id", "text": "full_text", "score": "score", "question": "assignment","essay_set": None},
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

def create_abductive_prompt(essay_text: str, dataset_info: dict) -> dict:
    """Create abductive reasoning prompt - infers best explanation from observations"""
    
    dataset_name = dataset_info.get('name', 'D_ASAP-AES')
    essay_set = dataset_info.get('essay_set', 1)
    score_range = get_score_range_for_dataset(dataset_name, essay_set)
    question = dataset_info.get('question', '')
    
    is_3way = '3way' in dataset_name.lower()
    is_2way = '2way' in dataset_name.lower()
    
    if is_3way:
        examples = """
OBSERVATION 1: Student claims "all solids must have the same solubility"
POSSIBLE EXPLANATIONS:
- Student correctly understands solubility is universal → Would expect: "Yes, measurements correct"
- Student misunderstands solubility as substance-specific → Would expect: discussion of different properties
- Student contradicts basic chemistry principles → Would expect: incorrect claim about uniformity
BEST EXPLANATION: Student contradicts established principle (different substances have different properties)
CLASSIFICATION: contradictory

OBSERVATION 2: Student says terminals have "0V means terminals are the same"
POSSIBLE EXPLANATIONS:
- Student understands 0V = same potential/connected → Would expect: "same electrical state"
- Student confuses physical identity with electrical state → Would expect: vague "same" language
- Student correctly explains voltage reading → Would expect: mention of connection/continuity
BEST EXPLANATION: Ambiguous answer suggesting confusion about what "same" means (physical vs electrical)
CLASSIFICATION: contradictory

OBSERVATION 3: Student says "same electrical state" for 0V reading
POSSIBLE EXPLANATIONS:
- Student randomly guessed → Would expect: no technical terminology
- Student memorized without understanding → Would expect: definition without application
- Student understands voltage measures potential difference → Would expect: correct technical terms
BEST EXPLANATION: Student correctly understands 0V indicates equal electrical potential
CLASSIFICATION: correct
"""
        
        system_prompt = f"""You are an expert evaluator using ABDUCTIVE REASONING.

ABDUCTIVE PROCESS:
1. Observe what the student wrote
2. Generate possible explanations for why they wrote it
3. Identify which explanation best fits the evidence
4. Derive classification from best explanation

EXAMPLES OF ABDUCTIVE REASONING:
{examples}

Apply this process: Observation → Possible Explanations → Best Fit → Classification

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Use abductive reasoning to classify this answer:

QUESTION:
{question}

STUDENT ANSWER (OBSERVATION):
{essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.

Your ENTIRE response must be EXACTLY ONE of these three words:
correct
contradictory
incorrect

Nothing else. Just the word."""

    elif is_2way:
        examples = """
OBSERVATION 1: Student says "switch must be in path with the bulb"
POSSIBLE EXPLANATIONS:
- Student memorized without understanding → Would expect: no reasoning about why
- Student understands circuit paths → Would expect: mention of path/circuit
- Student randomly guessed correctly → Would expect: very brief answer
BEST EXPLANATION: Student demonstrates understanding that switches affect components in their path
CLASSIFICATION: correct

OBSERVATION 2: Student recommends "adding shelter" for territorial lizards
POSSIBLE EXPLANATIONS:
- Student thinks shelter helps any problem → Would expect: generic "make comfortable"
- Student understands territorial behavior needs → Would expect: shelter for separate spaces
- Student doesn't understand territoriality → Would expect: irrelevant suggestion
BEST EXPLANATION: Student correctly infers territorial animals need physical separation/hiding spots
CLASSIFICATION: correct

OBSERVATION 3: Student says "different solubilities mean measurements are wrong"
POSSIBLE EXPLANATIONS:
- Student thinks all substances should behave identically → Would expect: claim of error
- Student understands solubility is substance-specific → Would expect: "both can be correct"
- Student confuses variability with experimental error → Would expect: focus on measurement problems
BEST EXPLANATION: Student misunderstands that different solubilities are expected, not errors
CLASSIFICATION: incorrect
"""
        
        system_prompt = f"""You are an expert evaluator using ABDUCTIVE REASONING.

ABDUCTIVE PROCESS:
1. Observe what the student wrote
2. Generate possible explanations
3. Identify best explanation
4. Classify based on that

EXAMPLES OF ABDUCTIVE REASONING:
{examples}

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Use abductive reasoning to classify:

QUESTION:
{question}

STUDENT ANSWER (OBSERVATION):
{essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.

Your ENTIRE response must be EXACTLY ONE of these two words:
correct
incorrect

Nothing else. Just the word."""

    else:
        examples = """
OBSERVATION 1: Student provides "amount of each sample, temperature, and rinsing duration"
POSSIBLE EXPLANATIONS:
- Student randomly listed items → Would expect: irrelevant details
- Student identified genuinely missing info → Would expect: procedurally important details
- Student didn't read procedure carefully → Would expect: things already specified
BEST EXPLANATION: Student correctly identified three critical missing procedural details
INFERENCE: Demonstrates good understanding, minor writing issues
SCORE: 2/3

OBSERVATION 2: Student wrote "It took 10 time units" with no explanation
POSSIBLE EXPLANATIONS:
- Student ran simulation but doesn't understand why → Would expect: correct number, no reasoning
- Student guessed lucky → Would expect: wrong answer or right answer with uncertainty
- Student understands deeply → Would expect: explanation of process interactions
BEST EXPLANATION: Student obtained correct result but cannot explain the underlying process
INFERENCE: Partial understanding - can execute but not explain
SCORE: 8/15

OBSERVATION 3: Student discusses "Coulomb's Law, electron-electron repulsion, 2p to 2s orbital jump"
POSSIBLE EXPLANATIONS:
- Student memorized terms without understanding → Would expect: misapplied concepts
- Student has partial understanding → Would expect: some correct, some errors
- Student fully understands → Would expect: all concepts correctly applied
BEST EXPLANATION: Student understands shell transitions and repulsion but incorrectly dismisses Zeff changes
INFERENCE: Solid grasp with one significant conceptual error
SCORE: 5/8
"""
        
        system_prompt = f"""You are an expert scorer using ABDUCTIVE REASONING.

ABDUCTIVE PROCESS:
1. Observe what the student wrote
2. Generate explanations for their knowledge state
3. Identify which explanation best fits the evidence
4. Infer appropriate score from that explanation

EXAMPLES OF ABDUCTIVE SCORING:
{examples}

SCORING RANGE: {score_range}
TASK: {dataset_info.get('description', 'Essay scoring')}"""

        user_prompt = f"""Use abductive reasoning to score:

QUESTION/PROMPT:
{question}

ESSAY (OBSERVATION):
{essay_text}

STOP. Do not write steps. Do not write explanations. Do not write reasoning.

Your ENTIRE response must be EXACTLY one number between {score_range[0]} and {score_range[1]}.

Nothing else. Just the number."""
    
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
        # FIXED: Check 'incorrect' BEFORE 'correct' to avoid substring issue
        if 'incorrect' in prediction_clean:
            return {'valid': True, 'extracted': 'incorrect', 'error': None, 'raw': prediction_text}
        elif 'contradictory' in prediction_clean:
            return {'valid': True, 'extracted': 'contradictory', 'error': None, 'raw': prediction_text}
        elif 'correct' in prediction_clean:
            return {'valid': True, 'extracted': 'correct', 'error': None, 'raw': prediction_text}
        else:
            return {'valid': False, 'extracted': None, 'error': f'Invalid 3way: {prediction_text[:50]}', 'raw': prediction_text}
    
    elif is_2way:
        # FIXED: Same issue here - check 'incorrect' before 'correct'
        if 'incorrect' in prediction_clean:
            return {'valid': True, 'extracted': 'incorrect', 'error': None, 'raw': prediction_text}
        elif 'correct' in prediction_clean:
            return {'valid': True, 'extracted': 'correct', 'error': None, 'raw': prediction_text}
        else:
            return {'valid': False, 'extracted': None, 'error': f'Invalid 2way: {prediction_text[:50]}', 'raw': prediction_text}
    
    else:
        # Numeric scoring remains the same
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
                    "X-Title": "S-GRADES Abductive"
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
        
        prompt_data = create_abductive_prompt(essay_text, dataset_info)
        
        messages = [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user", "content": prompt_data["user"]}
        ]
        
        api_result = call_openrouter_api(client, model_code, messages)
        
        # FIXED: Add comprehensive checks for API failures
        if (not api_result['success'] or 
            api_result.get('response') is None or 
            not hasattr(api_result['response'], 'choices') or
            len(api_result['response'].choices) == 0 or
            api_result['response'].choices[0].message.content is None):
            
            error_msg = api_result.get('error', 'Unknown API error')
            print(f"      Attempt {attempt + 1}/{max_retries}: API failed - {error_msg}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                return {
                    'success': False, 
                    'error': f"API failed after {max_retries} attempts: {error_msg}", 
                    'attempts': max_retries
                }
        
        # Safe to access response now
        response_text = api_result['response'].choices[0].message.content.strip()
        validation = validate_prediction(response_text, dataset_name, essay_set)
        
        if validation['valid']:
            if attempt > 0:
                print(f"      ✓ Valid after {attempt + 1} attempts")
            return {
                'success': True,
                'prediction': validation['extracted'],
                'raw_response': validation.get('raw', response_text),
                'tokens': api_result['tokens'],
                'attempts': attempt + 1
            }
        else:
            print(f"      Attempt {attempt + 1}/{max_retries}: Validation failed - {validation['error']}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return {
        'success': False,
        'error': f"Validation failed after {max_retries} attempts: {validation['error']}",
        'last_response': response_text if 'response_text' in locals() else 'No response',
        'attempts': max_retries
    }


def get_all_datasets():
    """Get list of all D_ datasets"""
    # TEST MODE: Only SciEntSBank (since you converted those)
    datasets = [
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
    csv_filename = f"{MODEL_NAME}_{normalized_name}_abductive.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    
    output_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved predictions: {csv_path}")
    
    return csv_path

def run_abductive_evaluation(api_key):
    """Main evaluation function with checkpoint/resume support"""
    client = get_client(api_key)
    
    print("="*70)
    print(f"ZERO-SHOT ABDUCTIVE EVALUATION - {MODEL_NAME}")
    print("="*70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # STEP 1: Try to load checkpoint
    checkpoint, completed_datasets = load_checkpoint(MODEL_NAME)
    
    if checkpoint:
        results = checkpoint
        print(f"\n✓ Resuming from checkpoint with {len(completed_datasets)} completed datasets\n")
    else:
        results = {
            'model_code': MODEL_CODE,
            'model_name': MODEL_NAME,
            'reasoning_approach': 'abductive',
            'timestamp': datetime.now().isoformat(),
            'datasets': []
        }
    
    all_datasets = get_all_datasets()
    
    # STEP 2: Filter out already completed datasets
    remaining_datasets = [d for d in all_datasets if d not in completed_datasets]
    
    if not remaining_datasets:
        print("✓ All datasets already completed!")
        return results
    
    print(f"📊 Datasets to process: {len(remaining_datasets)} of {len(all_datasets)}")
    print(f"   Remaining: {remaining_datasets}\n")
    
    # STEP 3: Process each dataset with error handling
    for dataset_idx, dataset_name in enumerate(remaining_datasets, 1):
        print(f"\n{'='*70}")
        print(f"Dataset {dataset_idx}/{len(remaining_datasets)}: {dataset_name}")
        print(f"Overall Progress: {len(completed_datasets) + dataset_idx}/{len(all_datasets)}")
        print(f"{'='*70}")
        
        try:
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
                'stats': {'valid': 0, 'invalid': 0},
                'start_time': datetime.now().isoformat()
            }
            
            predictions_map = {}
            
            # STEP 4: Process each essay with error handling
            for i, (_, row) in enumerate(test_df.iterrows(), 1):
                try:
                    if i % 100 == 0:
                        print(f"  Progress: {i}/{len(test_df)} (Valid: {dataset_result['stats']['valid']}, Failed: {dataset_result['stats']['invalid']})")
                    
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
                    
                except Exception as essay_error:
                    print(f"  ⚠️  Error processing essay {i}: {essay_error}")
                    dataset_result['failed_predictions'].append({
                        'id': essay_id if 'essay_id' in locals() else f'row_{i}',
                        'error': f'Processing error: {str(essay_error)}',
                        'attempts': 0
                    })
                    dataset_result['stats']['invalid'] += 1
                    continue
            
            # STEP 5: Save CSV even if some predictions failed
            csv_path = save_predictions_as_csv(test_df, predictions_map, dataset_name, cols["score"])
            dataset_result['csv_output'] = csv_path
            dataset_result['end_time'] = datetime.now().isoformat()
            
            results['datasets'].append(dataset_result)
            print(f"  ✓ Valid: {dataset_result['stats']['valid']} | ✗ Invalid: {dataset_result['stats']['invalid']}")
            
            # STEP 6: Save checkpoint after EACH dataset
            save_checkpoint(results, MODEL_NAME)
            completed_datasets.add(dataset_name)
            
            time.sleep(5)
            
        except Exception as dataset_error:
            print(f"  ✗✗✗ CRITICAL ERROR processing {dataset_name}: {dataset_error}")
            import traceback
            traceback.print_exc()
            
            # Still save checkpoint even after error
            save_checkpoint(results, MODEL_NAME)
            
            retry = input(f"\n  Dataset {dataset_name} failed. Continue with next dataset? (y/n): ").strip().lower()
            if retry != 'y':
                print("\n  ⚠️  Stopping execution. Progress saved in checkpoint.")
                return results
            else:
                print(f"\n  ↪️  Skipping {dataset_name}, moving to next dataset...\n")
                continue
    
    print("\n" + "="*70)
    print("✓ COMPLETE - ALL DATASETS PROCESSED")
    print("="*70)
    
    # STEP 7: Save final results
    final_filename = f"abductive_{MODEL_NAME}_{int(time.time())}.json"
    with open(final_filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved final results: {final_filename}")
    print(f"✓ CSV files saved in: {OUTPUT_DIR}/")
    
    # Clean up checkpoint file after successful completion
    checkpoint_file = f"abductive_{MODEL_NAME}_checkpoint.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✓ Cleaned up checkpoint file")
    
    return results

if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key")
        sys.exit(1)
    
    confirm = input("\nProceed with abductive evaluation? (y/n): ").strip().lower()
    if confirm == 'y':
        run_abductive_evaluation(api_key)
    else:
        print("Cancelled")
        