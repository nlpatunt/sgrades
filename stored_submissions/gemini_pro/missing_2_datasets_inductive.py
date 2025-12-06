import os
import json
import time
import sys
from openai import OpenAI
from datetime import datetime
from dataset_ranges import get_score_range_for_dataset
from single_model_test import SingleModelTester
from inductive_evaluation_gemini_pro import create_inductive_prompt

# Configuration
NUM_ESSAYS = None  # All essays

MODEL_CODE = "google/gemini-2.5-pro"
MODEL_NAME = "gemini-2.5-pro"

# ONLY these 2 missing datasets
MISSING_DATASETS = [
    "D_BEEtlE_3way",
    "D_CSEE"
]

def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

def load_essays_from_dataset(dataset_name: str, num_essays: int = None):
    """Load essays from BESESR"""
    try:
        tester = SingleModelTester()
        df = tester.download_test_data(dataset_name, num_essays)
        if df is None:
            return []
        essays_raw = tester.prepare_essays_for_prediction(df, dataset_name)
        essays = []
        for essay_raw in essays_raw:
            essay_info = {
                'text': essay_raw['text'],
                'id': essay_raw['id'],
                'essay_set': 1,
                'question': essay_raw.get('question', ''),
                'name': dataset_name,
                'description': f"{dataset_name} Evaluation"
            }
            if 'ASAP-AES' in dataset_name:
                id_col = 'essay_id' if 'essay_id' in df.columns else 'ID'
                matching_rows = df[df[id_col] == essay_raw['id']]
                if not matching_rows.empty and 'essay_set' in df.columns:
                    essay_info['essay_set'] = int(matching_rows.iloc[0]['essay_set'])
            essays.append(essay_info)
        print(f"  Loaded {len(essays)} essays from {dataset_name}")
        return essays
    except Exception as e:
        print(f"  Error loading {dataset_name}: {e}")
        return []

def call_openrouter_api(client, model_code: str, messages: list, max_retries: int = 3):
    """Call OpenRouter API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_code,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "S-GRADES Inductive Reasoning - Gemini Pro"
                }
            )
            usage = response.usage
            return {
                'response': response,
                'tokens': {
                    'prompt': usage.prompt_tokens if usage else 0,
                    'completion': usage.completion_tokens if usage else 0,
                    'total': (usage.prompt_tokens + usage.completion_tokens) if usage else 0
                },
                'success': True
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {'response': None, 'tokens': {}, 'success': False, 'error': str(e)}
            time.sleep(2 ** attempt)

def run_missing_evaluation(api_key):
    """Run inductive evaluation on missing datasets"""
    client = get_client(api_key)
    
    print("="*70)
    print("GEMINI PRO INDUCTIVE - MISSING 2 DATASETS")
    print("="*70)
    
    all_results = {
        'model_code': MODEL_CODE,
        'model_name': MODEL_NAME,
        'reasoning_approach': 'inductive',
        'timestamp': datetime.now().isoformat(),
        'datasets': []
    }
    
    for dataset_idx, dataset_name in enumerate(MISSING_DATASETS, 1):
        print(f"\n{'='*70}")
        print(f"Dataset {dataset_idx}/{len(MISSING_DATASETS)}: {dataset_name}")
        print(f"{'='*70}")
        
        essays = load_essays_from_dataset(dataset_name, NUM_ESSAYS)
        if not essays:
            print(f"  ✗ No essays loaded, skipping...")
            continue
        
        dataset_result = {
            'dataset_name': dataset_name,
            'essays_evaluated': 0,
            'predictions': []
        }
        
        for i, essay_info in enumerate(essays, 1):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(essays)} essays...")
            
            prompt_data = create_inductive_prompt(essay_info['text'], essay_info)
            messages = [
                {"role": "system", "content": prompt_data["system"]},
                {"role": "user", "content": prompt_data["user"]}
            ]
            
            api_result = call_openrouter_api(client, MODEL_CODE, messages)
            
            if api_result['success']:
                response_text = api_result['response'].choices[0].message.content.strip()
                dataset_result['predictions'].append({
                    'essay_id': essay_info['id'],
                    'essay_set': essay_info.get('essay_set', 1),
                    'prediction': response_text,
                    'tokens': api_result['tokens']
                })
            else:
                print(f"  Essay {i}: FAILED - {api_result.get('error', 'Unknown')}")
            
            time.sleep(2)
        
        dataset_result['essays_evaluated'] = len(dataset_result['predictions'])
        all_results['datasets'].append(dataset_result)
        print(f"  ✓ Completed: {dataset_result['essays_evaluated']} essays")
        time.sleep(5)
    
    print("\n" + "="*70)
    print("MISSING DATASETS COMPLETE")
    print("="*70)
    
    timestamp = int(time.time())
    filename = f"inductive_geminipro_missing2_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")
    return all_results

if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No API key provided")
        sys.exit(1)
    
    confirm = input("\nRun Gemini Pro on 2 missing datasets? (y/n): ").strip().lower()
    if confirm == 'y':
        run_missing_evaluation(api_key)
    else:
        print("Cancelled.")