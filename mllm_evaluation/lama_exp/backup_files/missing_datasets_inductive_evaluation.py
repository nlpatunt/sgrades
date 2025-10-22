import os
import json
import time
import sys
from openai import OpenAI
from datetime import datetime
from dataset_ranges import get_score_range_for_dataset
from single_model_test import SingleModelTester

# Configuration
NUM_ESSAYS = None  # None = all essays per dataset

# Model to test
MODEL_CODE = "meta-llama/llama-4-scout"
MODEL_NAME = "llama-4-scout"

# ONLY these 5 missing datasets
MISSING_DATASETS = [
    "D_grade_like_a_human_dataset_os_q1",
    "D_grade_like_a_human_dataset_os_q2",
    "D_grade_like_a_human_dataset_os_q3",
    "D_grade_like_a_human_dataset_os_q4",
    "D_grade_like_a_human_dataset_os_q5"
]

def get_client(api_key):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

def create_inductive_prompt(essay_text: str, dataset_info: dict) -> dict:
    """Create inductive reasoning prompt with examples - handles both numeric and categorical"""
    
    dataset_name = dataset_info.get('name', 'D_ASAP-AES')
    essay_set = dataset_info.get('essay_set', 1)
    score_range = get_score_range_for_dataset(dataset_name, essay_set)
    question = dataset_info.get('question', '')
    
    is_3way = '3way' in dataset_name.lower()
    is_2way = '2way' in dataset_name.lower()
    
    if is_3way:
        examples = """
EXAMPLE 1:
Question: "Janet found the solubility of silver chloride to be 86 grams in 50 milliliters of water. Mike found the solubility of silver nitrate to be 108 grams in 50 milliliters of water. Janet and Mike thought they must have made a mistake. They thought the solubilities of all materials in water should be the same. Is it possible that both solubilities are correct? Explain your answer."
Student Answer: "No because no solid that one has more than the other in the same amount of water."
Classification: contradictory
Reasoning: Answer contradicts the fundamental principle that different substances have different solubilities in water. The student incorrectly assumes all solids must have the same solubility in a given amount of water, which is scientifically incorrect. The correct answer is "Yes, both solubilities can be correct" because different chemical compounds have different solubility properties based on their molecular structure and interactions with water. The response also contains grammatical issues that make it difficult to parse, suggesting confusion about the underlying concept. While the student attempts to relate solubility to the amount of water, they fundamentally misunderstand that solubility is a substance-specific property.

EXAMPLE 2:
Question: "What does a voltage reading of 0 tell you about the connection between a bulb terminal and a battery terminal?"
Student Answer: "the terminals are the same"
Classification: contradictory
Reasoning: Answer contradicts fundamental electrical principles. A voltage reading of 0 indicates that the two points being measured are at the same electrical potential, meaning there is a complete conducting path between them (they are electrically connected). The student's answer "the terminals are the same" is ambiguous and appears to misunderstand what voltage measures. If they mean "at the same potential," they're partially correct but incomplete. If they mean "identical terminals," this contradicts basic circuit concepts. A 0V reading specifically indicates continuity/connection in the circuit, not that terminals are physically the same. The response demonstrates confusion about voltage as a measure of potential difference and fails to explain the connection aspect that the question asks about.

EXAMPLE 3:
Question: "What does a voltage reading of 0 tell you about the connection between a bulb terminal and a battery terminal?"
Student Answer: "The bulb terminal and the battery terminal have the same electrical state."
Classification: correct
Reasoning: Answer correctly identifies that a 0V reading means both terminals are at the same electrical potential (same electrical state). This demonstrates understanding that voltage measures potential difference, and when the difference is zero, the two points are at the same potential, indicating they are electrically connected through a conducting path. The response uses appropriate terminology ("electrical state" meaning potential) and shows comprehension of what voltage readings indicate about circuit connectivity.
"""
        system_prompt = f"""You are an expert evaluator using INDUCTIVE REASONING for answer classification.

INDUCTIVE PROCESS:
1. Learn classification patterns from the examples below
2. Identify what distinguishes correct, contradictory, and incorrect answers
3. Apply these learned patterns to classify the new answer

CLASSIFICATION EXAMPLES:
{examples}

From these examples, identify patterns in:
- CORRECT: Accurate answer with sound scientific reasoning
- CONTRADICTORY: Answer that directly contradicts established scientific principles
- INCORRECT: Wrong answer based on misunderstanding or incomplete knowledge
- How scientific accuracy determines classification
- What level of explanation is expected

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Based on the patterns you learned, classify this student answer:

QUESTION:
{question}

STUDENT ANSWER:
{essay_text}

Provide exactly ONE word from these options: correct, contradictory, or incorrect"""

    elif is_2way:
        examples = """
EXAMPLE 1:
Question: "What role does the path play in determining whether or not a switch affects a bulb?"
Student Answer: "If the switch is contained in the path with the bulb, the switch affects the bulb."
Classification: correct
Reasoning: Answer accurately identifies that the switch must be in the same electrical path as the bulb to affect it. The student demonstrates understanding of the key concept that switches only control components within their circuit path.

EXAMPLE 2:
Question: "Elena has a male lizard that has lived for several years in the habitat she provided. She knows that some lizards are territorial. Besides additional food and water, what should she be sure to include in the habitat before she adds another male lizard?"
Student Answer: "Elena should add shelter."
Classification: correct
Reasoning: Answer correctly identifies that shelter is necessary when adding another male lizard to accommodate territorial behavior. Territorial animals need separate spaces or hiding areas to reduce aggression and establish individual territories. The student demonstrates understanding that physical structures in the habitat help manage territorial conflicts between males.

EXAMPLE 3:
Question: "Janet found the solubility of silver chloride to be 86 grams in 50 milliliters of water. Mike found the solubility of silver nitrate to be 108 grams in 50 milliliters of water. Janet and Mike thought they must have made a mistake. They thought the solubilities of all materials in water should be the same. Is it possible that both solubilities are correct? Explain your answer."
Student Answer: "No because it has 2 different chemicals and it does not have the same solubilities."
Classification: incorrect
Reasoning: Answer reaches the wrong conclusion. While the student correctly observes that these are two different chemicals with different solubilities, they incorrectly conclude this means the measurements must be wrong. The correct answer is "Yes, both can be correct" precisely because different chemicals have different solubilities - this is a fundamental property of matter. The student demonstrates partial understanding (recognizing they're different substances) but fails to connect this to why both measurements could be valid. The answer suggests confusion about whether solubility is a universal constant or a substance-specific property.
"""
        system_prompt = f"""You are an expert evaluator using INDUCTIVE REASONING for answer classification.

INDUCTIVE PROCESS:
1. Learn classification patterns from the examples below
2. Identify what distinguishes correct from incorrect answers
3. Apply these learned patterns to classify the new answer

CLASSIFICATION EXAMPLES:
{examples}

From these examples, identify patterns in:
- What makes an answer CORRECT vs INCORRECT
- How completeness and accuracy affect classification
- What level of detail is expected

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Based on the patterns you learned, classify this student answer:

QUESTION:
{question}

STUDENT ANSWER:
{essay_text}

Provide exactly ONE word: correct or incorrect"""

    else:
        examples = """
EXAMPLE 1:
Question: "A group of students wrote the following procedure for their investigation. Procedure: 1. Determine the mass of four different samples. 2. Pour vinegar in each of four separate, but identical, containers. 3. Place a sample of one material into one container and label. Repeat with remaining samples, placing a single sample into a single container. 4. After 24 hours, remove the samples from the containers and rinse each sample with distilled water. 5. Allow the samples to sit and dry for 30 minutes. 6. Determine the mass of each sample. The students' data are recorded in the table below. Sample: Marble (Starting Mass: 9.8g, Ending Mass: 9.4g, Difference: -0.4g), Limestone (Starting Mass: 10.4g, Ending Mass: 9.1g, Difference: -1.3g), Wood (Starting Mass: 11.2g, Ending Mass: 11.2g, Difference: 0.0g), Plastic (Starting Mass: 7.2g, Ending Mass: 7.1g, Difference: -0.1g). After reading the group's procedure, describe what additional information you would need in order to replicate the experiment. Make sure to include at least three pieces of information."
Student Answer: "In order to replicate the experiment, I would need to know the amount of each sample that is put into the containers. I would also need to know the environment in which the containers where during the 24 hours. Things such as temperature are necessary. I would also need to know how long the samples should be rinsed after the 24 hours."
Score: 2/3
Reasoning: Answer provides three valid pieces of missing information (sample amount, environmental conditions/temperature, and rinsing duration), meeting the requirement. However, minor grammatical error ("where" instead of "were") and could be more specific about other missing details like vinegar amount or container size.

EXAMPLE 2:
Question: "Now run with these flags: -l 4:100,1:0. These flags specify one process with 4 instructions (all to use the CPU), and one that simply issues an I/O and waits for it to be done. How long does it take to complete both processes? Use -c and -p to find out if you were right."
Student Answer: "It took 10 time units to finish both processes"
Score: 8/15
Reasoning: Answer provides the correct numerical result (10 time units) demonstrating the student ran the simulation and obtained the right output. However, the response lacks explanation of why it takes 10 units, shows no work or process details, and doesn't demonstrate understanding of the underlying concepts of CPU scheduling, I/O operations, or how the two processes interact. A complete answer would explain the execution sequence and justify the result.

EXAMPLE 3:
Question: "When studying the emission sources within the Milky Way, a satellite detected interplanetary clouds containing silicon atoms that have lost five electrons. The ionization energies corresponding to the removal of the third, fourth, and fifth electrons in silicon are 3231, 4356, and 16091 kJ/mol, respectively. Using core charge calculations and your understanding of Coulomb's Law, briefly explain 1) why the removal of each additional electron requires more energy than the removal of the previous one, and 2) the relative magnitude of the values observed."
Student Answer: "Coulomb's Law takes into consideration electron-electron repulsion, nuclear charge, and the distance between the observed electron and the nucleus. The removal of each additional electron requires more energy than the removal of the previous one due to electron-electron repulsion. The effective nuclear charge and radius do not contribute as much because the atomic number is not increasing and the electrons are still in the same shell. The successive increases in ionization energy are due to a reduction in e-e repulsion as electrons are removed. However, between the fourth and fifth ionization energies, the factor increased much more. This increase is due to the inverse relationship in Coulomb's Law between radius and potential energy. The Si5+ atom jumps from the 2p to 2s orbital between the fourth and fifth ionizations, moving closer to the nucleus, which requires much more ionization energy."
Score: 5/8
Reasoning: Answer demonstrates solid understanding of Coulomb's Law and correctly identifies electron-electron repulsion as a key factor in increasing ionization energies. The student accurately explains the dramatic jump between fourth and fifth ionization energies by recognizing the electron shell change (2p to 2s). However, the response contains a significant conceptual error: the student incorrectly states that effective nuclear charge and radius "do not contribute as much" when in fact Zeff increases with each electron removal (fewer electrons shielding the same nuclear charge), which is a primary driver of increasing ionization energy. The explanation of the radius relationship is somewhat unclear and could be more precise. The response would score higher with correct treatment of effective nuclear charge and clearer explanation of how decreased shielding affects successive ionizations.
"""
        system_prompt = f"""You are an expert essay scorer using INDUCTIVE REASONING.

INDUCTIVE PROCESS:
1. Learn patterns from the examples below
2. Identify scoring criteria from the example patterns
3. Apply these learned patterns to score the new essay

SCORING EXAMPLES TO LEARN FROM:
{examples}

From these examples, identify patterns in:
- What makes a high score vs low score
- How content quality affects scoring  
- How writing mechanics impact scores
- What level of development is expected

SCORING RANGE: {score_range}
TASK: {dataset_info.get('description', 'Essay scoring task')}"""

        user_prompt = f"""Based on the patterns you learned from the examples above, score this essay:

QUESTION/PROMPT:
{question}

ESSAY TO SCORE:
{essay_text}

Apply the patterns you identified. Provide your score as a single number within the range {score_range}."""

    return {
        "system": system_prompt,
        "user": user_prompt
    }

def load_essays_from_dataset(dataset_name: str, num_essays: int = None):
    """Load essays from BESESR"""
    try:
        tester = SingleModelTester()
        
        df = tester.download_test_data(dataset_name, num_essays)
        if df is None:
            print(f"  Failed to download from BESESR")
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
        
        print(f"  Loaded {len(essays)} essays from {dataset_name} via BESESR")
        return essays
        
    except Exception as e:
        print(f"  Error loading {dataset_name} from BESESR: {e}")
        import traceback
        traceback.print_exc()
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
                    "X-Title": "S-GRADES Inductive Reasoning Experiment"
                }
            )
            
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            return {
                'response': response,
                'tokens': {
                    'prompt': prompt_tokens,
                    'completion': completion_tokens,
                    'total': prompt_tokens + completion_tokens
                },
                'success': True
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    'response': None,
                    'tokens': {'prompt': 0, 'completion': 0, 'total': 0},
                    'success': False,
                    'error': str(e)
                }
            time.sleep(2 ** attempt)

def run_missing_datasets_evaluation(api_key):
    """Run inductive evaluation on ONLY the 9 missing datasets"""
    
    client = get_client(api_key)
    
    print("="*70)
    print("INDUCTIVE REASONING - MISSING 9 DATASETS - LLAMA 4 SCOUT")
    print("="*70)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing {len(MISSING_DATASETS)} missing datasets")
    print(f"Essays per dataset: {'All' if NUM_ESSAYS is None else NUM_ESSAYS}")
    print()
    
    all_results = {
        'model_code': MODEL_CODE,
        'model_name': MODEL_NAME,
        'reasoning_approach': 'inductive',
        'timestamp': datetime.now().isoformat(),
        'datasets': []
    }
    
    successful_datasets = 0
    
    for dataset_idx, dataset_name in enumerate(MISSING_DATASETS, 1):
        print(f"\n{'='*70}")
        print(f"Dataset {dataset_idx}/{len(MISSING_DATASETS)}: {dataset_name}")
        print(f"{'='*70}")
        
        try:
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
                
                time.sleep(0.5)
            
            dataset_result['essays_evaluated'] = len(dataset_result['predictions'])
            
            if dataset_result['essays_evaluated'] > 0:
                all_results['datasets'].append(dataset_result)
                successful_datasets += 1
            
            print(f"  ✓ Completed: {dataset_result['essays_evaluated']} essays")
            
        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}: {e}")
        
        time.sleep(2)
    
    all_results['successful_datasets'] = successful_datasets
    
    print("\n" + "="*70)
    print("MISSING DATASETS EVALUATION COMPLETE")
    print("="*70)
    print(f"Successful datasets: {successful_datasets}/{len(MISSING_DATASETS)}")
    
    timestamp = int(time.time())
    filename = f"inductive_llama4scout_gradelikeahuman{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")
    return all_results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("ERROR: No API key provided")
        sys.exit(1)
    
    print("\nThis will process 9 missing datasets with ALL essays")
    print("Estimated: ~5,000-6,000 API calls")
    print("Estimated cost: ~$3-5")
    print("Estimated time: 2-4 hours")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == 'y':
        run_missing_datasets_evaluation(api_key)
    else:
        print("Cancelled.")