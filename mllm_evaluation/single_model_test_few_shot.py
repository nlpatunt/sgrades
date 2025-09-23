#!/usr/bin/env python3
"""
Few-Shot Model Testing with Automatic Training Examples
Uses training data from each dataset to provide examples before scoring
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple
import zipfile
import io
import csv
import random

class FewShotModelTester:
    def __init__(self, besesr_url="http://localhost:8000"):
        self.base_url = besesr_url
        self.training_cache = {}
        
    def test_connection(self):
        """Test BESESR connection"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def get_available_datasets(self):
        """Get list of available datasets"""
        try:
            response = requests.get(f"{self.base_url}/api/available-datasets")
            data = response.json()
            return [ds['name'] for ds in data.get('datasets', [])]
        except Exception as e:
            print(f"Error getting datasets: {e}")
            return []
    
    def download_test_data(self, dataset_name: str, num_essays: int = None):
        """Download unlabeled test data from D_{dataset_name}"""
        test_dataset_name = f"{dataset_name}"
        print(f"Downloading test data (unlabeled): {test_dataset_name}")
        
        try:
            response = requests.get(f"{self.base_url}/api/datasets/download/{test_dataset_name}")
            
            if response.status_code != 200:
                print(f"Failed to download {test_dataset_name}: HTTP {response.status_code}")
                return None
            
            # Extract CSV files
            zip_content = zipfile.ZipFile(io.BytesIO(response.content))
            
            # Find test file
            csv_file = None
            for file_info in zip_content.filelist:
                if file_info.filename.endswith('.csv'):
                    if 'test' in file_info.filename.lower():
                        csv_file = file_info.filename
                        break
            
            # If no test file, use any CSV
            if not csv_file:
                for file_info in zip_content.filelist:
                    if file_info.filename.endswith('.csv'):
                        csv_file = file_info.filename
                        break
            
            if not csv_file:
                print("No CSV file found in test dataset")
                return None
            
            # Load CSV
            with zip_content.open(csv_file) as f:
                df = pd.read_csv(f)
            
            total_essays = len(df)
            print(f"Loaded {total_essays} unlabeled test essays from {csv_file}")
            
            # Sample essays only if num_essays is specified
            if num_essays is not None and total_essays > num_essays:
                df = df.sample(n=num_essays, random_state=42)
                print(f"Sampled {num_essays} essays for testing")
            else:
                print(f"Using all {total_essays} essays for testing")
            
            return df
            
        except Exception as e:
            print(f"Failed to download test data: {e}")
            return None

    def simulate_model(self, essay_text: str, range_info=None):
        """Simulate model scoring with dataset-specific range"""
        word_count = len(essay_text.split())
        
        # Handle categorical responses
        if range_info and range_info.get("type") == "categorical":
            categories = range_info["categories"]
            # Simple heuristic: longer text = better category
            if word_count > 100:
                return categories[-1]  # Best category
            elif word_count > 50:
                return categories[len(categories)//2] if len(categories) > 2 else categories[0]
            else:
                return categories[0]  # Worst category
        
        # Handle numeric scoring
        if range_info:
            min_score, max_score = range_info["min"], range_info["max"]
        else:
            min_score, max_score = 0, 5
        
        # Simple heuristic scoring based on text length
        if word_count < 50:
            base_score = min_score + (max_score - min_score) * 0.3
        elif word_count < 150:
            base_score = min_score + (max_score - min_score) * 0.5
        elif word_count < 300:
            base_score = min_score + (max_score - min_score) * 0.7
        else:
            base_score = min_score + (max_score - min_score) * 0.8
        
        # Add randomness
        import random
        variation_range = (max_score - min_score) * 0.15
        variation = random.uniform(-variation_range, variation_range)
        
        score = max(min_score, min(max_score, base_score + variation))
        
        # Round appropriately
        if isinstance(score, float) and (max_score - min_score) < 10:
            return round(score, 1)
        else:
            return round(score, 2)
    
    def download_training_data(self, dataset_name: str) -> pd.DataFrame:
        """Download training data for few-shot examples"""
        if dataset_name in self.training_cache:
            return self.training_cache[dataset_name]
        
        try:
            print(f"Downloading training data for {dataset_name}...")
            response = requests.get(f"{self.base_url}/api/datasets/download/{dataset_name}")
            
            if response.status_code != 200:
                print(f"Failed to download {dataset_name}: HTTP {response.status_code}")
                return None
            
            # Extract and find train.csv
            zip_content = zipfile.ZipFile(io.BytesIO(response.content))
            train_file = None
            
            for file_info in zip_content.filelist:
                if file_info.filename.endswith('.csv'):
                    if 'train' in file_info.filename.lower():
                        train_file = file_info.filename
                        break
            
            if not train_file:
                print(f"No train.csv found in {dataset_name}")
                return None
            
            # Load training data
            with zip_content.open(train_file) as f:
                df = pd.read_csv(f)
            
            self.training_cache[dataset_name] = df
            print(f"Loaded {len(df)} training examples from {train_file}")
            return df
            
        except Exception as e:
            print(f"Failed to download training data: {e}")
            return None
    
    def get_dataset_columns(self, dataset_name: str):
        """Get correct column names for each dataset"""
        
        ID_COLUMNS = {
            "D_ASAP-AES": "essay_id",
            "D_ASAP2": "essay_id", 
            "D_BEEtlE_2way": "ID",
            "D_BEEtlE_3way": "ID",
            "D_SciEntSBank_2way": "ID",
            "D_SciEntSBank_3way": "ID",
            "D_persuade_2": "essay_id_comp",
            "D_CSEE": "index",
            "D_Mohlar": "ID",
            "D_ASAP_plus_plus": "essay_id",
            "D_Rice_Chem_Q1": "sis_id",
            "D_Rice_Chem_Q2": "sis_id",
            "D_Rice_Chem_Q3": "sis_id",
            "D_Rice_Chem_Q4": "sis_id",
            "D_Ielts_Writing_Dataset": "ID",
            "D_Ielst_Writing_Task_2_Dataset": "ID",
        }
        
        SCORE_COLUMNS = {
            "D_ASAP-AES": "domain1_score",
            "D_ASAP2": "score",
            "D_BEEtlE_2way": "label", 
            "D_BEEtlE_3way": "label",
            "D_SciEntSBank_2way": "label",
            "D_SciEntSBank_3way": "label",
            "D_persuade_2": "holistic_essay_score",
            "D_CSEE": "overall_score",
            "D_Mohlar": "grade",
            "D_ASAP_plus_plus": "overall_score",
            "D_Rice_Chem_Q1": "Score",
            "D_Rice_Chem_Q2": "Score",
            "D_Rice_Chem_Q3": "Score",
            "D_Rice_Chem_Q4": "Score",
            "D_Ielts_Writing_Dataset": "Overall_Score",
            "D_Ielst_Writing_Task_2_Dataset": "band_score",
        }
        
        TEXT_COLUMNS = ['essay', 'essay_text', 'full_text', 'text', 'answer', 'student_answer', 'response']
        
        return {
            'id_col': ID_COLUMNS.get(dataset_name, "ID"),
            'score_col': SCORE_COLUMNS.get(dataset_name, "score"),
            'text_cols': TEXT_COLUMNS
        }
    
    def select_training_examples(self, training_df: pd.DataFrame, dataset_name: str, num_examples: int = 5) -> List[Dict]:
        """Select diverse examples from training data"""
        
        columns = self.get_dataset_columns(dataset_name)
        score_col = columns['score_col']
        text_cols = columns['text_cols']
        
        # Find text column
        text_col = None
        for col in text_cols:
            if col in training_df.columns:
                text_col = col
                break
        
        if not text_col or score_col not in training_df.columns:
            print(f"Required columns not found in {dataset_name}")
            print(f"Available columns: {list(training_df.columns)}")
            return []
        
        # Filter valid examples
        valid_df = training_df[
            (training_df[text_col].notna()) & 
            (training_df[text_col].str.len() > 30) &  # At least 30 characters
            (training_df[score_col].notna())
        ].copy()
        
        if len(valid_df) == 0:
            print(f"No valid examples found in {dataset_name}")
            return []
        
        examples = []
        
        # For categorical datasets (BEEtlE, SciEntSBank)
        if dataset_name in ["D_BEEtlE_2way", "D_BEEtlE_3way", "D_SciEntSBank_2way", "D_SciEntSBank_3way"]:
            unique_labels = valid_df[score_col].unique()
            examples_per_label = max(1, num_examples // len(unique_labels))
            
            for label in unique_labels:
                label_examples = valid_df[valid_df[score_col] == label]
                if len(label_examples) > 0:
                    sample_size = min(examples_per_label, len(label_examples))
                    selected = label_examples.sample(n=sample_size, random_state=42)
                    
                    for _, row in selected.iterrows():
                        examples.append({
                            "text": str(row[text_col]).strip()[:400],  # Limit length
                            "score": row[score_col],
                            "reasoning": f"This answer demonstrates {row[score_col]} understanding."
                        })
        
        # For numeric datasets
        else:
            valid_df[score_col] = pd.to_numeric(valid_df[score_col], errors='coerce')
            valid_df = valid_df.dropna(subset=[score_col])
            
            if len(valid_df) == 0:
                return []
            
            # Select examples from different score ranges
            min_score = valid_df[score_col].min()
            max_score = valid_df[score_col].max()
            
            # Create percentile-based ranges
            percentiles = np.linspace(0, 100, num_examples + 1)
            
            for i in range(len(percentiles) - 1):
                low_pct, high_pct = percentiles[i], percentiles[i + 1]
                low_val = np.percentile(valid_df[score_col], low_pct)
                high_val = np.percentile(valid_df[score_col], high_pct)
                
                range_examples = valid_df[
                    (valid_df[score_col] >= low_val) & 
                    (valid_df[score_col] <= high_val)
                ]
                
                if len(range_examples) > 0:
                    example = range_examples.sample(n=1, random_state=42).iloc[0]
                    
                    examples.append({
                        "text": str(example[text_col]).strip()[:400],
                        "score": example[score_col],
                        "reasoning": f"Score {example[score_col]:.1f} - {self.get_score_description(example[score_col], min_score, max_score)}"
                    })
        
        return examples[:num_examples]
    
    def get_score_description(self, score: float, min_score: float, max_score: float) -> str:
        """Get description for score level"""
        if max_score == min_score:
            return "average quality"
            
        relative_position = (score - min_score) / (max_score - min_score)
        
        if relative_position < 0.2:
            return "poor quality work"
        elif relative_position < 0.4:
            return "below average quality"
        elif relative_position < 0.6:
            return "average quality work"
        elif relative_position < 0.8:
            return "good quality work"
        else:
            return "excellent quality work"
    
    def get_dataset_range_info(self, dataset_name: str, row=None):
        """Get scoring range information for different datasets"""
        
        range_configs = {
            "D_ASAP-AES": {"type": "fixed_range", "min": 0, "max": 60, "description": "0-60 scale (60 = exceptional essay)"},
            "D_ASAP2": {"type": "fixed_range", "min": 0, "max": 60, "description": "0-60 scale (varies by essay set)"},
            "D_BEEtlE_2way": {"type": "categorical", "categories": ["incorrect", "correct"], "description": "correct/incorrect classification"},
            "D_BEEtlE_3way": {"type": "categorical", "categories": ["incorrect", "partial_correct", "correct"], "description": "three-way classification"},
            "D_SciEntSBank_2way": {"type": "categorical", "categories": ["incorrect", "correct"], "description": "correct/incorrect classification"},
            "D_SciEntSBank_3way": {"type": "categorical", "categories": ["incorrect", "contradictory", "correct"], "description": "three-way classification"},
            "D_persuade_2": {"type": "fixed_range", "min": 1, "max": 5, "description": "1-5 holistic essay scale"},
            "D_CSEE": {"type": "fixed_range", "min": 0, "max": 100, "description": "0-100 percentage scale"},
            "D_Mohlar": {"type": "fixed_range", "min": 0, "max": 10, "description": "0-10 grade scale"},
            "D_Ielts_Writing_Dataset": {"type": "fixed_range", "min": 0, "max": 9, "description": "0-9 IELTS band scale"},
            "D_Ielst_Writing_Task_2_Dataset": {"type": "fixed_range", "min": 0, "max": 9, "description": "0-9 IELTS band scale"},
        }
        
        return range_configs.get(dataset_name, {"type": "fixed_range", "min": 0, "max": 5, "description": "0-5 scale"})
    
    def create_few_shot_prompt(self, dataset_name: str, target_essay: str, range_info: Dict) -> str:
        """Create few-shot prompt with training examples"""
        
        # Get training data
        training_df = self.download_training_data(dataset_name)
        if training_df is None:
            return self.create_zero_shot_fallback(target_essay, range_info)
        
        # Select examples
        examples = self.select_training_examples(training_df, dataset_name, num_examples=5)
        if not examples:
            return self.create_zero_shot_fallback(target_essay, range_info)
        
        # Build prompt
        prompt = "You are an expert grader. Here are examples from the training data showing how to score essays:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Training Example {i}:\n"
            prompt += f"Text: {example['text']}\n"
            prompt += f"Score: {example['score']}\n"
            prompt += f"Explanation: {example['reasoning']}\n\n"
        
        # Add target essay
        prompt += "Now score this new essay using the same standards shown in the training examples:\n\n"
        prompt += f"Text: {target_essay[:800]}...\n\n"  # Limit target length
        
        if range_info and range_info.get("type") == "categorical":
            categories = range_info["categories"]
            prompt += f"Choose from these options: {', '.join(categories)}\n"
        else:
            if range_info:
                min_score, max_score = range_info["min"], range_info["max"]
                prompt += f"Provide a numerical score between {min_score} and {max_score}:\n"
            else:
                prompt += "Provide a numerical score:\n"
        
        prompt += "Score:"
        return prompt
    
    def create_zero_shot_fallback(self, essay_text: str, range_info: Dict) -> str:
        """Fallback to zero-shot if training data unavailable"""
        return f"Rate this essay on a 0-5 scale where 5=Excellent, 0=Very Poor.\n\nEssay: {essay_text}\n\nScore:"
    
    def prepare_essays_for_prediction(self, df: pd.DataFrame, dataset_name: str):
        """Extract essays from test dataframe with range information"""
        essays = []
        print(f"DEBUG: Available columns in {dataset_name}: {list(df.columns)}")
        print(f"DEBUG: Dataset shape: {df.shape}")
        
        columns = self.get_dataset_columns(dataset_name)
        id_column = columns['id_col']
        text_cols = columns['text_cols']
        
        # Find essay text column
        text_col = None
        for col in text_cols:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            print(f"No text column found in {dataset_name}")
            return []
        
        for idx, row in df.iterrows():
            essay_text = str(row[text_col]).strip()
            
            if len(essay_text) < 5:  # Skip very short essays
                continue
                
            essay_id = row.get(id_column, f"{dataset_name}_test_{idx}")
            range_info = self.get_dataset_range_info(dataset_name, row)
            
            essays.append({
                'id': essay_id,
                'text': essay_text,
                'range_info': range_info,
                'question': str(row.get('Question', '')),
                'prompt': str(row.get('prompt', f'Essay prompt for {dataset_name}'))
            })
        
        print(f"Prepared {len(essays)} essays for prediction (using ID column: {id_column})")
        return essays
    
    def call_gemini_flash_few_shot(self, essay_text: str, dataset_name: str, range_info: Dict):
        """Call Gemini Flash with few-shot prompt"""
        import openai
        
        try:
            prompt = self.create_few_shot_prompt(dataset_name, essay_text, range_info)
            
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            response = client.chat.completions.create(
                model="google/gemini-flash-1.5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0.3
            )
            
            time.sleep(3)  # Rate limiting
            return self.parse_response(response.choices[0].message.content.strip(), range_info)
            
        except Exception as e:
            print(f"Gemini Flash API error: {e}")
            time.sleep(5)
            return None
    
    def call_api_model(self, essay_text: str, prompt: str, model_type: str, range_info=None, question="", dataset_name=None):
        """Route to appropriate model API with few-shot prompting"""
        if model_type == "simulation":
            return self.simulate_model(essay_text, range_info)
        if model_type == "gemini-flash":
            return self.call_gemini_flash_few_shot(essay_text, dataset_name, range_info)
        # Add other models as needed
        else:
            print(f"Model {model_type} not implemented for few-shot")
            return None
    
    def parse_response(self, response_text: str, range_info: Dict):
        """Parse model response"""
        import re
        
        if range_info.get("type") == "categorical":
            response_lower = response_text.lower().strip()
            for category in range_info["categories"]:
                if category.lower() in response_lower:
                    return category
            return range_info["categories"][0]  # Default
        else:
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                score = float(numbers[0])
                if range_info:
                    return max(range_info["min"], min(range_info["max"], score))
                return score
            return 2.5  # Default
    
    def evaluate_with_ground_truth(self, dataset_name: str, predictions: List[Dict], model_type: str = "Unknown", leaderboard_name: str = None):
        """Submit predictions to BESESR for evaluation against ground truth"""
        try:
            # Dataset submission requirements
            SUBMISSION_REQUIREMENTS = {
                "D_ASAP-AES": ["essay_id", "domain1_score"],
                "D_ASAP2": ["essay_id", "score"],
                "D_BEEtlE_2way": ["ID", "label"],
                "D_BEEtlE_3way": ["ID", "label"],
                "D_SciEntSBank_2way": ["ID", "label"],
                "D_SciEntSBank_3way": ["ID", "label"],
                "D_persuade_2": ["essay_id_comp", "holistic_essay_score"],
                "D_CSEE": ["index", "overall_score"],
                "D_Mohlar": ["ID", "grade"],
                "D_ASAP_plus_plus": ["essay_id", "overall_score"],
                "D_Ielts_Writing_Dataset": ["ID", "Overall_Score"],
                "D_Ielst_Writing_Task_2_Dataset": ["ID", "band_score"],
            }
            
            # Get the required format for this dataset
            if dataset_name not in SUBMISSION_REQUIREMENTS:
                print(f"Warning: {dataset_name} not in submission requirements, using generic format")
                id_col, score_col = "essay_id", "predicted_score"
            else:
                id_col, score_col = SUBMISSION_REQUIREMENTS[dataset_name]
            
            # Check for duplicate IDs
            submission_ids = [str(pred['id']) for pred in predictions]
            unique_ids = set(submission_ids)
            
            if len(submission_ids) != len(unique_ids):
                print(f"Found duplicate IDs in predictions!")
                # Remove duplicates - keep first occurrence
                seen_ids = set()
                filtered_predictions = []
                for pred in predictions:
                    pred_id = str(pred['id'])
                    if pred_id not in seen_ids:
                        filtered_predictions.append(pred)
                        seen_ids.add(pred_id)
                predictions = filtered_predictions
                print(f"After removing duplicates: {len(predictions)} predictions")
            else:
                print(f"✓ No duplicate IDs found in {len(predictions)} predictions")
            
            # Create CSV content
            csv_content = io.StringIO()
            writer = csv.writer(csv_content)
            writer.writerow([id_col, score_col])
            
            for pred in predictions:
                pred_id = pred['id']
                final_score = pred['predicted_score']
                
                # Round appropriately
                if isinstance(final_score, (int, float)):
                    if "Ielts" in dataset_name:
                        final_score = round(final_score * 2) / 2  # Round to nearest 0.5
                    else:
                        final_score = round(final_score, 2)
                
                writer.writerow([pred_id, final_score])
            
            # Prepare file for submission
            csv_content.seek(0)
            csv_data = csv_content.getvalue()
            
            files = {'file': ('predictions.csv', csv_data, 'text/csv')}
            
            data = {
                'dataset_name': dataset_name,
                'model_name': leaderboard_name or f'few-shot-{model_type}_test1',
                'submitter_name': leaderboard_name or f'few-shot-{model_type}_test1',
                'submitter_email': 'fewshot.researcher@example.com',
                'description': f'Few-shot evaluation using {model_type} with training examples'
            }
                    
            print(f"Submitting {len(predictions)} predictions to {dataset_name} for ground truth evaluation...")
            print(f"Format: {id_col}, {score_col} ({len(predictions)} predictions)")
            
            # Submit to BESESR
            response = requests.post(
                f"{self.base_url}/api/submissions/test-single-dataset",
                files=files, 
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    evaluation = result.get('evaluation', {})
                    metrics = result.get('metrics', {}) or evaluation.get('metrics', {})
                    
                    if metrics:
                        print(f"✓ Ground truth evaluation successful: {len(predictions)} essays evaluated")
                        return metrics
                    else:
                        print("✗ No metrics returned from ground truth evaluation")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    validation_errors = result.get('validation_errors', [])
                    print(f"✗ Ground truth evaluation failed: {error_msg}")
                    if validation_errors:
                        print(f"Validation errors: {validation_errors}")
            else:
                print(f"✗ HTTP {response.status_code}: {response.text[:200]}")
            
            return None
            
        except Exception as e:
            print(f"✗ Exception during ground truth evaluation: {e}")
            return None
    
    def run_single_test(self, dataset_name: str, model_type: str, num_essays: int = None, leaderboard_name: str = None):
        """Complete few-shot test with leaderboard submission"""
        
        if dataset_name.startswith("D_"):
            ground_truth_dataset = dataset_name[2:]  # Remove D_ prefix
        else:
            ground_truth_dataset = dataset_name
            
        print("=" * 60)
        print(f"FEW-SHOT MODEL TEST")
        print(f"Model: {model_type.upper()}")
        print(f"Dataset: {dataset_name}")
        print(f"Test Data: {dataset_name} (unlabeled)")
        print(f"Ground Truth: {ground_truth_dataset} (labeled)")
        print(f"Essays: {num_essays if num_essays else 'All'}")
        print("=" * 60)
        
        # Test BESESR connection
        if not self.test_connection():
            print("BESESR not accessible at localhost:8000")
            return None
        print("✓ BESESR connected")
        
        # Download test data
        df = self.download_test_data(dataset_name, num_essays)
        if df is None:
            print(f"Failed to download test data from {dataset_name}")
            return None
        
        # Prepare essays for prediction
        essays = self.prepare_essays_for_prediction(df, dataset_name)
        if not essays:
            print("No essays prepared for prediction")
            return None

        print(f"\nGenerating few-shot predictions with {model_type} on {len(essays)} test essays...")
        
        predictions = []
        results_detail = []
        
        # Generate predictions for each test essay
        for i, essay in enumerate(essays, 1):
            print(f"Essay {i}/{len(essays)}: ", end="", flush=True)
            
            start_time = time.time()
            predicted_score = self.call_api_model(
                essay['text'], 
                essay['prompt'], 
                model_type,
                range_info=essay.get('range_info'),
                question=essay.get('question', ''),
                dataset_name=dataset_name
            )
            end_time = time.time()
            
            if predicted_score is not None:
                predictions.append(predicted_score)
                
                # Show range in output
                range_info = essay.get('range_info', {})
                if range_info.get("type") == "categorical":
                    range_display = f" ({range_info['description']})"
                else:
                    max_val = range_info.get('max', 5)
                    range_display = f"/{max_val}"
                
                print(f"Predicted: {predicted_score}{range_display}, Time: {end_time-start_time:.1f}s")
                
                results_detail.append({
                    'essay_id': essay['id'],
                    'predicted': predicted_score,
                    'text_preview': essay['text'][:100] + "...",
                    'time': end_time - start_time,
                    'range_info': range_info
                })
            else:
                print("FAILED")
        
        if not predictions:
            print("No successful predictions")
            return None
        
        print(f"\nSubmitting {len(predictions)} predictions for ground truth evaluation...")
        
        # Prepare predictions for evaluation
        besesr_predictions = []
        for detail in results_detail:
            besesr_predictions.append({
                'id': detail['essay_id'],
                'predicted_score': detail['predicted']
            })

        metrics = self.evaluate_with_ground_truth(dataset_name, besesr_predictions, 
                                            model_type, leaderboard_name)
        
        if metrics:
            print("✓ Ground truth evaluation completed successfully")
            
            # Add prediction statistics
            if predictions and isinstance(predictions[0], str):
                # Handle categorical predictions
                unique_preds = list(set(predictions))
                metrics.update({
                    'num_predictions': len(predictions),
                    'prediction_mean': 0.0,
                    'prediction_std': 0.0, 
                    'prediction_min': unique_preds[0] if unique_preds else 'unknown',
                    'prediction_max': unique_preds[-1] if unique_preds else 'unknown',
                })
            else:
                # Handle numeric predictions
                metrics.update({
                    'num_predictions': len(predictions),
                    'prediction_mean': np.mean(predictions),
                    'prediction_std': np.std(predictions),
                    'prediction_min': np.min(predictions),
                    'prediction_max': np.max(predictions),
                })
        else:
            print("✗ Ground truth evaluation failed")
            
            # Provide basic statistics without ground truth
            if predictions and isinstance(predictions[0], str):
                unique_preds = list(set(predictions))
                metrics = {
                    'num_predictions': len(predictions),
                    'prediction_mean': 0.0,
                    'prediction_std': 0.0,
                    'prediction_min': unique_preds[0] if unique_preds else 'unknown',
                    'prediction_max': unique_preds[-1] if unique_preds else 'unknown',
                    'mean_absolute_error': 0.0,
                    'root_mean_squared_error': 0.0,
                    'pearson_correlation': 0.0,
                    'accuracy_within_1.0': 0.0,
                    'accuracy_within_0.5': 0.0
                }
            else:
                # Handle numeric predictions
                metrics = {
                    'num_predictions': len(predictions),
                    'prediction_mean': np.mean(predictions),
                    'prediction_std': np.std(predictions),
                    'prediction_min': np.min(predictions),
                    'prediction_max': np.max(predictions),
                    'mean_absolute_error': 0.0,
                    'root_mean_squared_error': 0.0,
                    'pearson_correlation': 0.0,
                    'accuracy_within_1.0': 0.0,
                    'accuracy_within_0.5': 0.0
                }
            
            for detail in results_detail:
                detail['actual'] = 'Evaluation failed'
        
        # Display comprehensive results
        self.display_results(model_type, dataset_name, metrics, results_detail)
        
        return {
            'model': f"{model_type}-few-shot",
            'dataset': dataset_name,
            'test_data_source': dataset_name,
            'ground_truth_source': ground_truth_dataset,
            'metrics': metrics,
            'details': results_detail,
            'success': metrics.get('mean_absolute_error', 0) > 0 or metrics.get('f1_score', 0) > 0  # True if we got real evaluation
        }
    
    def display_results(self, model_type: str, dataset_name: str, metrics: dict, details: list):
        """Display formatted results"""
        
        if dataset_name.startswith("D_"):
            test_data_name = dataset_name
            ground_truth_name = dataset_name[2:]
        else:
            test_data_name = f"D_{dataset_name}"
            ground_truth_name = dataset_name
        
        print("\n" + "=" * 60)
        print("FEW-SHOT RESULTS")
        print("=" * 60)
        
        print(f"Model: {model_type.upper()}-FEW-SHOT")
        print(f"Dataset: {dataset_name}")
        print(f"Test Data: {test_data_name} (unlabeled)")
        print(f"Ground Truth: {ground_truth_name} (labeled)")
        print(f"Essays Evaluated: {metrics.get('num_predictions', 0)}")
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"├─ Mean Absolute Error: {metrics.get('mean_absolute_error', 0):.3f}")
        print(f"├─ Root Mean Squared Error: {metrics.get('root_mean_squared_error', 0):.3f}")
        print(f"├─ Correlation: {metrics.get('pearson_correlation', 0):.3f}")
        print(f"├─ Quadratic Weighted Kappa: {metrics.get('quadratic_weighted_kappa', 0):.3f}")
        print(f"├─ F1 Score: {metrics.get('f1_score', 0):.3f}")
        print(f"├─ Precision: {metrics.get('precision', 0):.3f}")
        print(f"├─ Recall: {metrics.get('recall', 0):.3f}")
        print(f"├─ Accuracy within 1.0: {metrics.get('accuracy_within_1.0', 0):.1%}")
        print(f"└─ Accuracy within 0.5: {metrics.get('accuracy_within_0.5', 0):.1%}")
        
        print(f"\nSCORE STATISTICS:")
        
        pred_mean = metrics.get('prediction_mean', 0)
        pred_std = metrics.get('prediction_std', 0)
        pred_min = metrics.get('prediction_min', 0)
        pred_max = metrics.get('prediction_max', 0)
        
        is_categorical = any(isinstance(val, str) for val in [pred_mean, pred_std, pred_min, pred_max])

        if is_categorical:
            print(f"├─ Predicted Mean: {pred_mean}")
            print(f"├─ Predicted Std: {pred_std}")
            print(f"├─ Predicted Min: {pred_min}")
            print(f"└─ Predicted Max: {pred_max}")
        else:
            print(f"├─ Predicted Mean: {pred_mean:.2f}")
            print(f"├─ Predicted Std: {pred_std:.2f}")
            print(f"├─ Predicted Min: {pred_min:.2f}")
            print(f"└─ Predicted Max: {pred_max:.2f}")
        
        print(f"\nSAMPLE PREDICTIONS:")
        for i, detail in enumerate(details[:5], 1):
            predicted = detail['predicted']
            actual = detail.get('actual', 'Unknown')
            
            range_info = detail.get('range_info', {})
            if range_info.get("type") == "categorical":
                range_display = f" ({range_info['description']})"
            else:
                max_val = range_info.get('max', 5)
                range_display = f"/{max_val}"
            
            print(f"{i}. ID: {detail['essay_id']}")
            print(f"   Predicted: {predicted}{range_display} | Time: {detail['time']:.1f}s")
            print(f"   Text: {detail['text_preview']}")
            print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 1 model on 1 dataset with few-shot prompting")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., D_ASAP-AES)")
    parser.add_argument("--model", required=True, 
                   choices=["gemini-flash", "gpt4o", "gpt4o-mini", "claude-sonnet", "claude-haiku", "simulation"],
                   help="Model to test")
    parser.add_argument("--essays", type=int, default=None, help="Number of essays to test (default: all)")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    tester = FewShotModelTester()
    
    if args.list_datasets:
        print("Available datasets:")
        datasets = tester.get_available_datasets()
        for i, ds in enumerate(datasets, 1):
            print(f"{i:2d}. {ds}")
        return
    
    leaderboard_name = f"few-shot-{args.model}_test1"
    
    result = tester.run_single_test(args.dataset, args.model, args.essays, leaderboard_name) 
    
    if result and result['success']:
        print(f"\nFew-shot test completed successfully!")
        print(f"Leaderboard name: {leaderboard_name}")
        print(f"Correlation: {result['metrics'].get('pearson_correlation', 0):.3f}")
        print(f"MAE: {result['metrics'].get('mean_absolute_error', 0):.3f}")
    else:
        print("Few-shot test completed but ground truth evaluation failed!")

if __name__ == "__main__":
    main()