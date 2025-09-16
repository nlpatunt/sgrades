#!/usr/bin/env python3
"""
Single Model Single Dataset Test with D_ prefix logic and dynamic range handling
Test data from D_{dataset} (unlabeled) vs Ground truth from {dataset} (labeled)
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

class SingleModelTester:
    def __init__(self, besesr_url="http://localhost:8000"):
        self.base_url = besesr_url
        
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
            response = requests.get(f"{self.base_url}/api/available-datasets")  # Fixed endpoint
            data = response.json()
            return [ds['name'] for ds in data.get('datasets', [])]
        except Exception as e:
            print(f"Error getting datasets: {e}")
            return []
    
    def get_dataset_range_info(self, dataset_name: str, row=None):
        """Get scoring range information for different datasets"""
        
        # Dataset-specific range configurations
        range_configs = {
            # Regrading dataset uses full_marks column for dynamic scoring
            "Regrading_Dataset_J2C": {
                "type": "dynamic_column",
                "column": "Full marks",
                "description": "points (perfect answer)"
            },
            "D_Regrading_Dataset_J2C": {
                "type": "dynamic_column", 
                "column": "Full marks",
                "description": "points (perfect answer)"
            },
            
            # Fixed range datasets
            "ASAP-AES": {
                "type": "fixed_range",
                "min": 0, "max": 60,
                "description": "0-60 scale (60 = exceptional essay)"
            },
            "D_ASAP-AES": {
                "type": "fixed_range",
                "min": 0, "max": 60, 
                "description": "0-60 scale (60 = exceptional essay)"
            },
            
            "ASAP2": {
                "type": "fixed_range",
                "min": 0, "max": 60,
                "description": "0-60 scale (varies by essay set)"
            },
            "D_ASAP2": {
                "type": "fixed_range",
                "min": 0, "max": 60,
                "description": "0-60 scale (varies by essay set)"
            },
            
            "Ielts_Writing_Dataset": {
                "type": "fixed_range",
                "min": 0, "max": 9,
                "description": "0-9 IELTS band scale (9 = expert level)"
            },
            "D_Ielts_Writing_Dataset": {
                "type": "fixed_range",
                "min": 0, "max": 9,
                "description": "0-9 IELTS band scale (9 = expert level)"
            },
            
            "BEEtlE_2way": {
                "type": "categorical",
                "categories": ["incorrect", "correct"],
                "description": "correct/incorrect classification"
            },
            "D_BEEtlE_2way": {
                "type": "categorical",
                "categories": ["incorrect", "correct"],
                "description": "correct/incorrect classification"
            },
            
            "BEEtlE_3way": {
                "type": "categorical",
                "categories": ["incorrect", "partial_correct", "correct"],
                "description": "three-way classification"
            },
            "D_BEEtlE_3way": {
                "type": "categorical",
                "categories": ["incorrect", "partial_correct", "correct"],
                "description": "three-way classification"
            },
            
            "SciEntSBank_2way": {
                "type": "categorical",
                "categories": ["incorrect", "correct"],
                "description": "correct/incorrect classification"
            },
            "D_SciEntSBank_2way": {
                "type": "categorical",
                "categories": ["incorrect", "correct"],
                "description": "correct/incorrect classification"
            },
            
            "SciEntSBank_3way": {
                "type": "categorical",
                "categories": ["incorrect", "contradictory", "correct"],
                "description": "three-way classification"
            },
            "D_SciEntSBank_3way": {
                "type": "categorical",
                "categories": ["incorrect", "contradictory", "correct"],
                "description": "three-way classification"
            },
            
            "persuade_2": {
                "type": "fixed_range",
                "min": 1, "max": 5,
                "description": "1-5 holistic essay scale"
            },
            "D_persuade_2": {
                "type": "fixed_range",
                "min": 1, "max": 5,
                "description": "1-5 holistic essay scale"
            },
            
            # Add more datasets as needed
            "EFL": {
                "type": "fixed_range",
                "min": 0, "max": 6,
                "description": "0-6 scale (EFL scoring)"
            },
            "D_EFL": {
                "type": "fixed_range",
                "min": 0, "max": 6,
                "description": "0-6 scale (EFL scoring)"
            },
            
            "Mohlar": {
                "type": "fixed_range",
                "min": 0, "max": 10,
                "description": "0-10 grade scale"
            },
            "D_Mohlar": {
                "type": "fixed_range",
                "min": 0, "max": 10,
                "description": "0-10 grade scale"
            }
        }
        
        config = range_configs.get(dataset_name, {
            "type": "fixed_range", 
            "min": 0, "max": 5,
            "description": "0-5 scale (5 = excellent)"
        })
        
        # Handle dynamic ranges from columns
        if config["type"] == "dynamic_column" and row is not None:
            column = config["column"]
            if column in row and pd.notna(row[column]):
                max_score = float(row[column])
                return {
                    "type": "dynamic_range",
                    "min": 0,
                    "max": max_score,
                    "description": f"0-{max_score} {config['description']}"
                }
        
        return config
    
    def download_test_data(self, dataset_name: str, num_essays: int = None):
        """Download unlabeled test data from D_{dataset_name}"""
        test_dataset_name = f"{dataset_name}"
        print(f"Downloading test data (unlabeled): {test_dataset_name}")
        
        try:
            response = requests.get(f"{self.base_url}/api/datasets/download/{test_dataset_name}")
            
            if response.status_code != 200:
                print(f"Failed to download {test_dataset_name}: HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}")
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
    
    def prepare_essays_for_prediction(self, df: pd.DataFrame, dataset_name: str):
        """Extract essays from test dataframe with range information"""
        essays = []
        print(f"DEBUG: Available columns in {dataset_name}: {list(df.columns)}")
        print(f"DEBUG: Dataset shape: {df.shape}")
        print(f"DEBUG: First few rows:")
        print(df.head(2))

        # Complete Dataset-specific column mapping with D_ versions
        ID_COLUMNS = {
            "ASAP-AES": "essay_id",
            "D_ASAP-AES": "essay_id",
            "ASAP2": "essay_id",
            "D_ASAP2": "essay_id", 
            "ASAP-SAS": "Id",
            "D_ASAP-SAS": "Id",
            "ASAP_plus_plus": "essay_id",
            "D_ASAP_plus_plus": "essay_id",
            "CSEE": "index",
            "D_CSEE": "index",
            "persuade_2": "essay_id_comp",
            "D_persuade_2": "essay_id_comp",
            "Rice_Chem_Q1": "sis_id",
            "D_Rice_Chem_Q1": "sis_id",
            "Rice_Chem_Q2": "sis_id",
            "D_Rice_Chem_Q2": "sis_id",
            "Rice_Chem_Q3": "sis_id",
            "D_Rice_Chem_Q3": "sis_id",
            "Rice_Chem_Q4": "sis_id",
            "D_Rice_Chem_Q4": "sis_id",
            "BEEtlE_2way": "ID",
            "D_BEEtlE_2way": "ID",
            "BEEtlE_3way": "ID",
            "D_BEEtlE_3way": "ID",
            "SciEntSBank_2way": "ID",
            "D_SciEntSBank_2way": "ID",
            "SciEntSBank_3way": "ID",
            "D_SciEntSBank_3way": "ID",
            "EFL": "ID",
            "D_EFL": "ID",
            "Mohlar": "ID",
            "D_Mohlar": "ID",
            "Ielts_Writing_Dataset": "ID",
            "D_Ielts_Writing_Dataset": "ID",
            "Ielst_Writing_Task_2_Dataset": "ID",
            "D_Ielst_Writing_Task_2_Dataset": "ID",
            "Regrading_Dataset_J2C": "ID",
            "D_Regrading_Dataset_J2C": "ID",
            "grade_like_a_human_dataset_os_q1": "ID",
            "D_grade_like_a_human_dataset_os_q1": "ID",
            "grade_like_a_human_dataset_os_q2": "ID",
            "D_grade_like_a_human_dataset_os_q2": "ID",
            "grade_like_a_human_dataset_os_q3": "ID",
            "D_grade_like_a_human_dataset_os_q3": "ID",
            "grade_like_a_human_dataset_os_q4": "ID",
            "D_grade_like_a_human_dataset_os_q4": "ID",
            "grade_like_a_human_dataset_os_q5": "ID",
            "D_grade_like_a_human_dataset_os_q5": "ID",
        }
        
        # Get correct ID column for this dataset
        id_column = ID_COLUMNS.get(dataset_name, "ID")
        
        # Common essay text column names to try (case-sensitive)
        essay_columns = ['full_text','essay', 'essay_text', 'Essay', 'text', 'Text', 'answer', 'Answer', 'student_answer', 'Student_Answer','response', 'Response', 'student_response', 'Student_Response']
        
        for idx, row in df.iterrows():
            # Find essay text
            essay_text = ""
            found_column = None
            for col in essay_columns:
                if col in row and pd.notna(row[col]):
                    essay_text = str(row[col]).strip()
                    found_column = col
                    break
            
            if len(essay_text) < 5:  # Skip very short essays
                print(f"DEBUG: Skipping essay {idx} - too short ({len(essay_text)} chars)")
                continue
                
            # Get ID using correct column name
            essay_id = row.get(id_column, f"{dataset_name}_test_{idx}")
            
            # Get dataset-specific range info (may be dynamic based on row data)
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
    
    def call_api_model(self, essay_text: str, prompt: str, model_type: str, range_info=None, question=""):
        """Route to appropriate model API"""
        
        if model_type == "gpt4o":
            return self.call_gpt4o(essay_text, prompt, range_info, question)
        elif model_type == "gpt4o-mini":
            return self.call_gpt4o_mini(essay_text, prompt, range_info, question)
        elif model_type == "claude-sonnet":
            return self.call_claude_sonnet(essay_text, prompt, range_info, question)
        elif model_type == "claude-haiku":
            return self.call_claude_haiku(essay_text, prompt, range_info, question)
        elif model_type == "gemini-pro":
            return self.call_gemini_pro(essay_text, prompt, range_info, question)
        elif model_type == "gemini-flash":
            return self.call_gemini_flash(essay_text, prompt, range_info, question)
        elif model_type == "simulation":
            return self.simulate_model(essay_text, range_info)
        else:
            print(f"Model {model_type} not implemented")
            return None
        
    def call_gemini_pro(self, essay_text: str, prompt: str, range_info=None, question=""):
        """Call Gemini Pro via OpenRouter"""
        import openai
        
        try:
            scoring_prompt = self.create_scoring_prompt(essay_text, question, range_info)
            if scoring_prompt is None:
                return None
            
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            response = client.chat.completions.create(
                model="google/gemini-pro-1.5",
                messages=[{"role": "user", "content": scoring_prompt}],
                max_tokens=15,
                temperature=0.3
            )
            
            return self.parse_model_response(response.choices[0].message.content.strip(), range_info)
            
        except Exception as e:
            print(f"Gemini Pro API error: {e}")
            return None

    def call_gemini_flash(self, essay_text: str, prompt: str, range_info=None, question=""):
        """Call Gemini Flash via OpenRouter"""
        import openai
        
        try:
            scoring_prompt = self.create_scoring_prompt(essay_text, prompt, range_info)
            if scoring_prompt is None:
                return None
            
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
        
            
            response = client.chat.completions.create(
                model="google/gemini-flash-1.5",
                messages=[{"role": "user", "content": scoring_prompt}],
                max_tokens=15,
                temperature=0.3
            )
            
            return self.parse_model_response(response.choices[0].message.content.strip(), range_info)
            
        except Exception as e:
            print(f"Gemini Flash API error: {e}")
            return None
    
    def call_gpt4o(self, essay_text: str, prompt: str, range_info=None, question=""):
        """Call GPT-4o via OpenRouter"""
        import openai
        
        try:
            scoring_prompt = self.create_scoring_prompt(essay_text, question, range_info)
            if scoring_prompt is None:
                return None
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": scoring_prompt}],
                max_tokens=15,
                temperature=0.3
            )
            
            return self.parse_model_response(response.choices[0].message.content.strip(), range_info)
            
        except Exception as e:
            print(f"GPT-4o API error: {e}")
            return None

    def call_gpt4o_mini(self, essay_text: str, prompt: str, range_info=None, question=""):
        """Call GPT-4o-mini via OpenRouter"""
        import openai
        
        try:
            scoring_prompt = self.create_scoring_prompt(essay_text, question, range_info)
            if scoring_prompt is None:
                return None
            
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": scoring_prompt}],
                max_tokens=15,
                temperature=0.3
            )
            
            return self.parse_model_response(response.choices[0].message.content.strip(), range_info)
            
        except Exception as e:
            print(f"GPT-4o-mini API error: {e}")
            return None

    def call_claude_sonnet(self, essay_text: str, prompt: str, range_info=None, question=""):
        """Call Claude Sonnet via OpenRouter"""
        import openai
        
        try:
            scoring_prompt = self.create_scoring_prompt(essay_text, question, range_info)
            if scoring_prompt is None:
                return None
            
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            response = client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": scoring_prompt}],
                max_tokens=15,
                temperature=0.3
            )
            
            return self.parse_model_response(response.choices[0].message.content.strip(), range_info)
            
        except Exception as e:
            print(f"Claude Sonnet API error: {e}")
            return None

    def call_claude_haiku(self, essay_text: str, prompt: str, range_info=None, question=""):
        """Call Claude Haiku via OpenRouter"""
        import openai
        
        try:
            scoring_prompt = self.create_scoring_prompt(essay_text, question, range_info)
            if scoring_prompt is None:
                return None
            
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            response = client.chat.completions.create(
                model="anthropic/claude-3.5-haiku",
                messages=[{"role": "user", "content": scoring_prompt}],
                max_tokens=15,
                temperature=0.3
            )
            
            return self.parse_model_response(response.choices[0].message.content.strip(), range_info)
            
        except Exception as e:
            print(f"Claude Haiku API error: {e}")
            return None


    def calculate_additional_metrics(self, predictions, actuals, task_type="regression"):
        """Calculate QWK, F1, Precision, Recall"""
        import numpy as np
        from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
        
        metrics = {}
        
        try:
            # Quadratic Weighted Kappa (QWK)
            def quadratic_weighted_kappa(y_true, y_pred):
                """Calculate QWK for continuous scores"""
                # Convert to integer bins for kappa calculation
                y_true_int = np.round(y_true).astype(int)
                y_pred_int = np.round(y_pred).astype(int)
                
                return cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')
            
            metrics['qwk'] = quadratic_weighted_kappa(actuals, predictions)
            
            # For classification tasks (like BEEtlE, SciEntSBank)
            if task_type == "classification":
                # Convert categorical to numeric if needed
                if isinstance(predictions[0], str):
                    label_map = {"incorrect": 0, "partial_correct": 1, "correct": 2}
                    pred_numeric = [label_map.get(p, 0) for p in predictions]
                    actual_numeric = [label_map.get(a, 0) for a in actuals]
                else:
                    pred_numeric = predictions
                    actual_numeric = actuals
                
                metrics['f1_macro'] = f1_score(actual_numeric, pred_numeric, average='macro')
                metrics['f1_micro'] = f1_score(actual_numeric, pred_numeric, average='micro')
                metrics['precision_macro'] = precision_score(actual_numeric, pred_numeric, average='macro')
                metrics['recall_macro'] = recall_score(actual_numeric, pred_numeric, average='macro')
            
            # For regression tasks - create binary classification at threshold
            else:
                # Binary classification: good (>= median) vs poor (< median)
                median_score = np.median(actuals)
                actual_binary = (np.array(actuals) >= median_score).astype(int)
                pred_binary = (np.array(predictions) >= median_score).astype(int)
                
                metrics['f1_binary'] = f1_score(actual_binary, pred_binary)
                metrics['precision_binary'] = precision_score(actual_binary, pred_binary)
                metrics['recall_binary'] = recall_score(actual_binary, pred_binary)
        
        except Exception as e:
            print(f"Error calculating additional metrics: {e}")
            metrics = {
                'qwk': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0,
                'precision_macro': 0.0, 'recall_macro': 0.0
            }
        
        return metrics

    def create_scoring_prompt(self, essay_text: str, question: str, range_info=None):
        """Create standardized scoring prompt"""
        # Clean and validate essay text
        essay_text = str(essay_text).strip()
        
        # Skip completely empty essays
        if len(essay_text) < 5:
            return None  # Return None for empty essays
        
        # Truncate very long essays to avoid token limits
        if len(essay_text) > 2000:
            essay_text = essay_text[:2000] + "..."
        
        if range_info and range_info["type"] != "categorical":
            min_score, max_score = range_info["min"], range_info["max"]
            description = range_info["description"]
            
            return f"""You are an expert essay grader. Rate this essay using the {description}.

    Essay: {essay_text}

    Scoring Guidelines for {min_score}-{max_score} scale:
    - {max_score}: Exceptional essay with sophisticated ideas, excellent organization, and strong writing
    - {max_score*0.8:.0f}: Good essay with clear ideas and solid writing
    - {max_score*0.6:.0f}: Adequate essay meeting basic requirements
    - {max_score*0.4:.0f}: Below average essay with some issues
    - {max_score*0.2:.0f}: Poor essay with significant problems
    - {min_score}: Very poor or off-topic essay

    Consider: Content quality, organization, language use, development of ideas, and adherence to prompt.

    Provide only a numerical score between {min_score} and {max_score}:"""

        else:
            # Handle categorical or default case
            return f"""Rate this essay on a 0-5 scale where 5=Excellent, 0=Very Poor.

    Essay: {essay_text}

    Score:"""
 
    def parse_model_response(self, response_text: str, range_info):
        """Parse model response based on range type"""
        import re
        
        if range_info and range_info["type"] == "categorical":
            # Handle categorical responses
            response_lower = response_text.lower().strip()
            for category in range_info["categories"]:
                if category.lower() in response_lower:
                    return category
            return range_info["categories"][0]  # Default to first category
        
        else:
            # Handle numeric responses
            numbers = re.findall(r'\d+\.?\d*', response_text)
            if numbers:
                score = float(numbers[0])
                if range_info:
                    return max(range_info["min"], min(range_info["max"], score))
                else:
                    return max(0, min(5, score))
            
            # Default scores
            if range_info:
                return (range_info["min"] + range_info["max"]) / 2
            return 2.5
    
    def simulate_model(self, essay_text: str, range_info=None):
        """Simulate model scoring with dataset-specific range"""
        word_count = len(essay_text.split())
        
        # Handle categorical responses
        if range_info and range_info["type"] == "categorical":
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
    
    def evaluate_with_ground_truth(self, dataset_name: str, predictions: List[Dict], model_type: str = "Unknown"):
        """Submit predictions to BESESR for evaluation against ground truth from {dataset_name}"""
        try:
            # Complete Dataset submission requirements with D_ versions
            SUBMISSION_REQUIREMENTS = {
                "ASAP-AES": ["essay_id", "domain1_score"],
                "D_ASAP-AES": ["essay_id", "domain1_score"],
                "ASAP2": ["essay_id", "score"],
                "D_ASAP2": ["essay_id", "score"], 
                "ASAP-SAS": ["Id", "Score1"],
                "D_ASAP-SAS": ["Id", "Score1"],
                "ASAP_plus_plus": ["essay_id", "overall_score"],
                "D_ASAP_plus_plus": ["essay_id", "overall_score"],
                "CSEE": ["index", "overall_score"],
                "D_CSEE": ["index", "overall_score"],
                "persuade_2": ["essay_id_comp", "holistic_essay_score"],
                "D_persuade_2": ["essay_id_comp", "holistic_essay_score"],
                "Rice_Chem_Q1": ["sis_id", "Score"],
                "D_Rice_Chem_Q1": ["sis_id", "Score"],
                "Rice_Chem_Q2": ["sis_id", "Score"],
                "D_Rice_Chem_Q2": ["sis_id", "Score"],
                "Rice_Chem_Q3": ["sis_id", "Score"],
                "D_Rice_Chem_Q3": ["sis_id", "Score"],
                "Rice_Chem_Q4": ["sis_id", "Score"],
                "D_Rice_Chem_Q4": ["sis_id", "Score"],
                "BEEtlE_2way": ["ID", "label"],
                "D_BEEtlE_2way": ["ID", "label"],
                "BEEtlE_3way": ["ID", "label"],
                "D_BEEtlE_3way": ["ID", "label"],
                "SciEntSBank_2way": ["ID", "label"],
                "D_SciEntSBank_2way": ["ID", "label"],
                "SciEntSBank_3way": ["ID", "label"],
                "D_SciEntSBank_3way": ["ID", "label"],
                "EFL": ["ID", "_Human_Mean"],
                "D_EFL": ["ID", "_Human_Mean"],
                "Mohlar": ["ID", "grade"],
                "D_Mohlar": ["ID", "grade"],
                "Ielts_Writing_Dataset": ["ID", "Overall_Score"],
                "D_Ielts_Writing_Dataset": ["ID", "Overall_Score"],
                "Ielst_Writing_Task_2_Dataset": ["ID", "band_score"],
                "D_Ielst_Writing_Task_2_Dataset": ["ID", "band_score"],
                "Regrading_Dataset_J2C": ["ID", "grade"],
                "D_Regrading_Dataset_J2C": ["ID", "grade"],
                "grade_like_a_human_dataset_os_q1": ["ID", "score_1"],
                "D_grade_like_a_human_dataset_os_q1": ["ID", "score_1"],
                "grade_like_a_human_dataset_os_q2": ["ID", "score_1"],
                "D_grade_like_a_human_dataset_os_q2": ["ID", "score_1"],
                "grade_like_a_human_dataset_os_q3": ["ID", "score_1"],
                "D_grade_like_a_human_dataset_os_q3": ["ID", "score_1"],
                "grade_like_a_human_dataset_os_q4": ["ID", "score_1"],
                "D_grade_like_a_human_dataset_os_q4": ["ID", "score_1"],
                "grade_like_a_human_dataset_os_q5": ["ID", "score_1"],
                "D_grade_like_a_human_dataset_os_q5": ["ID", "score_1"],
            }
            
            # Get the required format for this dataset
            if dataset_name not in SUBMISSION_REQUIREMENTS:
                print(f"Warning: {dataset_name} not in submission requirements, using generic format")
                id_col, score_col = "essay_id", "predicted_score"
            else:
                id_col, score_col = SUBMISSION_REQUIREMENTS[dataset_name]
            
            # Create CSV content in memory
            csv_content = io.StringIO()
            writer = csv.writer(csv_content)
            
            # Write header
            writer.writerow([id_col, score_col])
            
            # Write predictions - now predictions already contain properly scaled scores
            for pred in predictions:
                pred_id = pred['id']
                final_score = pred['predicted_score']  # Already in correct range
                
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
            
            files = {
                'file': ('predictions.csv', csv_data, 'text/csv')
            }
            data = {
                'dataset_name': dataset_name,
                'methodology': 'zero-shot',
                'model_name': f'{model_type.upper()} Zero-shot',  # Add this line
                'submitter_name': f'{model_type.upper()} Zero-shot Model'  
            }
            
            print(f"Submitting predictions to {dataset_name} for ground truth evaluation...")
            print(f"Format: {id_col}, {score_col} ({len(predictions)} predictions)")
            
            # Submit to BESESR test endpoint (uses ground truth from {dataset_name})
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
    
    def get_ground_truth_scores(self, dataset_name: str, essay_ids: List[str]) -> Dict[str, float]:
        """Fetch ground truth scores for given essay IDs"""
        try:
            gt_dataset_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
            dev_key = os.getenv("DEV_GROUND_TRUTH_KEY")
            url = f"{self.base_url}/api/download-ground-truth/{gt_dataset_name}"
            print(f"DEBUG: Fetching from URL: {url}")
            print(f"DEBUG: Using dev key: {dev_key}")
            
            response = requests.get(url, params={"dev_key": dev_key})
            
            print(f"DEBUG: Response status: {response.status_code}")
            print(f"DEBUG: Response text: {response.text[:200]}")
            print(f"Fetching ground truth from: {gt_dataset_name}")
            
            response = requests.get(
                f"{self.base_url}/api/datasets/download-ground-truth/{gt_dataset_name}",
                params={"dev_key": "your_dev_key_here"}
            )
            
            if response.status_code != 200:
                print(f"Failed to download ground truth dataset: HTTP {response.status_code}")
                return {}
            
            # Download ground truth dataset
            response = requests.get(f"{self.base_url}/api/datasets/download/{gt_dataset_name}")
            if response.status_code != 200:
                print(f"Failed to download ground truth dataset: HTTP {response.status_code}")
                return {}
            
            # Extract and load ground truth CSV
            zip_content = zipfile.ZipFile(io.BytesIO(response.content))
            csv_file = None
            for file_info in zip_content.filelist:
                if file_info.filename.endswith('.csv'):
                    if 'test' in file_info.filename.lower():
                        csv_file = file_info.filename
                        break
            
            if not csv_file:
                for file_info in zip_content.filelist:
                    if file_info.filename.endswith('.csv'):
                        csv_file = file_info.filename
                        break
            
            if not csv_file:
                print("No CSV file found in ground truth dataset")
                return {}
            
            print(f"Loading ground truth from: {csv_file}")
            with zip_content.open(csv_file) as f:
                gt_df = pd.read_csv(f)
            
            print(f"Ground truth dataset columns: {list(gt_df.columns)}")
            
            # For Regrading dataset
            id_col = "ID"
            score_col = "grade"
            
            if id_col not in gt_df.columns or score_col not in gt_df.columns:
                print(f"Required columns not found! Available: {list(gt_df.columns)}")
                return {}
            
            # Create mapping of ID to ground truth score
            gt_scores = {}
            for _, row in gt_df.iterrows():
                essay_id = str(row[id_col])
                if essay_id in essay_ids:
                    gt_scores[essay_id] = row[score_col]
            
            print(f"Successfully mapped {len(gt_scores)} ground truth scores")
            return gt_scores
            
        except Exception as e:
            print(f"Warning: Could not fetch ground truth scores: {e}")
            return {}
        
    def run_single_test(self, dataset_name: str, model_type: str, num_essays: int = 10):
        if dataset_name.startswith("D_"):
            ground_truth_dataset = dataset_name[2:]  # Remove D_ prefix
        else:
            ground_truth_dataset = dataset_name
            
        print("=" * 60)
        print(f"SINGLE MODEL TEST")
        print(f"Model: {model_type.upper()}")
        print(f"Dataset: {dataset_name}")
        print(f"Test Data: {dataset_name} (unlabeled)")
        print(f"Ground Truth: {ground_truth_dataset} (labeled)")
        print(f"Essays: {num_essays}")
        print("=" * 60)
        
        # Test BESESR connection
        if not self.test_connection():
            print("BESESR not accessible at localhost:8000")
            return None
        print("✓ BESESR connected")
        
        # Step 1: Download unlabeled test data from D_{dataset_name}
        df = self.download_test_data(dataset_name, num_essays)
        if df is None:
            print(f"Failed to download test data from {dataset_name}")
            return None
        
        # Step 2: Prepare essays for prediction with range info
        essays = self.prepare_essays_for_prediction(df, dataset_name)
        if not essays:
            print("No essays prepared for prediction")
            return None

        # Step 2.5: Fetch ground truth scores for display
        essay_ids = [str(essay['id']) for essay in essays]
        print(f"Attempting to fetch ground truth scores for {len(essay_ids)} essays...")
        ground_truth_scores = self.get_ground_truth_scores(dataset_name, essay_ids)
        
        if ground_truth_scores:
            print(f"✓ Fetched ground truth for {len(ground_truth_scores)} essays")
        else:
            print("⚠ Ground truth scores not available for real-time display")
            print("  (Will be calculated during evaluation)")
            ground_truth_scores = {}

        print(f"\nGenerating predictions with {model_type} on {len(essays)} test essays...")
        
        predictions = []
        results_detail = []
        
        # Step 3: Generate predictions for each test essay
        for i, essay in enumerate(essays, 1):
            print(f"Essay {i}/{len(essays)}: ", end="", flush=True)
            
            start_time = time.time()
            predicted_score = self.call_api_model(
                essay['text'], 
                essay['prompt'], 
                model_type,
                range_info=essay.get('range_info'),
                question=essay.get('question', '')
            )
            end_time = time.time()
            
            if predicted_score is not None:
                predictions.append(predicted_score)
                
                # Show the appropriate range in output
                range_info = essay.get('range_info', {})
                if range_info.get("type") == "categorical":
                    range_display = f" ({range_info['description']})"
                else:
                    max_val = range_info.get('max', 5)
                    range_display = f"/{max_val}"
                
                # Get actual ground truth score for this essay
                actual_score = ground_truth_scores.get(str(essay['id']), "Unknown")
                if actual_score != "Unknown":
                    actual_display = f", Actual: {actual_score}{range_display if range_info.get('type') != 'categorical' else ''}"
                else:
                    actual_display = ", Actual: Unknown"
                
                print(f"Predicted: {predicted_score}{range_display}{actual_display}, Time: {end_time-start_time:.1f}s")
                
                results_detail.append({
                    'essay_id': essay['id'],
                    'predicted': predicted_score,
                    'actual': actual_score,  # Store actual ground truth score
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
        
        # Step 4: Prepare predictions for ground truth evaluation
        besesr_predictions = []
        for detail in results_detail:
            besesr_predictions.append({
                'id': detail['essay_id'],
                'predicted_score': detail['predicted']
            })
        
        # Step 5: Evaluate against ground truth from {dataset_name}
        metrics = self.evaluate_with_ground_truth(dataset_name, besesr_predictions, model_type)
        
        if metrics:
            print("✓ Ground truth evaluation completed successfully")
            
            # Handle string vs numeric predictions - FIXED VERSION
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
            
            for detail in results_detail:
                if detail.get('actual') == "Unknown":
                    if metrics.get('mean_absolute_error', 0) > 0:
                        predicted = detail['predicted']
                        mae = metrics['mean_absolute_error']
                        if isinstance(predicted, str):
                            estimated_actual = predicted 
                        else:
                            estimated_actual = round(predicted + mae, 1)
                        detail['actual'] = f"~{estimated_actual} (est)"
                    else:
                        detail['actual'] = 'Not available'
        else:
            print("✗ Ground truth evaluation failed")
            print("This could mean:")
            print("  - Ground truth dataset not available")
            print("  - Prediction format incorrect")
            print("  - ID mismatch between test and ground truth")
            
            # Provide basic statistics without ground truth - FIXED VERSION
            if predictions and isinstance(predictions[0], str):
                # Handle categorical predictions
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
            'model': model_type,
            'dataset': dataset_name,
            'test_data_source': dataset_name,
            'ground_truth_source': ground_truth_dataset,
            'metrics': metrics,
            'details': results_detail,
            'success': metrics.get('mean_absolute_error', 0) > 0  # True if we got real evaluation
        }
    
    def display_results(self, model_type: str, dataset_name: str, metrics: dict, details: list):
        """Display formatted results with proper D_ prefix handling"""
        
        # Determine correct display names
        if dataset_name.startswith("D_"):
            test_data_name = dataset_name  # Already has D_ prefix
            ground_truth_name = dataset_name[2:]  # Remove D_ prefix
        else:
            test_data_name = f"D_{dataset_name}"  # Add D_ prefix
            ground_truth_name = dataset_name  # Keep as is
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        print(f"Model: {model_type.upper()}")
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
        
        # Handle string vs numeric values for display
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
            
            # Show range information
            range_info = detail.get('range_info', {})
            if range_info.get("type") == "categorical":
                range_display = f" ({range_info['description']})"
            else:
                max_val = range_info.get('max', 5)
                range_display = f"/{max_val}"
            
            print(f"{i}. ID: {detail['essay_id']}")
            print(f"   Predicted: {predicted}{range_display} | Actual: {actual}{range_display if actual != 'Unknown' else ''} | Time: {detail['time']:.1f}s")
            print(f"   Text: {detail['text_preview']}")
            print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 1 model on 1 dataset with dynamic range support")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., D_Regrading_Dataset_J2C)")  # Add this line
    parser.add_argument("--model", required=True, 
                   choices=[
                       "gpt4o", "gpt4o-mini", "claude-sonnet", "claude-haiku",
                       "gemini-pro", "gemini-flash", "qwen-max", "step-1v",
                       "simulation"
                   ], 
                   help="Model to test")
    
    parser.add_argument("--essays", type=int, default=None, help="Number of essays to test (default: all available)")
    parser.add_argument("--list-datasets", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    tester = SingleModelTester()
    
    if args.list_datasets:
        print("Available datasets:")
        datasets = tester.get_available_datasets()
        for i, ds in enumerate(datasets, 1):
            print(f"{i:2d}. {ds}")
        return
    
    result = tester.run_single_test(args.dataset, args.model, args.essays) 
    
    if result and result['success']:
        print(f"\nTest completed successfully!")
        print(f"Test Data: {result['test_data_source']} (unlabeled)")
        print(f"Ground Truth: {result['ground_truth_source']} (labeled)")
        print(f"Correlation: {result['metrics'].get('pearson_correlation', 0):.3f}")
        print(f"MAE: {result['metrics'].get('mean_absolute_error', 0):.3f}")
    else:
        print("Test completed but ground truth evaluation failed!")

if __name__ == "__main__":
    main()