import asyncio
import pandas as pd
import numpy as np
import requests
import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import csv
from pathlib import Path
import time
import torch
from mllm_models import MLLMModelFactory, BESESRModelTester, BaseMLLMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('besesr_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BESESRIntegration:
    """Integration between BESESR platform and MLLM models"""
    
    def __init__(self, besesr_base_url: str = "http://localhost:8000"):
        self.base_url = besesr_base_url
        self.session = requests.Session()
        self.tester = BESESRModelTester()
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def fetch_datasets_from_besesr(self) -> Dict[str, Any]:
        """Fetch available datasets from BESESR platform"""
        try:
            response = self.session.get(f"{self.base_url}/api/datasets/")
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched {data.get('total_count', 0)} datasets from BESESR")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch datasets: {e}")
            return {"datasets": [], "total_count": 0}
    
    def download_dataset(self, dataset_name: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Download a specific dataset from BESESR"""
        try:
            response = self.session.get(f"{self.base_url}/api/datasets/download/{dataset_name}")
            response.raise_for_status()
            
            # Save ZIP file
            zip_path = self.results_dir / f"{dataset_name}.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract and load CSV files
            import zipfile
            splits = {}
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.results_dir / dataset_name)
                
            # Load train, validation, test splits
            dataset_dir = self.results_dir / dataset_name
            for split in ['train', 'validation', 'test']:
                csv_path = dataset_dir / f"{split}.csv"
                if csv_path.exists():
                    splits[split] = pd.read_csv(csv_path)
                    logger.info(f"Loaded {dataset_name} {split}: {len(splits[split])} rows")
            
            return splits
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {e}")
            return None
    
    def prepare_essays_for_evaluation(self, 
                                    dataset_splits: Dict[str, pd.DataFrame],
                                    dataset_name: str,
                                    max_essays: int = 50) -> List[Dict[str, Any]]:
        """Prepare essays from dataset for model evaluation"""
        essays = []
        
        # Use test split if available, otherwise train
        if 'test' in dataset_splits:
            df = dataset_splits['test']
        elif 'train' in dataset_splits:
            df = dataset_splits['train']
        else:
            logger.warning(f"No suitable split found for {dataset_name}")
            return []
        
        # Sample essays to avoid overwhelming evaluation
        if len(df) > max_essays:
            df = df.sample(n=max_essays, random_state=42)
        
        # Dataset-specific column mapping based on your BESESR datasets
        column_mappings = {
            'ASAP-AES': {'essay': 'essay', 'prompt': 'prompt', 'score': 'domain1_score'},
            'ASAP2': {'essay': 'essay', 'prompt': 'prompt', 'score': 'score'},
            'BEEtlE_2way': {'essay': 'student_answer', 'prompt': 'question_text', 'score': 'label'},
            'BEEtlE_3way': {'essay': 'student_answer', 'prompt': 'question_text', 'score': 'label'},
            'CSEE': {'essay': 'essay', 'prompt': 'prompt', 'score': 'overall_score'},
            'EFL': {'essay': 'essay', 'prompt': 'prompt', 'score': 'score'},
            'SciEntSBank_2way': {'essay': 'student_answer', 'prompt': 'question_text', 'score': 'label'},
            'SciEntSBank_3way': {'essay': 'student_answer', 'prompt': 'question_text', 'score': 'label'},
            'Mohlar': {'essay': 'essay', 'prompt': 'prompt', 'score': 'score'},
            'persuade_2': {'essay': 'essay', 'prompt': 'prompt', 'score': 'holistic_score'},
            'grade_like_a_human_dataset_os_q1': {'essay': 'essay', 'prompt': 'question', 'score': 'score_1'},
            'grade_like_a_human_dataset_os_q2': {'essay': 'essay', 'prompt': 'question', 'score': 'score_1'},
            'grade_like_a_human_dataset_os_q3': {'essay': 'essay', 'prompt': 'question', 'score': 'score_1'},
            'grade_like_a_human_dataset_os_q4': {'essay': 'essay', 'prompt': 'question', 'score': 'score_1'},
            'grade_like_a_human_dataset_os_q5': {'essay': 'essay', 'prompt': 'question', 'score': 'score_1'},
            'Rice_Chem_Q1': {'essay': 'response', 'prompt': 'question', 'score': 'Score'},
            'Rice_Chem_Q2': {'essay': 'response', 'prompt': 'question', 'score': 'Score'},
            'Rice_Chem_Q3': {'essay': 'response', 'prompt': 'question', 'score': 'Score'},
            'Rice_Chem_Q4': {'essay': 'response', 'prompt': 'question', 'score': 'Score'}
        }
        
        mapping = column_mappings.get(dataset_name, {
            'essay': 'essay', 'prompt': 'prompt', 'score': 'score'
        })
        
        for idx, row in df.iterrows():
            # Extract text content safely
            essay_text = str(row.get(mapping['essay'], '')).strip()
            prompt_text = str(row.get(mapping['prompt'], '')).strip()
            
            # Skip if essential content is missing
            if not essay_text or len(essay_text) < 10:
                continue
                
            essay_data = {
                'essay_id': f"{dataset_name}_{idx}",
                'essay_text': essay_text,
                'prompt': prompt_text,
                'dataset': dataset_name,
                'human_score': row.get(mapping['score'], None),
                'images': [],  # No images in current BESESR datasets
                'metadata': {
                    'row_index': idx,
                    'dataset_split': 'test' if 'test' in dataset_splits else 'train'
                }
            }
            
            essays.append(essay_data)
        
        logger.info(f"Prepared {len(essays)} essays from {dataset_name}")
        return essays
    
    def load_models_for_evaluation(self, model_configs: Dict[str, Dict]) -> Dict[str, BaseMLLMModel]:
        """Load specified models for evaluation"""
        factory = MLLMModelFactory()
        loaded_models = {}
        
        for model_name, config in model_configs.items():
            try:
                logger.info(f"Loading model: {model_name}")
                
                if config.get('type') == 'api':
                    # API-based models need keys
                    api_key = config.get('api_key') or os.getenv(config.get('api_key_env'))
                    if not api_key:
                        logger.warning(f"No API key found for {model_name}, skipping")
                        continue
                    model = factory.create_model(config['model_id'], api_key=api_key)
                else:
                    # Local models
                    model = factory.create_model(config['model_id'])
                
                model.load_model()
                loaded_models[model_name] = model
                self.tester.add_model(model)
                
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                continue
                
        return loaded_models
    
    def evaluate_models_on_dataset(self, 
                                 dataset_name: str,
                                 essays: List[Dict],
                                 models: Dict[str, BaseMLLMModel],
                                 traits: List[str] = None) -> Dict[str, Dict]:
        """Evaluate all loaded models on a dataset"""
        
        if traits is None:
            # Use EssayJudge paper traits
            traits = [
                "lexical_accuracy", "lexical_diversity", 
                "grammatical_accuracy", "grammatical_diversity", 
                "punctuation_accuracy", "coherence",
                "argument_clarity", "justifying_persuasiveness", 
                "organizational_structure", "essay_length"
            ]
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name} on {dataset_name}")
            
            try:
                model_results = self.tester.evaluate_model_on_dataset(
                    model_name, dataset_name, essays, traits
                )
                results[model_name] = model_results
                
                # Save intermediate results
                self.save_results({model_name: {dataset_name: model_results}}, 
                                f"intermediate_{model_name}_{dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {e}")
                results[model_name] = {trait: 0.0 for trait in traits}
        
        return results
    
    def generate_csv_predictions(self, 
                               dataset_name: str,
                               essays: List[Dict],
                               model: BaseMLLMModel,
                               target_trait: str = "overall_score") -> str:
        """Generate CSV predictions in BESESR format for submission"""
        
        predictions = []
        
        for essay in essays:
            try:
                # Generate prediction using the model
                score = model.score_essay(
                    essay['essay_text'], 
                    essay['prompt'], 
                    target_trait,
                    essay.get('images', [])
                )
                
                # Scale score appropriately based on dataset
                scaled_score = self.scale_score_for_dataset(score, dataset_name)
                
                predictions.append({
                    'essay_id': essay['essay_id'],
                    'predicted_score': scaled_score
                })
                
            except Exception as e:
                logger.error(f"Failed to predict for {essay['essay_id']}: {e}")
                predictions.append({
                    'essay_id': essay['essay_id'],
                    'predicted_score': 2.5  # Default middle score
                })
        
        # Save to CSV
        csv_filename = f"{model.model_name}_{dataset_name}_predictions.csv"
        csv_path = self.results_dir / csv_filename
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['essay_id', 'predicted_score'])
            writer.writeheader()
            writer.writerows(predictions)
        
        logger.info(f"Generated predictions CSV: {csv_path}")
        return str(csv_path)
    
    def scale_score_for_dataset(self, score: float, dataset_name: str) -> float:
        """Scale 0-5 model score to dataset-specific range"""
        
        # Dataset score ranges based on BESESR configuration
        score_ranges = {
            'ASAP-AES': (0, 60),
            'ASAP2': (0, 60), 
            'BEEtlE_2way': (0, 1),
            'BEEtlE_3way': (0, 2),
            'CSEE': (0, 100),
            'EFL': (0, 100),
            'SciEntSBank_2way': (0, 1),
            'SciEntSBank_3way': (0, 2),
            'persuade_2': (0, 100),
            'grade_like_a_human_dataset_os_q1': (0, 100),
            'grade_like_a_human_dataset_os_q2': (0, 100),
            'grade_like_a_human_dataset_os_q3': (0, 100),
            'grade_like_a_human_dataset_os_q4': (0, 100),
            'grade_like_a_human_dataset_os_q5': (0, 100),
            'Rice_Chem_Q1': (0, 100),
            'Rice_Chem_Q2': (0, 100),
            'Rice_Chem_Q3': (0, 100),
            'Rice_Chem_Q4': (0, 100)
        }
        
        min_score, max_score = score_ranges.get(dataset_name, (0, 100))
        
        # Scale from 0-5 to dataset range
        scaled = min_score + (score / 5.0) * (max_score - min_score)
        return round(scaled, 2)
    
    def submit_predictions_to_besesr(self, csv_path: str, dataset_name: str, model_name: str):
        """Submit predictions to BESESR platform"""
        
        try:
            with open(csv_path, 'rb') as f:
                files = {'file': (os.path.basename(csv_path), f, 'text/csv')}
                data = {
                    'dataset_name': dataset_name,
                    'model_name': model_name,
                    'model_description': f"MLLM evaluation of {model_name}",
                    'researcher_name': "MLLM Evaluation Framework"
                }
                
                response = self.session.post(
                    f"{self.base_url}/api/submissions/upload-single",
                    files=files,
                    data=data
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Successfully submitted {model_name} predictions for {dataset_name}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to submit predictions: {e}")
            return None
    
    def save_results(self, results: Dict, filename_suffix: str = ""):
        """Save evaluation results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}_{filename_suffix}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved results to {filepath}")
        return str(filepath)
    
    def generate_comparison_report(self, results: Dict[str, Dict]) -> str:
        """Generate a comprehensive comparison report"""
        
        report_lines = [
            "# BESESR MLLM Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Performance Summary"
        ]
        
        # Calculate overall rankings
        leaderboard = self.tester.generate_leaderboard(results)
        
        report_lines.append("\n### Overall Leaderboard (Average Score)")
        for rank, (model, score) in enumerate(leaderboard.items(), 1):
            report_lines.append(f"{rank:2d}. {model:<25} {score:.3f}")
        
        # Dataset-specific performance
        report_lines.append("\n## Dataset-Specific Performance")
        
        all_datasets = set()
        for model_results in results.values():
            all_datasets.update(model_results.keys())
        
        for dataset in sorted(all_datasets):
            report_lines.append(f"\n### {dataset}")
            dataset_scores = []
            
            for model_name, model_results in results.items():
                if dataset in model_results:
                    avg_score = np.mean(list(model_results[dataset].values()))
                    dataset_scores.append((model_name, avg_score))
            
            # Sort by performance
            dataset_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (model, score) in enumerate(dataset_scores, 1):
                report_lines.append(f"{rank:2d}. {model:<25} {score:.3f}")
        
        # Trait-specific analysis
        report_lines.append("\n## Trait-Specific Analysis")
        
        all_traits = set()
        for model_results in results.values():
            for dataset_results in model_results.values():
                all_traits.update(dataset_results.keys())
        
        for trait in sorted(all_traits):
            report_lines.append(f"\n### {trait}")
            trait_scores = []
            
            for model_name, model_results in results.items():
                trait_values = []
                for dataset_results in model_results.values():
                    if trait in dataset_results:
                        trait_values.append(dataset_results[trait])
                
                if trait_values:
                    avg_trait_score = np.mean(trait_values)
                    trait_scores.append((model_name, avg_trait_score))
            
            trait_scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (model, score) in enumerate(trait_scores[:5], 1):  # Top 5
                report_lines.append(f"{rank:2d}. {model:<25} {score:.3f}")
        
        # Save report
        report_content = "\n".join(report_lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"evaluation_report_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated report: {report_path}")
        return str(report_path)
    
    async def run_full_evaluation_pipeline(self, 
                                         model_configs: Dict[str, Dict],
                                         dataset_names: List[str] = None,
                                         max_essays_per_dataset: int = 25,
                                         submit_to_platform: bool = True):
        """Run the complete evaluation pipeline"""
        
        logger.info("Starting BESESR-MLLM evaluation pipeline")
        
        # Step 1: Load models
        logger.info("Step 1: Loading models...")
        models = self.load_models_for_evaluation(model_configs)
        
        if not models:
            logger.error("No models loaded successfully. Exiting.")
            return
        
        # Step 2: Fetch available datasets
        logger.info("Step 2: Fetching datasets from BESESR...")
        datasets_info = self.fetch_datasets_from_besesr()
        
        if dataset_names is None:
            # Use all available datasets
            dataset_names = [ds['name'] for ds in datasets_info.get('datasets', [])]
        
        logger.info(f"Will evaluate on {len(dataset_names)} datasets")
        
        # Step 3: Evaluate on each dataset
        all_results = {}
        
        for dataset_name in dataset_names:
            logger.info(f"Step 3: Processing dataset {dataset_name}")
            
            # Download dataset
            splits = self.download_dataset(dataset_name)
            if not splits:
                logger.warning(f"Skipping {dataset_name} - download failed")
                continue
            
            # Prepare essays
            essays = self.prepare_essays_for_evaluation(
                splits, dataset_name, max_essays_per_dataset
            )
            
            if not essays:
                logger.warning(f"Skipping {dataset_name} - no essays prepared")
                continue
            
            # Evaluate models
            dataset_results = self.evaluate_models_on_dataset(
                dataset_name, essays, models
            )
            
            # Update overall results
            for model_name, model_results in dataset_results.items():
                if model_name not in all_results:
                    all_results[model_name] = {}
                all_results[model_name][dataset_name] = model_results
            
            # Generate and submit predictions if requested
            if submit_to_platform:
                for model_name, model in models.items():
                    try:
                        csv_path = self.generate_csv_predictions(
                            dataset_name, essays, model
                        )
                        
                        self.submit_predictions_to_besesr(
                            csv_path, dataset_name, model_name
                        )
                        
                    except Exception as e:
                        logger.error(f"Failed to submit {model_name} to platform: {e}")
        
        # Step 4: Generate final reports
        logger.info("Step 4: Generating final reports...")
        
        self.save_results(all_results, "final_results")
        report_path = self.generate_comparison_report(all_results)
        
        logger.info("Evaluation pipeline completed!")
        logger.info(f"Results saved in: {self.results_dir}")
        logger.info(f"Report generated: {report_path}")
        
        return all_results

# =============================================================================
# CONFIGURATION AND MAIN EXECUTION
# =============================================================================

def get_model_configurations():
    """Define model configurations for evaluation"""
    
    return {
        # Open-source models
        "Yi-VL-6B": {
            "model_id": "yi-vl",
            "type": "local"
        },
        "Qwen2-VL-7B": {
            "model_id": "qwen2-vl", 
            "type": "local"
        },
        "DeepSeek-VL-7B": {
            "model_id": "deepseek-vl",
            "type": "local"
        },
        "LLaVA-NEXT-8B": {
            "model_id": "llava-next",
            "type": "local"
        },
        "InternVL2-8B": {
            "model_id": "internvl2",
            "type": "local"
        },
        
        # Closed-source models (require API keys)
        "GPT-4o": {
            "model_id": "gpt-4o",
            "type": "api",
            "api_key_env": "OPENAI_API_KEY"
        },
        "Claude-3.5-Sonnet": {
            "model_id": "claude-3.5-sonnet",
            "type": "api", 
            "api_key_env": "ANTHROPIC_API_KEY"
        },
        "Gemini-1.5-Pro": {
            "model_id": "gemini-1.5-pro",
            "type": "api",
            "api_key_env": "GOOGLE_API_KEY"
        }
    }

async def main():
    """Main execution function"""
    
    # Initialize integration
    integration = BESESRIntegration("http://localhost:8000")
    
    # Get model configurations
    model_configs = get_model_configurations()
    
    # Define datasets to evaluate (or None for all)
    target_datasets = [
        "ASAP-AES", "BEEtlE_2way", "CSEE", "EFL",
        "grade_like_a_human_dataset_os_q1", "Rice_Chem_Q1"
    ]
    
    # Run evaluation
    results = await integration.run_full_evaluation_pipeline(
        model_configs=model_configs,
        dataset_names=target_datasets,
        max_essays_per_dataset=30,  # Limit for testing
        submit_to_platform=True
    )
    
    print("\n" + "="*60)
    print("BESESR-MLLM Evaluation Complete!")
    print("="*60)
    
    if results:
        # Quick summary
        leaderboard = integration.tester.generate_leaderboard(results)
        print("\nTop 5 Models:")
        for i, (model, score) in enumerate(list(leaderboard.items())[:5], 1):
            print(f"{i}. {model}: {score:.3f}")
    
    print(f"\nResults saved in: {integration.results_dir}")

if __name__ == "__main__":
    asyncio.run(main())