import requests
import pandas as pd
from typing import Dict, List, Optional, Any
import random
from datetime import datetime
import os
import json
from datasets import load_dataset
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceDatasetLoader:
    """FULLY DYNAMIC dataset loader - auto-discovers datasets from HuggingFace profile"""
    
    def __init__(self):
        self.hf_api_base = "https://datasets-server.huggingface.co"
        self.cache: Dict[str, Any] = {}
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.username = "nlpatunt"  # Your HuggingFace username
        self.authenticated = False
        
        # Authenticate with HuggingFace
        if self.hf_token:
            try:
                login(token=self.hf_token, add_to_git_credential=False)
                self.authenticated = True
                print("🔐 Successfully authenticated with HuggingFace for dynamic discovery")
            except Exception as e:
                self.authenticated = False
                print(f"⚠️ HF authentication failed: {e}")
        else:
            self.authenticated = False
            print("⚠️ No HuggingFace token found - dynamic discovery disabled")
    
    def get_configured_datasets(self) -> Dict[str, Dict[str, Any]]:
        """TRULY dynamic with multi-config dataset support"""
        
        if not self.authenticated:
            print("❌ Not authenticated - cannot discover datasets dynamically")
            return self._get_fallback_datasets()
        
        try:
            print("🔍 Dynamically discovering datasets from HuggingFace profile...")
            
            api = HfApi(token=self.hf_token)
            user_datasets = list(api.list_datasets(author=self.username))
            
            print(f"📊 Found {len(user_datasets)} total datasets in profile")
            
            dynamic_datasets = {}
            processed_count = 0
            
            for dataset in user_datasets:
                try:
                    dataset_id = dataset.id
                    dataset_name = dataset_id.split('/')[-1]
                    
                    # 🔧 FIXED: Only use filtering, remove private bypass
                    if self._should_include_dataset(dataset_name):
                        print(f"📋 Auto-configuring: {dataset_name}")
                        
                        # ✅ Handle multi-config datasets
                        configs = self._auto_configure_dataset_with_configs(dataset_id, dataset_name)
                        
                        if configs:
                            # Add all configurations
                            for config_name, config in configs.items():
                                dynamic_datasets[config_name] = config
                                processed_count += 1
                                
                        if processed_count >= 12:  # 🔧 Changed limit to 12 since you want 12 datasets
                            print("✅ Reached target dataset count (12)")
                            break
                            
                except Exception as e:
                    print(f"⚠️ Error processing dataset {dataset.id}: {e}")
                    continue
            
            if dynamic_datasets:
                print(f"✅ Successfully discovered {len(dynamic_datasets)} datasets dynamically!")
                print(f"📋 Datasets: {list(dynamic_datasets.keys())}")
                
                self._cache_discovered_datasets(dynamic_datasets)
                return dynamic_datasets
            else:
                print("⚠️ No datasets discovered, falling back to static configuration")
                return self._get_fallback_datasets()
                
        except Exception as e:
            print(f"❌ Error during dynamic discovery: {e}")
            return self._get_fallback_datasets()
    
    def _should_include_dataset(self, dataset_name: str) -> bool:
        """Ultra-strict filtering - exact name matching only"""
        
        # EXACT NAMES of your 12 target datasets (update with actual names from your HF profile)
        exact_target_names = {
            'ASAP-AES', 'ASAP-SAS', 'ASAP2', 'ASAP_plus_plus',
            'rice__chem', 'CSEE', 'EFL', 'grade_like_a_human_dataset_os', 
            'persuade_2', 'BEEtlE', 'SciEntSBank', 'automatic_short_answer_grading_mohlar'
        }
        print(f"🔍 Checking dataset: {dataset_name}")  
        if dataset_name in exact_target_names:
            print(f"   ✅ Including {dataset_name}: Exact target match")
            return True
        else:
            print(f"   ❌ Excluding {dataset_name}: Not in exact target list")
            return False
    
    def _auto_configure_dataset_with_configs(self, dataset_id: str, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """Auto-configure dataset handling multiple configs (2way/3way)"""
        
        configs_to_try = self._get_possible_configs(dataset_name)
        
        if not configs_to_try:
            # Single config dataset
            config = self._auto_configure_single_dataset(dataset_id, dataset_name, None)
            if config:
                return {dataset_name: config}
            else:
                return {}
        
        # Multi-config dataset
        results = {}
        
        for config_name in configs_to_try:
            try:
                print(f"   🔧 Trying config: {config_name}")
                config = self._auto_configure_single_dataset(dataset_id, dataset_name, config_name)
                
                if config:
                    # Create unique name for each config
                    unique_name = f"{dataset_name}_{config_name}"
                    config['config'] = config_name  # Store the config name
                    results[unique_name] = config
                    print(f"   ✅ Successfully configured: {unique_name}")
                
            except Exception as e:
                print(f"   ❌ Failed config {config_name}: {e}")
                continue
        
        return results
    
    def _get_possible_configs(self, dataset_name: str) -> List[str]:
        """Get possible configurations for multi-config datasets"""
        
        multi_config_datasets = {
        }
        
        dataset_lower = dataset_name.lower()
        
        for keyword, configs in multi_config_datasets.items():
            if keyword in dataset_lower:
                return configs
        
        return []  # Single config dataset

    def _auto_configure_single_dataset(self, dataset_id: str, dataset_name: str, config_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Configure a single dataset with specific config"""
        try:
            # MANUAL CONFIGS for problematic datasets
            manual_configs = {
                'ASAP2': {
                    "description": "ASAP 2.0 Dataset - Automated Essay Scoring",
                    "essay_column": "full_text",
                    "score_column": "score", 
                    "prompt_column": "prompt_name",
                    "score_range": (0, 60)
                },
                'BEEtlE': {
                    "description": "Basic Elements of English Teaching and Learning Evaluation",
                    "essay_column": "student_answer",
                    "score_column": "label",
                    "prompt_column": "question",
                    "score_range": (0, 2)
                },
                'SciEntSBank': {
                    "description": "Science Entailment Bank Dataset",
                    "essay_column": "student_answer",
                    "score_column": "label", 
                    "prompt_column": "question",
                    "score_range": (0, 2)
                },
                'automatic_short_answer_grading_mohlar': {
                    "description": "Automatic Short Answer Grading - Mohlar Dataset",
                    "essay_column": "student_answer",
                    "score_column": "score",
                    "prompt_column": "question", 
                    "score_range": (0, 5)
                },
                'CSEE': {
                    "description": "Computer Science Essay Evaluation Dataset",
                    "essay_column": "essay_text",
                    "score_column": "score",
                    "prompt_column": "prompt",
                    "score_range": (0, 5)
                }
            }
            
            # Check if this dataset needs manual config
            if dataset_name in manual_configs:
                print(f"   🔧 Using manual config for {dataset_name}")
                base_config = manual_configs[dataset_name].copy()
                return {
                    "huggingface_id": dataset_id,
                    "config": config_name,
                    "split": "train",
                    "auto_discovered": False,
                    "manual_override": True,
                    "discovery_time": datetime.now().isoformat(),
                    **base_config
                }
            
            # Continue with auto-configuration for other datasets
            print(f"   🔧 Starting auto-configuration for {dataset_name}...")
            
            # Get appropriate split
            split_needed = self._get_dataset_split(dataset_name)
            
            # Load sample with specific config
            try:
                if config_name:
                    dataset = load_dataset(
                        dataset_id, 
                        config_name,
                        split=f"{split_needed}[:3]", 
                        token=self.hf_token,
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        dataset_id, 
                        split=f"{split_needed}[:3]", 
                        token=self.hf_token,
                        trust_remote_code=True
                    )
            except TypeError:
                # Fallback for older versions
                dataset = load_dataset(
                    dataset_id, 
                    config_name,
                    split=f"{split_needed}[:3]", 
                    use_auth_token=self.hf_token
                )
            
            if len(dataset) == 0:
                return None
            
            sample = dataset[0]
            columns = list(sample.keys())
            
            # Smart column detection
            essay_col = self._detect_essay_column(columns, sample)
            score_col = self._detect_score_column(columns, sample)
            prompt_col = self._detect_prompt_column(columns, sample)
            
            # Auto-detect score range
            score_range = self._detect_score_range(dataset, score_col)
            
            # Skip if essential columns not found
            if not essay_col or not score_col:
                return None
            
            # Generate enhanced description
            description = self._generate_enhanced_description(dataset_name, config_name, columns, sample)
            
            # Generate configuration
            config = {
                "huggingface_id": dataset_id,
                "description": description,
                "essay_column": essay_col,
                "score_column": score_col,
                "prompt_column": prompt_col,
                "config": config_name,
                "split": split_needed,
                "score_range": score_range,
                "auto_discovered": True,
                "discovery_time": datetime.now().isoformat(),
                "column_count": len(columns),
                "dataset_variant": config_name
            }
            
            return config
            
        except Exception as e:
            print(f"   ❌ Error configuring {dataset_name} with config {config_name}: {e}")
            return None

    def _get_dataset_split(self, dataset_name: str) -> str:
        """Get appropriate split for dataset"""
        
        # Some datasets only have test split
        test_only_keywords = []
        dataset_lower = dataset_name.lower()
        
        if any(keyword in dataset_lower for keyword in test_only_keywords):
            return 'test'
        
        return 'train'  # Default

    def _generate_enhanced_description(self, dataset_name: str, config_name: Optional[str], columns: list, sample: dict) -> str:
        """Generate enhanced description including config info"""
        
        # Base descriptions
        base_descriptions = {
            'beetle': "Basic Elements of English Teaching and Learning Evaluation",
            'scientsbank': "Science Entailment Bank",
            'asap': "Automated Student Assessment Prize",
            'rice': "Rice University Chemistry Dataset",
            'csee': "Computer Science Essay Evaluation",
            'efl': "English as Foreign Language Essays",
            'persuade': "Persuasive Essays Dataset",
            'grade': "Grade Like a Human Dataset",
            'mohlar': "Automatic Short Answer Grading - Mohlar Dataset"
        }
        
        # Find base description
        dataset_lower = dataset_name.lower()
        base_desc = None
        
        for keyword, desc in base_descriptions.items():
            if keyword in dataset_lower:
                base_desc = desc
                break
        
        if not base_desc:
            base_desc = f"Auto-discovered dataset: {dataset_name.replace('_', ' ').title()}"
        
        # Add config information
        if config_name:
            if config_name == '2way':
                base_desc += " (2-way classification)"
            elif config_name == '3way':
                base_desc += " (3-way classification)"
            else:
                base_desc += f" ({config_name} configuration)"
        
        return base_desc

    def _get_dataset_config(self, dataset_name: str) -> Optional[str]:
        """Get required config for datasets that need specific configs"""
        config_mapping = {
            'beetle': '2way',
            'scientsbank': '2way', 
        }
        
        dataset_lower = dataset_name.lower()
        for keyword, config in config_mapping.items():
            if keyword in dataset_lower:
                return config
        
        return None

    def _get_dataset_split(self, dataset_name: str) -> str:
        """Get appropriate split for dataset"""
        
        # Some datasets only have test split
        test_only_keywords = []
        dataset_lower = dataset_name.lower()
        
        if any(keyword in dataset_lower for keyword in test_only_keywords):
            return 'test'
        
        return 'train'  # Default

    def _detect_essay_column_enhanced(self, columns: list, sample: dict, dataset_name: str) -> str:
        """Enhanced essay column detection with dataset-specific fixes"""
        
        # Dataset-specific fixes
        if 'efl' in dataset_name.lower():
            # For EFL dataset, look for actual text columns
            for col in ['essay_text', 'text', 'essay', 'content']:
                if col in columns and isinstance(sample.get(col), str):
                    if len(sample[col]) > 50:  # Ensure it's actual essay text
                        return col
        
        # Original detection logic
        essay_keywords = [
            'full_text','essay_text', 'essay', 'text', 'content', 'response', 
            'student_answer', 'answer', 'writing', 'student_response'
        ]
        
        # Check for exact matches first
        candidates = []
        for col in columns:
            col_lower = col.lower()
            for keyword in essay_keywords:
                if keyword in col_lower:
                    if isinstance(sample[col], str) and len(sample[col]) > 30:
                        priority = essay_keywords.index(keyword)
                        candidates.append((col, priority, len(sample[col])))
                    break
        
        if candidates:
            candidates.sort(key=lambda x: (x[1], -x[2]))
            return candidates[0][0]
        
        # Fallback: find longest text column
        text_columns = []
        for col in columns:
            if isinstance(sample[col], str) and len(sample[col]) > 20:
                text_columns.append((col, len(sample[col])))
        
        if text_columns:
            return max(text_columns, key=lambda x: x[1])[0]
        
        return columns[0] if columns else "text"

    def _detect_score_column_enhanced(self, columns: list, sample: dict, dataset_name: str) -> str:
        """Enhanced score column detection"""
        
        score_keywords = [
            'overall_score', 'holistic_score', 'domain1_score', 'grade', 
            'score', 'rating', 'total_score', 'final_score', 'label'
        ]
        
        # Check for exact matches first
        for keyword in score_keywords:
            for col in columns:
                if keyword.lower() == col.lower():
                    if isinstance(sample[col], (int, float)) and sample[col] is not None:
                        return col
        
        # Check for partial matches
        for keyword in score_keywords:
            for col in columns:
                if keyword.lower() in col.lower():
                    if isinstance(sample[col], (int, float)) and sample[col] is not None:
                        return col
        
        # Fallback: find first numeric column
        for col in columns:
            if isinstance(sample[col], (int, float)) and sample[col] is not None:
                if 0 <= sample[col] <= 100:  # Reasonable score range
                    return col
        
        return columns[1] if len(columns) > 1 else columns[0]

    
    def _detect_essay_column(self, columns: list, sample: dict) -> str:
        """Auto-detect essay/text column using smart heuristics"""
        
        # Priority keywords for essay columns
        essay_keywords = [
            'full_text','essay', 'text', 'content', 'response', 'student_answer', 
            'answer', 'writing', 'student_response', 'full_text'
        ]
        
        # Check for exact/partial matches
        candidates = []
        for col in columns:
            col_lower = col.lower()
            for keyword in essay_keywords:
                if keyword in col_lower:
                    # Check if it actually contains text data
                    if isinstance(sample[col], str) and len(sample[col]) > 30:
                        priority = essay_keywords.index(keyword)  # Earlier = higher priority
                        candidates.append((col, priority, len(sample[col])))
                    break
        
        if candidates:
            # Sort by priority, then by text length
            candidates.sort(key=lambda x: (x[1], -x[2]))
            return candidates[0][0]
        
        # Fallback: find longest text column
        text_columns = []
        for col in columns:
            if isinstance(sample[col], str) and len(sample[col]) > 10:
                text_columns.append((col, len(sample[col])))
        
        if text_columns:
            return max(text_columns, key=lambda x: x[1])[0]
        
        # Ultimate fallback
        return columns[0] if columns else "text"
    
    def _detect_score_column(self, columns: list, sample: dict) -> str:
        """Auto-detect score column using smart heuristics"""
        
        score_keywords = [
            'holistic_score', 'domain1_score', 'score', 'grade', 'rating', 
            'overall_score', 'total_score', 'final_score','label'
        ]
        
        # Check for exact/partial matches
        for keyword in score_keywords:
            for col in columns:
                if keyword.lower() in col.lower():
                    # Verify it's numeric
                    if isinstance(sample[col], (int, float)) and sample[col] is not None:
                        return col
        
        # Fallback: find first numeric column that looks like a score
        for col in columns:
            if isinstance(sample[col], (int, float)) and sample[col] is not None:
                # Check if values are in reasonable score range
                if 0 <= sample[col] <= 100:  # Reasonable score range
                    return col
        
        # Ultimate fallback
        return columns[1] if len(columns) > 1 else columns[0]
    
    def _detect_prompt_column(self, columns: list, sample: dict) -> str:
        """Auto-detect prompt column"""
        
        prompt_keywords = [
            'question_text','prompt_name', 'prompt', 'question', 'task', 'essay_set', 'set', 
            'prompt_text', 'writing_prompt', 'assignment'
        ]
        
        for keyword in prompt_keywords:
            for col in columns:
                if keyword.lower() in col.lower():
                    return col
        
        # Look for text columns that aren't the essay
        for col in columns:
            if isinstance(sample[col], str) and 10 <= len(sample[col]) <= 200:
                return col
        
        return "prompt"  # Default fallback
    
    def _detect_rubric_columns(self, columns: list, sample: dict) -> List[str]:
        """Detect rubric/dimension scoring columns"""
        
        rubric_keywords = [
            'domain', 'trait', 'dimension', 'criteria', 'rubric',
            'content', 'organization', 'style', 'grammar', 'mechanics'
        ]
        
        rubric_cols = []
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in rubric_keywords):
                if isinstance(sample[col], (int, float)) and sample[col] is not None:
                    rubric_cols.append(col)
        
        return rubric_cols
    
    def _detect_score_range(self, dataset, score_col: str) -> tuple:
        """Auto-detect score range from dataset sample"""
        try:
            # Sample more data to get accurate range
            sample_size = min(20, len(dataset))
            larger_sample = dataset.select(range(sample_size))
            
            scores = []
            for row in larger_sample:
                if row[score_col] is not None:
                    scores.append(row[score_col])
            
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                
                # Round to reasonable bounds
                if min_score == max_score:
                    return (0, 5)  # Default range
                
                return (int(min_score), int(max_score))
            else:
                return (0, 5)  # Default fallback
                
        except Exception as e:
            print(f"   ⚠️ Error detecting score range: {e}")
            return (0, 5)  # Default fallback
    
    def _generate_description(self, dataset_name: str, columns: list, sample: dict) -> str:
        """Generate a description based on dataset name and structure"""
        
        # Known dataset patterns
        if 'asap' in dataset_name.lower():
            if 'sas' in dataset_name.lower():
                return "Automated Student Assessment Prize - Short Answer Scoring"
            else:
                return "Automated Student Assessment Prize - Automated Essay Scoring"
        
        elif 'rice' in dataset_name.lower():
            return "Rice University Chemistry Dataset"
        
        elif 'beetle' in dataset_name.lower():
            return "Basic Elements of English Teaching and Learning Evaluation"
        
        elif 'persuade' in dataset_name.lower():
            return "Persuasive Essays Dataset"
        
        elif 'csee' in dataset_name.lower():
            return "Computer Science Essay Evaluation"
        
        elif 'efl' in dataset_name.lower():
            return "English as Foreign Language Essays"
        
        elif 'grade' in dataset_name.lower():
            return "Grade Like a Human Dataset"
        
        else:
            # Generate based on content
            return f"Auto-discovered essay dataset: {dataset_name.replace('_', ' ').title()}"
    
    def _cache_discovered_datasets(self, datasets: Dict[str, Dict[str, Any]]):
        """Cache discovered datasets for performance"""
        try:
            cache_file = "dataset_discovery_cache.json"
            cache_data = {
                "discovered_at": datetime.now().isoformat(),
                "username": self.username,
                "dataset_count": len(datasets),
                "datasets": datasets
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"💾 Cached {len(datasets)} discovered datasets")
            
        except Exception as e:
            print(f"⚠️ Failed to cache discoveries: {e}")
    
    def _get_fallback_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Fallback static configuration if dynamic discovery fails"""
        print("🔄 Using fallback static dataset configuration")
        
        return {
            "ASAP-AES": {
                "huggingface_id": "nlpatunt/ASAP-AES",
                "description": "Automated Student Assessment Prize - Automated Essay Scoring",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "config": None,
                "split": "train",
                "score_range": (0, 60),
                "auto_discovered": False
            },
            "rice_chem": {
                "huggingface_id": "nlpatunt/rice_chem",
                "description": "Rice University Chemistry Dataset",
                "essay_column": "essay_text",
                "score_column": "holistic_score",
                "prompt_column": "prompt",
                "config": None,
                "split": "train",
                "score_range": (1, 5),
                "auto_discovered": False
            }
            # Add minimal fallback datasets here...
        }
    
    def refresh_dataset_discovery(self) -> Dict[str, Dict[str, Any]]:
        """Manually refresh dataset discovery"""
        print("🔄 Refreshing dataset discovery...")
        
        # Clear cache
        self.cache.clear()
        
        # Re-discover
        return self.get_configured_datasets()
    
    def inspect_dataset_structure(self, dataset_id: str, sample_size: int = 3):
        """Inspect actual dataset structure to get correct column names"""
        try:
            print(f"🔍 Inspecting dataset: {dataset_id}")
            
            # Load small sample
            try:
                dataset = load_dataset(dataset_id, split="train[:3]", token=self.hf_token)
            except TypeError:
                dataset = load_dataset(dataset_id, split="train[:3]", use_auth_token=self.hf_token)
            
            if len(dataset) == 0:
                return {"error": "No data found"}
            
            # Get first sample
            sample = dataset[0]
            
            analysis = {
                "dataset_id": dataset_id,
                "total_columns": len(sample.keys()),
                "column_names": list(sample.keys()),
                "sample_data": {},
                "potential_mappings": {
                    "essay_columns": [],
                    "score_columns": [],
                    "prompt_columns": [],
                    "rubric_columns": []
                },
                "score_analysis": {}
            }
            
            # Show sample data for each column (truncated)
            for col, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    analysis["sample_data"][col] = value[:100] + "..."
                else:
                    analysis["sample_data"][col] = value
            
            # Smart column detection
            for col in sample.keys():
                col_lower = col.lower()
                
                # Detect essay/text columns
                if any(keyword in col_lower for keyword in ['essay', 'text', 'content', 'response', 'answer']):
                    analysis["potential_mappings"]["essay_columns"].append(col)
                
                # Detect score columns  
                if any(keyword in col_lower for keyword in ['score', 'grade', 'rating', 'domain', 'holistic']):
                    analysis["potential_mappings"]["score_columns"].append(col)
                    
                    # Analyze score ranges
                    try:
                        scores = [row[col] for row in dataset if row[col] is not None]
                        if scores:
                            analysis["score_analysis"][col] = {
                                "min": min(scores),
                                "max": max(scores),
                                "unique_values": list(set(scores))[:10],
                                "count": len(scores)
                            }
                    except:
                        pass
                
                # Detect prompt columns
                if any(keyword in col_lower for keyword in ['prompt', 'question', 'task', 'set']):
                    analysis["potential_mappings"]["prompt_columns"].append(col)
                
                # Detect rubric columns
                if any(keyword in col_lower for keyword in ['rubric', 'criteria', 'trait', 'dimension']):
                    analysis["potential_mappings"]["rubric_columns"].append(col)
            
            return analysis
            
        except Exception as e:
            return {"error": str(e), "dataset_id": dataset_id}
    
    def load_dataset_sample(
        self,
        dataset_id: str,
        config: Optional[str] = None,
        split: str = "train",
        sample_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Load a sample from a HF dataset (public or private)"""

        print(f"📥 Loading sample from {dataset_id} (config={config}, split={split})")

        try:
            # Try using the modern token parameter
            try:
                ds = load_dataset(dataset_id, config, split=split, token=self.hf_token)
            except TypeError:
                # Fallback for older versions
                ds = load_dataset(dataset_id, config, split=split, use_auth_token=self.hf_token)
        except Exception as e:
            print(f"❌ load_dataset failed: {e}")
            return self._load_via_api(dataset_id, config, split, sample_size)

        # Sample rows
        if len(ds) > sample_size:
            idx = random.sample(range(len(ds)), sample_size)
            ds = ds.select(idx)

        essays: List[Dict[str, Any]] = [{"row": r} for r in ds]
        print(f"✅ Loaded {len(essays)} rows from {dataset_id}")
        return essays
    
    def _load_via_api(self, dataset_id: str, config: str = None, split: str = "train", 
                     sample_size: int = 100) -> List[Dict[str, Any]]:
        """Fallback method using HuggingFace API"""
        try:
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            url = f"{self.hf_api_base}/rows"
            params = {
                "dataset": dataset_id,
                "config": config or "default",
                "split": split,
                "offset": 0,
                "length": min(sample_size, 100)
            }
            
            print(f"📡 Trying API access for {dataset_id}...")
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                essays = data.get("rows", [])
                print(f"✅ Loaded {len(essays)} essays via API from {dataset_id}")
                return essays
            else:
                print(f"⚠️ API access failed for {dataset_id}: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ API fallback failed for {dataset_id}: {e}")
            return []


class BESESRDatasetManager:
    """Enhanced dataset manager with dynamic discovery support"""
    
    def __init__(self):
        self.hf_loader = HuggingFaceDatasetLoader()
        print("🚀 Initializing dynamic dataset discovery...")
        self.datasets_config = self.hf_loader.get_configured_datasets()
        print(f"📊 Initialized with {len(self.datasets_config)} datasets")
    
    def refresh_datasets(self) -> int:
        """Refresh dataset discovery from HuggingFace"""
        print("🔄 Refreshing dataset discovery...")
        self.datasets_config = self.hf_loader.refresh_dataset_discovery()
        return len(self.datasets_config)
    
    def load_dataset_for_evaluation(self, dataset_name: str, sample_size: int = 50) -> List[Dict[str, Any]]:
        """Load dataset with enhanced column detection"""
        
        if dataset_name not in self.datasets_config:
            print(f"❌ Unknown dataset: {dataset_name}")
            return []
        
        config = self.datasets_config[dataset_name]
        print(f"🔐 Loading dataset: {dataset_name} ({'auto-discovered' if config.get('auto_discovered') else 'static'})")
        
        # Load from HuggingFace dataset
        raw_data = self.hf_loader.load_dataset_sample(
            dataset_id=config["huggingface_id"],
            config=config["config"],
            split=config["split"],
            sample_size=sample_size
        )
        
        if not raw_data:
            print(f"⚠️ No data loaded for {dataset_name}, using fallback")
            return [self.get_sample_essay(dataset_name)]
        
        # Convert to standard format with flexible column mapping
        standardized_essays = []
        for item in raw_data:
            try:
                row = item.get("row", {})
                
                # Flexible column mapping
                essay_text = self._get_column_value(row, [
                    config["essay_column"],
                    "essay", "text", "essay_text", "content", "student_answer", "response"
                ])
                
                prompt = self._get_column_value(row, [
                    config["prompt_column"],
                    "prompt", "question", "essay_set", "task", "assignment"
                ])
                
                human_score = self._get_column_value(row, [
                    config["score_column"],
                    "score", "holistic_score", "domain1_score", "grade", "rating"
                ])
                
                # Get rubric scores if available
                rubric_scores = {}
                if config.get("has_rubric", False):
                    for rubric_col in config.get("rubric_columns", []):
                        if rubric_col in row and row[rubric_col] is not None:
                            rubric_scores[rubric_col] = row[rubric_col]
                
                if essay_text and human_score is not None:
                    essay = {
                        "essay_id": f"{dataset_name}_{len(standardized_essays)}",
                        "dataset_name": dataset_name,
                        "essay_text": str(essay_text),
                        "prompt": str(prompt) if prompt else f"Prompt for {dataset_name}",
                        "human_score": float(human_score),
                        "score_range": config["score_range"],
                        "rubric_scores": rubric_scores,
                        "metadata": {
                            "original_row": row,
                            "loaded_at": datetime.now().isoformat(),
                            "source": "dynamic_huggingface" if config.get("auto_discovered") else "static_config",
                            "discovery_time": config.get("discovery_time")
                        }
                    }
                    
                    # Validate essay has sufficient content
                    if len(essay["essay_text"].strip()) > 20:
                        standardized_essays.append(essay)
                        
            except Exception as e:
                print(f"⚠️ Error processing essay in {dataset_name}: {e}")
                continue
        
        print(f"✅ Standardized {len(standardized_essays)} essays from {dataset_name}")
        return standardized_essays
    
    def _get_column_value(self, row: dict, possible_columns: list):
        """Try multiple possible column names and return first found value"""
        for col in possible_columns:
            if col in row and row[col] is not None:
                return row[col]
        return None
    
    def get_sample_essay(self, dataset_name: str) -> Dict[str, Any]:
        """Get a fallback sample essay"""
        return {
            "essay_id": f"{dataset_name}_fallback",
            "dataset_name": dataset_name,
            "essay_text": """
            Technology has fundamentally transformed education in the 21st century.
            Digital tools and platforms have revolutionized how students learn and teachers instruct,
            creating more interactive and personalized learning experiences.
            However, the integration of technology also presents challenges that must be addressed.
            """,
            "prompt": f"Sample prompt for {dataset_name}",
            "human_score": 3.5,
            "score_range": (1, 5),
            "metadata": {"type": "fallback", "reason": "dataset_load_failed"}
        }

# Global instance with dynamic discovery
dataset_manager = BESESRDatasetManager()