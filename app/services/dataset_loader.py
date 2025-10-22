import requests
import pandas as pd
from typing import Dict, List, Optional, Any
import random
from datetime import datetime
import os
import json
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()


class HuggingFaceDatasetLoader:
    """FULLY DYNAMIC dataset loader with config support - auto-discovers datasets from HuggingFace profile"""

    def __init__(self):
        self.hf_api_base = "https://datasets-server.huggingface.co"
        self.cache: Dict[str, Any] = {}
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        # Try fallback token loading
        if not self.hf_token:
            try:
                with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                    self.hf_token = f.read().strip()
                print(f"🔑 Loaded HF token from file: {self.hf_token[:15]}...")
            except Exception as e:
                print(f"❌ Could not load HF token: {e}")
                self.hf_token = None

        self.username = "nlpatunt"  # Your HuggingFace username
        self.authenticated = False

        # Authenticate with HuggingFace
        if self.hf_token:
            try:
                login(token=self.hf_token, add_to_git_credential=False)
                self.authenticated = True
                print("✅ HF authentication successful")
            except Exception as e:
                self.authenticated = False
                print(f"⚠️ HF authentication failed: {e}")
        else:
            self.authenticated = False
            print("⚠️ No HuggingFace token found - dynamic discovery disabled")

    def get_configured_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Load datasets from HuggingFace Collection with config support"""

        if not self.authenticated:
            print("❌ Not authenticated - cannot access collection")
            return self._get_fallback_datasets()

        try:
            print("🔍 Loading datasets from HuggingFace Collection with config support...")

            api = HfApi(token=self.hf_token)

            # Your collection ID (from the URL)
            collection_name = "nlpatunt/automatic-grading-datasets-without-labels-68a38400f057ffe50522c401"

            # Get the collection
            collection = api.get_collection(collection_name)
            print(f"📋 Found collection: {collection.title}")
            print(f"📊 Collection has {len(collection.items)} items")

            dynamic_datasets = {}
            processed_count = 0

            # Process each dataset in the collection
            for item in collection.items:
                try:
                    if item.item_type == "dataset":
                        dataset_id = item.item_id
                        dataset_name = dataset_id.split("/")[-1]  # Get just the name part

                        # Remove D_ prefix if it exists
                        if dataset_name.startswith("D_"):
                            dataset_name = dataset_name[2:]

                        print(f"📋 Processing collection dataset: {dataset_name}")

                        # Handle multi-config datasets with enhanced detection
                        configs = self._auto_configure_dataset_with_configs(dataset_id, dataset_name)

                        if configs:
                            # Add all configurations
                            for config_name, config in configs.items():
                                dynamic_datasets[config_name] = config
                                processed_count += 1
                                print(f"   ✅ Added: {config_name}")

                except Exception as e:
                    print(f"⚠️ Error processing collection item {item.item_id}: {e}")
                    continue

            # MOVE THIS OUTSIDE THE LOOP - process all datasets after collection is complete
            if dynamic_datasets:
                print(f"✅ Successfully loaded {len(dynamic_datasets)} dataset configs from collection!")

                # Create both D_ and regular versions
                expanded_datasets = {}

                for name, config in dynamic_datasets.items():
                    if name.startswith("D_"):
                        # This is already a D_ dataset (unlabeled)
                        expanded_datasets[name] = {**config, "dataset_type": "unlabeled"}
                        expanded_datasets[name] = self._override_dataset_config(name, expanded_datasets[name])

                        # Create regular version for ground truth
                        regular_name = name[2:]  # Remove "D_" prefix
                        gt_config = config.copy()
                        gt_config["huggingface_id"] = gt_config["huggingface_id"].replace("/D_", "/")
                        gt_config["dataset_type"] = "ground_truth"
                        expanded_datasets[regular_name] = gt_config

                    else:
                        # This is a regular dataset name, create both versions
                        # Regular version (ground truth)
                        expanded_datasets[name] = {**config, "dataset_type": "ground_truth"}

                        # D_ version (unlabeled)
                        d_name = f"D_{name}"
                        d_config = config.copy()
                        d_config["huggingface_id"] = d_config["huggingface_id"].replace(f"/{name}", f"/D_{name}")
                        d_config["dataset_type"] = "unlabeled"
                        expanded_datasets[d_name] = d_config

                self._cache_discovered_datasets(expanded_datasets)
                return expanded_datasets
            else:
                print("⚠️ No datasets found in collection, falling back to static configuration")
                return self._get_fallback_datasets()

        except Exception as e:
            print(f"❌ Error loading collection: {e}")
            print("🔄 Falling back to static configuration")
            return self._get_fallback_datasets()

    def _auto_configure_dataset_with_configs(self, dataset_id: str, dataset_name: str) -> Dict[str, Dict[str, Any]]:
        """Auto-configure dataset handling multiple configs with enhanced detection"""

        # Known multi-config datasets - handle these specially (after D_ removal)
        known_multi_config = {
            "OS_Dataset": ["q1", "q2", "q3", "q4", "q5"],
            "Rice_Chem": ["Q1", "Q2", "Q3", "Q4"],
            "BEEtlE": ["2way", "3way"],
            "SciEntSBank": ["2way", "3way"],
        }

        # Check if this is a known multi-config dataset
        if dataset_name in known_multi_config:
            print(f"   📁 Known multi-config dataset: {dataset_name}")
            configs = known_multi_config[dataset_name]
            print(f"   🔧 Using known configs: {configs}")

            results = {}
            for config_name in configs:
                try:
                    print(f"   🔧 Configuring {dataset_name} with config: {config_name}")
                    config = self._auto_configure_single_dataset(dataset_id, dataset_name, config_name)

                    if config:
                        # Create unique name for each config
                        unique_name = f"{dataset_name}_{config_name}"
                        config["config"] = config_name
                        results[unique_name] = config
                        print(f"   ✅ Successfully configured: {unique_name}")
                    else:
                        print(f"   ❌ Failed to configure: {dataset_name}_{config_name}")

                except Exception as e:
                    print(f"   ❌ Error configuring {config_name}: {e}")
                    continue

            if results:
                return results

        # For other datasets, try automatic config detection
        try:
            print(f"   🔍 Checking for configs in {dataset_name}...")
            configs = get_dataset_config_names(dataset_id, token=self.hf_token)
            print(f"   📁 Found configs: {configs}")

            if configs and len(configs) > 1:
                # Multiple configs detected
                results = {}
                for config_name in configs:
                    try:
                        print(f"   🔧 Configuring {dataset_name} with config: {config_name}")
                        config = self._auto_configure_single_dataset(dataset_id, dataset_name, config_name)

                        if config:
                            # Create unique name for each config
                            unique_name = f"{dataset_name}_{config_name}"
                            config["config"] = config_name
                            results[unique_name] = config
                            print(f"   ✅ Successfully configured: {unique_name}")
                        else:
                            print(f"   ❌ Failed to configure: {dataset_name}_{config_name}")

                    except Exception as e:
                        print(f"   ❌ Error configuring {config_name}: {e}")
                        continue

                if results:
                    return results

            elif configs and len(configs) == 1:
                # Single config
                config_name = configs[0] if configs[0] != "default" else None
                config = self._auto_configure_single_dataset(dataset_id, dataset_name, config_name)
                if config:
                    return {dataset_name: config}

        except Exception as e:
            print(f"   ⚠️ Config detection failed: {e}")
            # For known problematic datasets, don't try simple loading
            if dataset_name in known_multi_config:
                print(f"   ❌ Skipping simple config attempt for known multi-config dataset")
                return {}

        # Try as simple dataset (no configs) - but not for known multi-config datasets
        if dataset_name not in known_multi_config:
            try:
                config = self._auto_configure_single_dataset(dataset_id, dataset_name, None)
                if config:
                    return {dataset_name: config}
            except Exception as e:
                print(f"   ❌ Simple dataset configuration also failed: {e}")

        return {}

    def _auto_configure_single_dataset(
        self, dataset_id: str, dataset_name: str, config_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        try:
            # Initialize variables at the top
            base_dataset_name = dataset_name.split("_")[0] if "_" in dataset_name else dataset_name
            config_key = None

            # COMPLETE MANUAL CONFIGS for all known datasets with correct column mappings
            manual_configs = {
                "ASAP-AES_Set1": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 1: Persuasive Essays (Score Range: 2-12)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 1,
                    "score_range": (2, 12),
                },
                "ASAP-AES_Set2_Domain1": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 2 Domain 1: Content (Score Range: 1-6)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 2,
                    "score_range": (1, 6),
                },
                "ASAP-AES_Set2_Domain2": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 2 Domain 2: Organization (Score Range: 1-4)",
                    "essay_column": "essay",
                    "score_column": "domain2_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 2,
                    "score_range": (1, 4),
                },
                "ASAP-AES_Set3": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 3: Source Dependent Responses (Score Range: 0-3)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 3,
                    "score_range": (0, 3),
                },
                "ASAP-AES_Set4": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 4: Source Dependent Responses (Score Range: 0-3)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 4,
                    "score_range": (0, 3),
                },
                "ASAP-AES_Set5": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 5: Source Dependent Responses (Score Range: 0-4)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 5,
                    "score_range": (0, 4),
                },
                "ASAP-AES_Set6": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 6: Source Dependent Responses (Score Range: 0-4)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 6,
                    "score_range": (0, 4),
                },
                "ASAP-AES_Set7": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 7: Narrative Essays (Score Range: 0-30)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 7,
                    "score_range": (0, 30),
                },
                "ASAP-AES_Set8": {
                    "huggingface_id": "nlpatunt/D_ASAP-AES",
                    "description": "ASAP-AES Essay Set 8: Narrative Essays (Score Range: 0-60)",
                    "essay_column": "essay",
                    "score_column": "domain1_score",
                    "prompt_column": "essay_set",
                    "essay_set_filter": 8,
                    "score_range": (0, 60),
                },
                "ASAP2": {
                    "description": "ASAP 2.0 Dataset - Automated Essay Scoring",
                    "essay_column": "full_text",
                    "score_column": "score",
                    "prompt_column": "assignment",
                    "score_range": (0, 4),
                },
                "ASAP-SAS": {
                    "description": "ASAP Short Answer Scoring",
                    "essay_column": "essay_text",
                    "score_column": "Score1",
                    "prompt_column": "essay_set",
                    "score_range": (0, 3),
                },
                "ASAP_plus_plus": {
                    "description": "ASAP++ Enhanced Dataset",
                    "essay_column": "essay",
                    "score_column": "overall_score",
                    "prompt_column": "essay_set",
                    "score_range": (0, 60),
                },
                "CSEE": {
                    "description": "Computer Science Essay Evaluation Dataset",
                    "essay_column": "essay",
                    "score_column": "overall_score",
                    "prompt_column": "prompt",
                    "score_range": (0, 16),
                },
                "BEEtlE_2way": {
                    "description": "BEEtlE 2-way: Correct/Incorrect",
                    "essay_column": "student_answer",
                    "score_column": "label",
                    "prompt_column": "question_text",
                    "score_range": (0, 1),
                },
                "BEEtlE_3way": {
                    "description": "BEEtlE 3-way: Correct/Incorrect/Partially Correct",
                    "essay_column": "student_answer",
                    "score_column": "label",
                    "prompt_column": "question_text",
                    "score_range": (0, 2),
                },
                "SciEntSBank_2way": {
                    "description": "SciEntSBank 2-way: Correct/Incorrect",
                    "essay_column": "student_answer",
                    "score_column": "label",
                    "prompt_column": "question_text",
                    "score_range": (0, 1),
                },
                "SciEntSBank_3way": {
                    "description": "SciEntSBank 3-way: Correct/Incorrect/Contradictory",
                    "essay_column": "student_answer",
                    "score_column": "label",
                    "prompt_column": "question_text",
                    "score_range": (0, 2),
                },
                "Mohlar": {
                    "description": "Automatic Short Answer Grading - Mohlar Dataset",
                    "essay_column": "student_answer",
                    "score_column": "grade",
                    "prompt_column": "question",
                    "score_range": (0, 5),
                },
                "automatic_short_answer_grading_mohlar": {
                    "description": "Automatic Short Answer Grading - Mohlar Dataset",
                    "essay_column": "student_answer",
                    "score_column": "grade",
                    "prompt_column": "question",
                    "score_range": (0, 5),
                },
                "Ielts_Writing_Dataset": {
                    "description": "IELTS Writing Assessment Dataset",
                    "essay_column": "Essay",
                    "score_column": "Overall_Score",
                    "prompt_column": "Question",
                    "score_range": (1, 9),
                },
                "Ielts_Writing_Task_2_Dataset": {
                    "description": "IELTS Writing Task 2 Dataset",
                    "essay_column": "essay",
                    "score_column": "band_score",
                    "prompt_column": "prompt",
                    "score_range": (1, 9),
                },
                "OS_Dataset_q1": {
                    "description": "Grade Like a Human OS Question 1 (Score Range: 0-19)",
                    "essay_column": "answer",
                    "score_column": "score_1",
                    "prompt_column": "question",
                    "score_range": (0, 19),
                },
                "OS_Dataset_q2": {
                    "description": "Grade Like a Human OS Question 2 (Score Range: 0-16)",
                    "essay_column": "answer",
                    "score_column": "score_1",
                    "prompt_column": "question",
                    "score_range": (0, 16),
                },
                "OS_Dataset_q3": {
                    "description": "Grade Like a Human OS Question 3 (Score Range: 0-15)",
                    "essay_column": "answer",
                    "score_column": "score_1",
                    "prompt_column": "question",
                    "score_range": (0, 15),
                },
                "OS_Dataset_q4": {
                    "description": "Grade Like a Human OS Question 4 (Score Range: 0-16)",
                    "essay_column": "answer",
                    "score_column": "score_1",
                    "prompt_column": "question",
                    "score_range": (0, 16),
                },
                "OS_Dataset_q5": {
                    "description": "Grade Like a Human OS Question 5 (Score Range: 0-27)",
                    "essay_column": "answer",
                    "score_column": "score_1",
                    "prompt_column": "question",
                    "score_range": (0, 27),
                },
                "persuade_2": {
                    "description": "Persuasive Essays Dataset v2",
                    "essay_column": "full_text",
                    "score_column": "holistic_essay_score",
                    "prompt_column": "assignment",
                    "score_range": (1, 6),
                },
                "Regrading_Dataset_J2C": {
                    "description": "Regrading Dataset J2C",
                    "essay_column": "student_answer",
                    "score_column": "grade",
                    "prompt_column": "Question",
                    "score_range": (0, 8),
                },
                "Rice_Chem_Q1": {
                    "description": "Rice Chemistry Question 1 (Score Range: 0-8)",
                    "essay_column": "student_response",
                    "score_column": "Score",
                    "prompt_column": "Prompt",
                    "score_range": (0, 8),
                },
                "Rice_Chem_Q2": {
                    "description": "Rice Chemistry Question 2 (Score Range: 0-8)",
                    "essay_column": "student_response",
                    "score_column": "Score",
                    "prompt_column": "Prompt",
                    "score_range": (0, 8),
                },
                "Rice_Chem_Q3": {
                    "description": "Rice Chemistry Question 3 (Score Range: 0-9)",
                    "essay_column": "student_response",
                    "score_column": "Score",
                    "prompt_column": "Prompt",
                    "score_range": (0, 9),
                },
                "Rice_Chem_Q4": {
                    "description": "Rice Chemistry Question 4 (Score Range: 0-8)",
                    "essay_column": "student_response",
                    "score_column": "Score",
                    "prompt_column": "Prompt",
                    "score_range": (0, 8),
                },
            }

            # Special handling for ASAP-AES
            if dataset_name == "ASAP-AES" or dataset_name == "D_ASAP-AES":
                print(f"   🔧 ASAP-AES detected - using Set1 as default")
                config_key = "ASAP-AES_Set1"

            elif config_name:
                combined_name = f"{dataset_name}_{config_name}"
                if combined_name in manual_configs:
                    config_key = combined_name
                elif f"{base_dataset_name}_{config_name}" in manual_configs:
                    config_key = f"{base_dataset_name}_{config_name}"

            if not config_key:
                if dataset_name in manual_configs:
                    config_key = dataset_name
                elif base_dataset_name in manual_configs:
                    config_key = base_dataset_name

            print(f"DEBUG: config_key found: {config_key}")

            if config_key:
                print(f"   🔧 Using manual config for {config_key}")
                base_config = manual_configs[config_key].copy()

                return {
                    "huggingface_id": dataset_id,
                    "config": config_name,
                    "split": "train",
                    "auto_discovered": False,
                    "manual_override": True,
                    "discovery_time": datetime.now().isoformat(),
                    **base_config,
                }

            # Continue with auto-configuration for other datasets
            print(f"   🔧 Starting auto-configuration for {dataset_name} (config: {config_name})...")

            split_needed = self._get_dataset_split(dataset_name)

            try:
                if config_name:
                    dataset = load_dataset(
                        dataset_id, config_name, split=f"{split_needed}[:1]", token=self.hf_token, trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(dataset_id, split=f"{split_needed}[:1]", token=self.hf_token, trust_remote_code=True)
            except TypeError:
                try:
                    if config_name:
                        dataset = load_dataset(
                            dataset_id, config_name, split=f"{split_needed}[:1]", use_auth_token=self.hf_token
                        )
                    else:
                        dataset = load_dataset(dataset_id, split=f"{split_needed}[:1]", use_auth_token=self.hf_token)
                except Exception as e:
                    print(f"     ❌ Failed to load {dataset_id} with config {config_name}: {e}")
                    return None
            except Exception as e:
                print(f"     ❌ Failed to load {dataset_id}: {e}")
                return None

            if len(dataset) == 0:
                print(f"     ❌ No data in {split_needed} split")
                return None

            sample = dataset[0]
            columns = list(sample.keys())

            print(f"     📋 Available columns: {columns}")

            essay_col = self._detect_essay_column(columns, sample)
            score_col = self._detect_score_column(columns, sample)
            prompt_col = self._detect_prompt_column(columns, sample)

            if not essay_col or not score_col:
                print(f"     ❌ Could not detect required columns")
                return None

            score_range = self._detect_score_range(dataset, score_col)
            description = self._generate_enhanced_description(dataset_name, config_name, columns, sample)

            print(f"     📊 Detected - Essay: {essay_col}, Score: {score_col}, Range: {score_range}")

            return {
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
                "dataset_variant": config_name,
            }

        except Exception as e:
            print(f"     ❌ Error configuring {dataset_name} with config {config_name}: {e}")
            return None

    def _override_dataset_config(self, dataset_name: str, existing_config: Dict[str, Any]) -> Dict[str, Any]:
        manual_overrides = {
            "ASAP-AES": {
                "description": "ASAP-AES Essay Set 1: Persuasive Essays (Score Range: 2-12)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 1,
                "score_range": (2, 12),
            },
            "D_ASAP-AES": {
                "description": "ASAP-AES Essay Set 1: Persuasive Essays (Score Range: 2-12)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 1,
                "score_range": (2, 12),
            },
            "ASAP-AES_Set1": {
                "description": "ASAP-AES Essay Set 1: Persuasive Essays (Score Range: 2-12)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 1,
                "score_range": (2, 12),
            },
            "D_ASAP-AES_Set1": {
                "description": "ASAP-AES Essay Set 1: Persuasive Essays (Score Range: 2-12)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 1,
                "score_range": (2, 12),
            },
            "ASAP-AES_Set2_Domain1": {
                "description": "ASAP-AES Essay Set 2 Domain 1: Content (Score Range: 1-6)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 2,
                "score_range": (1, 6),
            },
            "D_ASAP-AES_Set2_Domain1": {
                "description": "ASAP-AES Essay Set 2 Domain 1: Content (Score Range: 1-6)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 2,
                "score_range": (1, 6),
            },
            "ASAP-AES_Set2_Domain2": {
                "description": "ASAP-AES Essay Set 2 Domain 2: Organization (Score Range: 1-4)",
                "essay_column": "essay",
                "score_column": "domain2_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 2,
                "score_range": (1, 4),
            },
            "D_ASAP-AES_Set2_Domain2": {
                "description": "ASAP-AES Essay Set 2 Domain 2: Organization (Score Range: 1-4)",
                "essay_column": "essay",
                "score_column": "domain2_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 2,
                "score_range": (1, 4),
            },
            "ASAP-AES_Set3": {
                "description": "ASAP-AES Essay Set 3: Source Dependent Responses (Score Range: 0-3)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 3,
                "score_range": (0, 3),
            },
            "D_ASAP-AES_Set3": {
                "description": "ASAP-AES Essay Set 3: Source Dependent Responses (Score Range: 0-3)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 3,
                "score_range": (0, 3),
            },
            "ASAP-AES_Set4": {
                "description": "ASAP-AES Essay Set 4: Source Dependent Responses (Score Range: 0-3)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 4,
                "score_range": (0, 3),
            },
            "D_ASAP-AES_Set4": {
                "description": "ASAP-AES Essay Set 4: Source Dependent Responses (Score Range: 0-3)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 4,
                "score_range": (0, 3),
            },
            "ASAP-AES_Set5": {
                "description": "ASAP-AES Essay Set 5: Source Dependent Responses (Score Range: 0-4)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 5,
                "score_range": (0, 4),
            },
            "D_ASAP-AES_Set5": {
                "description": "ASAP-AES Essay Set 5: Source Dependent Responses (Score Range: 0-4)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 5,
                "score_range": (0, 4),
            },
            "ASAP-AES_Set6": {
                "description": "ASAP-AES Essay Set 6: Source Dependent Responses (Score Range: 0-4)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 6,
                "score_range": (0, 4),
            },
            "D_ASAP-AES_Set6": {
                "description": "ASAP-AES Essay Set 6: Source Dependent Responses (Score Range: 0-4)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 6,
                "score_range": (0, 4),
            },
            "ASAP-AES_Set7": {
                "description": "ASAP-AES Essay Set 7: Narrative Essays (Score Range: 0-30)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 7,
                "score_range": (0, 30),
            },
            "D_ASAP-AES_Set7": {
                "description": "ASAP-AES Essay Set 7: Narrative Essays (Score Range: 0-30)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 7,
                "score_range": (0, 30),
            },
            "ASAP-AES_Set8": {
                "description": "ASAP-AES Essay Set 8: Narrative Essays (Score Range: 0-60)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 8,
                "score_range": (0, 60),
            },
            "D_ASAP-AES_Set8": {
                "description": "ASAP-AES Essay Set 8: Narrative Essays (Score Range: 0-60)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "essay_set_filter": 8,
                "score_range": (0, 60),
            },
        }

        if dataset_name in manual_overrides:
            print(f"   🔧 Applying manual override for {dataset_name}")
            override_config = manual_overrides[dataset_name].copy()
            updated_config = existing_config.copy()
            updated_config.update(override_config)
            return updated_config

        return existing_config

        if dataset_name in manual_overrides:
            print(f"   🔧 Applying manual override for {dataset_name}")
            override_config = manual_overrides[dataset_name].copy()
            updated_config = existing_config.copy()
            updated_config.update(override_config)
            return updated_config

        return existing_config

    def load_dataset_sample(
        self, dataset_id: str, config: Optional[str] = None, split: str = "train", sample_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Load a sample from a HF dataset with config support"""

        print(f"📥 Loading sample from {dataset_id} (config={config}, split={split})")

        try:
            # Special handling for q1 - force reload without cache
            if config == "q1":
                print(f"🔧 Special handling for q1 - bypassing cache")
                try:
                    ds = load_dataset(
                        dataset_id,
                        name=config,
                        split=split,
                        cache_dir=None,  # Bypass cache
                        download_mode="force_redownload",
                        token=self.hf_token,
                        trust_remote_code=True,
                    )
                except TypeError:
                    # Fallback for older versions
                    ds = load_dataset(dataset_id, name=config, split=split, use_auth_token=self.hf_token)
            else:
                # Normal loading for other configs
                try:
                    if config:
                        ds = load_dataset(dataset_id, config, split=split, token=self.hf_token, trust_remote_code=True)
                    else:
                        ds = load_dataset(dataset_id, split=split, token=self.hf_token, trust_remote_code=True)
                except TypeError:
                    # Fallback for older versions
                    if config:
                        ds = load_dataset(dataset_id, config, split=split, use_auth_token=self.hf_token)
                    else:
                        ds = load_dataset(dataset_id, split=split, use_auth_token=self.hf_token)

        except Exception as e:
            print(f"❌ load_dataset failed for {dataset_id}: {e}")
            # Try API fallback first
            api_result = self._load_via_api(dataset_id, config, split, sample_size)
            if api_result:
                return api_result
            # If API also fails, return empty list
            return []

        # Sample rows
        if len(ds) > sample_size:
            idx = random.sample(range(len(ds)), sample_size)
            ds = ds.select(idx)

        essays: List[Dict[str, Any]] = [{"row": r} for r in ds]
        print(f"✅ Loaded {len(essays)} rows from {dataset_id}")
        return essays

    def _get_dataset_split(self, dataset_name: str) -> str:
        no_validation_datasets = ["BEEtlE", "SciEntSBank"]

        test_only_keywords = []
        dataset_lower = dataset_name.lower()

        for keyword in no_validation_datasets:
            if keyword.lower() in dataset_lower:
                return "train"

        if any(keyword in dataset_lower for keyword in test_only_keywords):
            return "test"

        return "train"

    def _generate_enhanced_description(self, dataset_name: str, config_name: Optional[str], columns: list, sample: dict) -> str:
        """Generate enhanced description including config info"""

        # Base descriptions
        base_descriptions = {
            "beetle": "Basic Elements of English Teaching and Learning Evaluation",
            "scientsbank": "Science Entailment Bank",
            "asap": "Automated Student Assessment Prize",
            "rice": "Rice University Chemistry Dataset",
            "csee": "Computer Science Essay Evaluation",
            "persuade": "Persuasive Essays Dataset",
            "grade": "Grade Like a Human Dataset",
            "mohlar": "Automatic Short Answer Grading - Mohlar Dataset",
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
            if config_name == "2way":
                base_desc += " (2-way classification)"
            elif config_name == "3way":
                base_desc += " (3-way classification)"
            elif config_name.startswith("q") or config_name.startswith("Q"):
                base_desc += f" (Question {config_name.upper()})"
            else:
                base_desc += f" ({config_name} configuration)"

        return base_desc

    def _detect_essay_column(self, columns: list, sample: dict) -> str:
        """Auto-detect essay/text column using smart heuristics"""

        # Priority keywords for essay columns
        essay_keywords = [
            "full_text",
            "essay_text",
            "essay",
            "text",
            "content",
            "response",
            "student_answer",
            "answer",
            "writing",
            "student_response",
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
            "holistic_score",
            "domain1_score",
            "score",
            "grade",
            "rating",
            "overall_score",
            "total_score",
            "final_score",
            "label",
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
            "question_text",
            "prompt_name",
            "prompt",
            "question",
            "task",
            "essay_set",
            "set",
            "prompt_text",
            "writing_prompt",
            "assignment",
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

    def _cache_discovered_datasets(self, datasets: Dict[str, Dict[str, Any]]):
        """Cache discovered datasets for performance"""
        try:
            cache_file = "dataset_discovery_cache.json"
            cache_data = {
                "discovered_at": datetime.now().isoformat(),
                "username": self.username,
                "dataset_count": len(datasets),
                "datasets": datasets,
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            print(f"💾 Cached {len(datasets)} discovered datasets")

        except Exception as e:
            print(f"⚠️ Failed to cache discoveries: {e}")

    def _get_fallback_datasets(self) -> Dict[str, Dict[str, Any]]:
        print("Using fallback static dataset configuration with D_ separation")

        return {
            # UNLABELED DATASETS (D_ prefix) - for researcher download
            "D_ASAP-AES": {
                "huggingface_id": "nlpatunt/D_ASAP-AES",
                "description": "ASAP-AES Test Data (Unlabeled)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "config": None,
                "split": "test",
                "score_range": (0, 60),
                "auto_discovered": False,
                "dataset_type": "unlabeled",
            },
            "D_BEEtlE_2way": {
                "huggingface_id": "nlpatunt/D_BEEtlE",
                "description": "BEEtlE 2-way Test Data (Unlabeled)",
                "essay_column": "student_answer",
                "score_column": "label",
                "prompt_column": "question_text",
                "config": "2way",
                "split": "test",
                "score_range": (0, 1),
                "auto_discovered": False,
                "dataset_type": "unlabeled",
            },
            # LABELED DATASETS (no D_ prefix) - for ground truth evaluation (private)
            "ASAP-AES": {
                "huggingface_id": "nlpatunt/ASAP-AES",
                "description": "ASAP-AES Ground Truth (Labeled)",
                "essay_column": "essay",
                "score_column": "domain1_score",
                "prompt_column": "essay_set",
                "config": None,
                "split": "test",
                "score_range": (0, 60),
                "auto_discovered": False,
                "dataset_type": "ground_truth",
            },
            "BEEtlE_2way": {
                "huggingface_id": "nlpatunt/BEEtlE",
                "description": "BEEtlE 2-way Ground Truth (Labeled)",
                "essay_column": "student_answer",
                "score_column": "label",
                "prompt_column": "question_text",
                "config": "2way",
                "split": "test",
                "score_range": (0, 1),
                "auto_discovered": False,
                "dataset_type": "ground_truth",
            },
        }

    def _load_via_api(
        self, dataset_id: str, config: str = None, split: str = "train", sample_size: int = 100
    ) -> List[Dict[str, Any]]:
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
                "length": min(sample_size, 100),
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


class SGRADESDatasetManager:
    def __init__(self):
        self.hf_loader = HuggingFaceDatasetLoader()

        try:
            from app.services.cache_service import DatasetCache

            self.cache = DatasetCache()
        except ImportError:
            self.cache = None

        print("🚀 Initializing dynamic dataset discovery with config support...")
        self.datasets_config = self.hf_loader.get_configured_datasets()
        print(f"📊 Initialized with {len(self.datasets_config)} dataset configurations")

        # Print discovered datasets
        for name, config in self.datasets_config.items():
            status = "auto-discovered" if config.get("auto_discovered") else "static"
            config_info = f" (config: {config.get('config')})" if config.get("config") else ""
            print(f"  - {name}{config_info} [{status}]")

    def load_dataset_for_evaluation(self, dataset_name: str, sample_size: int = 50) -> List[Dict[str, Any]]:

        if self.cache:
            cache_key = f"{dataset_name}_{sample_size}"
            cached_data = self.cache.get_dataset(cache_key)
            if cached_data:
                print(f"💾 Using cached data for {dataset_name}")
                return cached_data

        if dataset_name not in self.datasets_config:
            print(f"❌ Unknown dataset: {dataset_name}")
            return []

        config = self.datasets_config[dataset_name]
        print(f"🔐 Loading dataset: {dataset_name} ({'auto-discovered' if config.get('auto_discovered') else 'static'})")

        try:
            # Load from HuggingFace dataset with config support
            raw_data = self.hf_loader.load_dataset_sample(
                dataset_id=config["huggingface_id"],
                config=config.get("config"),  # Pass the config (q1, q2, etc.)
                split=config["split"],
                sample_size=sample_size,
            )

            if not raw_data:
                print(f"⚠️ No data loaded for {dataset_name}")
                return []

            # Standardize the data structure
            standardized_essays = []

            for i, item in enumerate(raw_data):
                row = item.get("row", {})

                if config.get("essay_set_filter"):
                    essay_set = self._get_column_value(row, ["essay_set"])
                    if str(essay_set) != str(config["essay_set_filter"]):
                        continue

                # Extract values using configured column names
                essay_text = self._get_column_value(row, [config["essay_column"]])
                score = self._get_column_value(row, [config["score_column"]])
                prompt = self._get_column_value(row, [config.get("prompt_column", "prompt")])
                essay_id = self._get_column_value(row, ["essay_id", "id", "sis_id"])

                # Use fallback ID if not found
                if not essay_id:
                    essay_id = f"{dataset_name}_{i}"

                # Use fallback prompt if not found
                if not prompt:
                    prompt = f"Default prompt for {dataset_name}"

                if essay_text and score is not None:
                    standardized_essays.append(
                        {
                            "essay_id": str(essay_id),
                            "essay_text": str(essay_text),
                            "prompt": str(prompt),
                            "score": float(score),
                            "dataset_name": dataset_name,
                            "config": config.get("config"),  # Track which config this came from
                            "score_range": config["score_range"],
                        }
                    )

            print(f"✅ Standardized {len(standardized_essays)} essays from {dataset_name}")

            # Cache the result
            if self.cache and standardized_essays:
                self.cache.set_dataset(cache_key, standardized_essays)
                print(f"💾 Cached {len(standardized_essays)} essays for {dataset_name}")

            return standardized_essays

        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            return [self.get_sample_essay(dataset_name)]

    def load_ground_truth_scores(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load ground truth scores for evaluation with config support"""

        # Check cache first
        if self.cache:
            cache_key = f"ground_truth_{dataset_name}"
            cached_ground_truth = self.cache.get_dataset(cache_key)
            if cached_ground_truth:
                print(f"💾 Using cached ground truth for {dataset_name}")
                return cached_ground_truth

        if dataset_name not in self.datasets_config:
            print(f"❌ Unknown dataset for ground truth: {dataset_name}")
            return []

        config = self.datasets_config[dataset_name]

        try:
            # Load test split with human scores using config
            raw_data = self.hf_loader.load_dataset_sample(
                dataset_id=config["huggingface_id"],
                config=config.get("config"),  # Use the same config as training data
                split="test",  # Use test split for evaluation
                sample_size=999999,
            )

            ground_truth = []
            for item in raw_data:
                row = item.get("row", {})

                essay_id = self._get_column_value(row, ["essay_id", "id", "sis_id"])
                human_score = self._get_column_value(row, [config["score_column"]])

                if essay_id and human_score is not None:
                    ground_truth.append(
                        {
                            "essay_id": str(essay_id),
                            "human_score": float(human_score),
                            "score_range": config["score_range"],
                            "config": config.get("config"),
                        }
                    )

            print(f"✅ Loaded ground truth for {dataset_name}: {len(ground_truth)} essays")

            # Cache the ground truth
            if self.cache and ground_truth:
                self.cache.set_dataset(cache_key, ground_truth)
                print(f"💾 Cached ground truth for {dataset_name}")

            return ground_truth

        except Exception as e:
            print(f"❌ Error loading ground truth for {dataset_name}: {e}")
            return []

    def _get_column_value(self, row: Dict[str, Any], possible_columns: List[str]) -> Any:
        """Get value from row using possible column names"""
        for col in possible_columns:
            if col and col in row and row[col] is not None:
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
            "score": 3.5,
            "score_range": (1, 5),
            "config": None,
            "metadata": {"type": "fallback", "reason": "dataset_load_failed"},
        }

    def get_dataset_configs(self) -> Dict[str, Any]:
        """Get information about all available dataset configurations"""
        config_info = {}

        for dataset_name, config in self.datasets_config.items():
            config_info[dataset_name] = {
                "huggingface_id": config["huggingface_id"],
                "config": config.get("config"),
                "description": config["description"],
                "essay_column": config["essay_column"],
                "score_column": config["score_column"],
                "score_range": config["score_range"],
                "auto_discovered": config.get("auto_discovered", False),
                "available_columns": config.get("columns_detected", []),
            }

        return config_info

    def refresh_datasets(self) -> int:
        """Refresh dataset discovery from HuggingFace"""
        print("🔄 Refreshing dataset discovery...")
        self.datasets_config = self.hf_loader.get_configured_datasets()
        return len(self.datasets_config)


# Upload validation configurations for each dataset
EXPECTED_UPLOAD_FORMATS = {
    "ASAP-AES": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 60),
        "essay_id_pattern": r"^ASAP-AES_\d+$",
        "score_type": "float",
    },
    "ASAP2": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 60),
        "essay_id_pattern": r"^ASAP2_\d+$",
        "score_type": "float",
    },
    "ASAP-SAS": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 60),
        "essay_id_pattern": r"^ASAP-SAS_\d+$",
        "score_type": "float",
    },
    "ASAP_plus_plus": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 60),
        "essay_id_pattern": r"^ASAP_plus_plus_\d+$",
        "score_type": "float",
    },
    "CSEE": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 100),
        "essay_id_pattern": r"^CSEE_\d+$",
        "score_type": "float",
    },
    "BEEtlE_2way": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 1),
        "essay_id_pattern": r"^BEEtlE_2way_\d+$",
        "score_type": "int",
        "valid_values": [0, 1],
    },
    "BEEtlE_3way": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 2),
        "essay_id_pattern": r"^BEEtlE_3way_\d+$",
        "score_type": "int",
        "valid_values": [0, 1, 2],
    },
    "SciEntSBank_2way": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 1),
        "essay_id_pattern": r"^SciEntSBank_2way_\d+$",
        "score_type": "int",
        "valid_values": [0, 1],
    },
    "SciEntSBank_3way": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 2),
        "essay_id_pattern": r"^SciEntSBank_3way_\d+$",
        "score_type": "int",
        "valid_values": [0, 1, 2],
    },
    "Mohlar": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 5),
        "essay_id_pattern": r"^Mohlar_\d+$",
        "score_type": "float",
    },
    "Ielts_Writing_Dataset": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (1, 9),
        "essay_id_pattern": r"^Ielts_Writing_Dataset_\d+$",
        "score_type": "float",
    },
    "Ielts_Writing_Task_2_Dataset": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (1, 9),
        "essay_id_pattern": r"^Ielts_Writing_Task_2_Dataset_\d+$",
        "score_type": "float",
    },
    "persuade_2": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (1, 6),
        "essay_id_pattern": r"^persuade_2_\d+$",
        "score_type": "float",
    },
    "Regrading_Dataset_J2C": {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 100),
        "essay_id_pattern": r"^Regrading_Dataset_J2C_\d+$",
        "score_type": "float",
    },
}

# Add grade_like_a_human configs for q1-q6
for q in ["q1", "q2", "q3", "q4", "q5"]:
    EXPECTED_UPLOAD_FORMATS[f"OS_Dataset_{q}"] = {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 100),
        "essay_id_pattern": f"^OS_Dataset_{q}_\\d+$",
        "score_type": "float",
    }

# Add Rice_Chem configs for Q1-Q4
for Q in ["Q1", "Q2", "Q3", "Q4"]:
    EXPECTED_UPLOAD_FORMATS[f"Rice_Chem_{Q}"] = {
        "required_columns": ["essay_id", "predicted_score"],
        "score_range": (0, 100),
        "essay_id_pattern": f"^Rice_Chem_{Q}_\\d+$",
        "score_type": "float",
    }


# Global instance with dynamic discovery and config support
dataset_manager = SGRADESDatasetManager()
