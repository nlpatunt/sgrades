# app/api/routes/output_submissions.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import Response
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import os
import zipfile
import tempfile
import hashlib
import math
import gzip
from datetime import datetime
from pathlib import Path
import statistics
import numpy as np
import json
from app.models.database import OutputSubmission, Dataset, EvaluationResult
from app.models.pydantic_models import DatasetFormatResponse, AvailableDatasetsResponse, SubmissionResponse, TestSubmissionResponse, SubmissionsStatsResponse, BatchSubmissionResponse
from app.services.database_service import get_database
from app.services.csv_security_validator import CSVSecurityValidator
from app.services.file_storage import LocalFileStorage
from slowapi import Limiter
from fastapi import Request
from slowapi.util import get_remote_address

try:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score
    from scipy.stats import pearsonr
    import requests
    REAL_EVALUATION_AVAILABLE = True
except ImportError as e:
    REAL_EVALUATION_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    from scipy.stats import pearsonr

limiter = Limiter(key_func=get_remote_address)
file_storage = LocalFileStorage()

def download_ground_truth_private(dataset_name: str) -> Dict[str, Any]:
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
    print(f"🔍 DEBUG: Original dataset request: {dataset_name}")
    print(f"🔍 DEBUG: Normalized for ground truth: {normalized_name}")
    print(f"🔍 DEBUG: Attempting to load dataset: {normalized_name}")
    
    if not HF_DATASETS_AVAILABLE:
        return {"status": "error", "error": "HuggingFace datasets library not available"}
    
    id_columns_map = {
        "ASAP-AES": "essay_id",
        "ASAP2": "essay_id", 
        "ASAP-SAS": "Id",
        "ASAP_plus_plus": "essay_id",
        "BEEtlE_2way": "ID",
        "BEEtlE_3way": "ID",
        "SciEntSBank_2way": "ID", 
        "SciEntSBank_3way": "ID",
        "CSEE": "index",
        "Mohlar": "ID",
        "Ielts_Writing_Dataset": "ID",
        "Ielst_Writing_Task_2_Dataset": "ID",
        "persuade_2": "essay_id_comp",
        "Regrading_Dataset_J2C": "ID",
        "grade_like_a_human_dataset_os_q1": "ID",
        "grade_like_a_human_dataset_os_q2": "ID",
        "grade_like_a_human_dataset_os_q3": "ID",
        "grade_like_a_human_dataset_os_q4": "ID",
        "grade_like_a_human_dataset_os_q5": "ID",
        "Rice_Chem_Q1": "sis_id",
        "Rice_Chem_Q2": "sis_id",
        "Rice_Chem_Q3": "sis_id", 
        "Rice_Chem_Q4": "sis_id"
    }
        
    try:
        print(f"📥 Loading private ground truth dataset: nlpatunt/{normalized_name}")
       
        if normalized_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]:
            import pandas as pd
            import requests
            from io import StringIO
            
            if "BEEtlE" in normalized_name:
                suffix = "2way" if "2way" in normalized_name else "3way"
                urls = [
                    f"https://huggingface.co/datasets/nlpatunt/BEEtlE/resolve/main/test_{suffix}.csv",
                    f"https://huggingface.co/datasets/nlpatunt/BEEtlE/raw/main/test_{suffix}.csv"
                ]
            else:  # SciEntSBank
                suffix = "2way" if "2way" in normalized_name else "3way"
                urls = [
                    f"https://huggingface.co/datasets/nlpatunt/SciEntSBank/resolve/main/test_{suffix}.csv",
                    f"https://huggingface.co/datasets/nlpatunt/SciEntSBank/raw/main/test_{suffix}.csv"
                ]

            import os
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                try:
                    with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                        hf_token = f.read().strip()
                except:
                    hf_token = None
                    
            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            for url in urls:
                try:
                    print(f"Attempting direct CSV download from: {url}")
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    
                    df = pd.read_csv(StringIO(response.text))
                    print(f"✅ Loaded {normalized_name} via direct download: {len(df)} rows, columns: {list(df.columns)}")
                    
                    # Ensure proper ID column handling
                    id_column = id_columns_map.get(normalized_name, "ID")
                    if id_column in df.columns:
                        print(f"✅ ID column '{id_column}' found with {len(df)} rows")
                    
                    return {
                        "status": "success",
                        "dataset": df,
                        "rows": len(df),
                        "columns": list(df.columns)
                    }
                    
                except Exception as url_error:
                    print(f"Failed to download from {url}: {url_error}")
                    continue
            
            # If all URLs failed, return error
            return {"status": "error", "error": f"All download URLs failed for {normalized_name}"}
        
        # Standard HuggingFace dataset loading for other datasets
        elif normalized_name in ["Rice_Chem_Q1", "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"]:
            q_num = normalized_name.split("_")[-1]
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files=f"{q_num}/test.csv")
            dataset = dataset["train"]
        elif normalized_name.startswith("grade_like_a_human_dataset_os_q"):
            q_num = normalized_name.split("_q")[-1]
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", 
                                name=f"q{q_num}", 
                                split="test",
                                trust_remote_code=True)
        elif normalized_name == "persuade_2":
            dataset = load_dataset("nlpatunt/persuade_2", data_files="test.csv")
            dataset = dataset["train"] 
        elif normalized_name == "Mohlar":
            dataset = load_dataset("nlpatunt/Mohlar", data_files="test.csv")
            dataset = dataset["train"]
        else:
            try:
                dataset = load_dataset(f"nlpatunt/{normalized_name}", split="test", trust_remote_code=True)
            except:
                dataset = load_dataset(f"nlpatunt/{normalized_name}")
                if hasattr(dataset, 'keys'):
                    first_split = list(dataset.keys())[0]
                    dataset = dataset[first_split]
        
        # Convert to pandas for non-direct download cases
        if 'dataset' in locals():
            df = dataset.to_pandas()
        else:
            return {"status": "error", "error": f"No dataset loaded for {normalized_name}"}
        
        # Handle missing ID columns
        if normalized_name.startswith("grade_like_a_human_dataset_os") and "ID" not in df.columns:
            df["ID"] = range(1, len(df) + 1)
            print(f"Added ID column to {normalized_name}")

        # Clean up dataframe
        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"DEBUG: Dropped empty columns: {columns_to_drop}")

        id_column = id_columns_map.get(normalized_name, "ID")
        
        print(f"DEBUG: After pandas conversion - {id_column} column: {df[id_column].head(10).tolist()}")
        print(f"DEBUG: {id_column} column dtype: {df[id_column].dtype}")

        # Fix ID column if it was replaced with index
        if id_column in df.columns and (df[id_column] == df.index).all():
            print("ERROR: ID column was replaced with DataFrame index!")
            df[id_column] = df.index + 1
            print(f"FIXED: {id_column} column now: {df[id_column].head(10).tolist()}")
            
        print(f"✅ Loaded private dataset: {len(df)} rows, columns: {list(df.columns)}")
        print(f"🎯 Ground truth for {dataset_name} loaded from {normalized_name} repository")
        
        return {
            "status": "success",
            "dataset": df,
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        print(f"All loading methods failed: {str(e)}")
        return {"status": "error", "error": str(e)}

def create_rice_chem_validators():
    return {
        "Rice_Chem_Q1": RiceChemValidator("Q1"),
        "Rice_Chem_Q2": RiceChemValidator("Q2"),
        "Rice_Chem_Q3": RiceChemValidator("Q3"),
        "Rice_Chem_Q4": RiceChemValidator("Q4")
    }

def create_grade_like_human_validators():
    return {
        "grade_like_a_human_dataset_os_q1": GradeLikeHumanValidator("1"),
        "grade_like_a_human_dataset_os_q2": GradeLikeHumanValidator("2"),
        "grade_like_a_human_dataset_os_q3": GradeLikeHumanValidator("3"),
        "grade_like_a_human_dataset_os_q4": GradeLikeHumanValidator("4"),
        "grade_like_a_human_dataset_os_q5": GradeLikeHumanValidator("5")
    }

class BaseValidator:
    def __init__(self, required_columns, primary_score_column, id_column=None, valid_labels=None):
        self.required_columns = required_columns
        self.primary_score_column = primary_score_column
        self.id_column = id_column or required_columns[0]
        self.valid_labels = valid_labels
    
    def clean_labels_with_fallback(self, df, handle_invalid='discard', testing_mode=False):
        if not self.valid_labels or self.primary_score_column not in df.columns:
            return df, []
        
        warnings = []
        df_clean = df.copy()
        original_count = len(df_clean)
        
        missing_mask = df_clean[self.primary_score_column].isna() | \
                      (df_clean[self.primary_score_column] == '') | \
                      (df_clean[self.primary_score_column].astype(str).str.strip() == '')
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            if handle_invalid == 'assign_fallback' or testing_mode:
                df_clean.loc[missing_mask, self.primary_score_column] = '5'
                warnings.append(f"Assigned fallback value 5 to {missing_count} missing labels")
            else:
                df_clean = df_clean[~missing_mask]
                warnings.append(f"Removed {missing_count} rows with missing labels")
        
        df_clean[self.primary_score_column] = df_clean[self.primary_score_column].astype(str).str.strip()

        label_mapping = {}
        for valid_label in self.valid_labels:
            update_dict = {
                valid_label.lower(): valid_label,
                valid_label.upper(): valid_label,
                valid_label.capitalize(): valid_label,
                valid_label: valid_label
            }
            label_mapping.update(update_dict)
        
        if 'correct' in self.valid_labels:
            label_mapping.update({
                '1': 'correct', 'true': 'correct', 'yes': 'correct', 'right': 'correct'
            })
        if 'incorrect' in self.valid_labels:
            label_mapping.update({
                '0': 'incorrect', 'false': 'incorrect', 'no': 'incorrect', 'wrong': 'incorrect'
            })
        if 'contradictory' in self.valid_labels:
            label_mapping.update({
                'contradictory': 'contradictory'  
            })
        
        df_clean['_original_label'] = df_clean[self.primary_score_column].copy()
        df_clean[self.primary_score_column] = df_clean[self.primary_score_column].str.lower().map(label_mapping).fillna(df_clean[self.primary_score_column])
        
        valid_mask = df_clean[self.primary_score_column].isin(self.valid_labels)
        invalid_mask = ~valid_mask
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            invalid_examples = df_clean[invalid_mask]['_original_label'].unique()[:5].tolist()
            
            if handle_invalid == 'assign_fallback' or testing_mode:
                df_clean.loc[invalid_mask, self.primary_score_column] = '4'
                warnings.append(f"Assigned fallback value 4 to {invalid_count} invalid labels. Examples: {invalid_examples}")
            else:
                df_clean = df_clean[valid_mask]
                warnings.append(f"Removed {invalid_count} rows with invalid labels. Examples: {invalid_examples}")
        
        if '_original_label' in df_clean.columns:
            df_clean = df_clean.drop('_original_label', axis=1)
        
        success_rate = len(df_clean) / original_count if original_count > 0 else 0
        if success_rate < 0.5 and handle_invalid != 'assign_fallback':
            warnings.append(f"WARNING: Only {success_rate:.1%} of rows are valid")
        
        return df_clean, warnings
    
    def validate(self, df, testing_mode=False):
        errors = []
        warnings = []
   
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {sorted(list(missing_cols))}")
        
        if self.id_column in df.columns:
            duplicate_ids = df[df[self.id_column].duplicated()]
            if len(duplicate_ids) > 0:
                errors.append(f"Found {len(duplicate_ids)} duplicate {self.id_column} values - IDs must be unique")
        
        if self.primary_score_column not in df.columns:
            errors.append(f"{self.primary_score_column} column is required for evaluation")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "primary_score_column": self.primary_score_column
            }
        
        df_clean = df.copy()
        
        if self.valid_labels:
            handle_mode = 'assign_fallback' if testing_mode else 'discard'
            
            df_clean, label_warnings = self.clean_labels_with_fallback(df_clean, handle_mode, testing_mode)
            warnings.extend(label_warnings)
            
            if len(df_clean) == 0:
                errors.append("No valid rows remaining after label validation")
            elif len(df_clean) / len(df) < 0.5 and not testing_mode:
                errors.append(f"Too many invalid labels: only {len(df_clean)}/{len(df)} rows are valid")
            
            if not testing_mode:
                valid_labels_found = df_clean[self.primary_score_column].isin(self.valid_labels + ['4', '5']).any()
                if not valid_labels_found:
                    errors.append(f"No valid labels found. Expected: {self.valid_labels}")
        
        else:
            validator_class = self.__class__.__name__
            if validator_class in ["IELTSWritingValidator", "IELTSTask2Validator"]:
                df_clean[self.primary_score_column] = df_clean[self.primary_score_column].astype(str)
               
                less_than_pattern = df_clean[self.primary_score_column].str.match(r'^<(\d+\.?\d*)$')
                if less_than_pattern.any():
                    converted_values = df_clean.loc[less_than_pattern, self.primary_score_column].str.extract(r'^<(\d+\.?\d*)$')[0].astype(float) - 0.5
                    df_clean.loc[less_than_pattern, self.primary_score_column] = converted_values
                    warnings.append(f"Converted {less_than_pattern.sum()} '<x' values to x-0.5 format (e.g., <4 -> 3.5)")
                
                greater_than_pattern = df_clean[self.primary_score_column].str.match(r'^>(\d+\.?\d*)$')
                if greater_than_pattern.any():
                    converted_values = df_clean.loc[greater_than_pattern, self.primary_score_column].str.extract(r'^>(\d+\.?\d*)$')[0].astype(float) + 0.5
                    df_clean.loc[greater_than_pattern, self.primary_score_column] = converted_values
                    warnings.append(f"Converted {greater_than_pattern.sum()} '>x' values to x+0.5 format (e.g., >8 -> 8.5)")
       
            if validator_class == "MohlarValidator":
                df_clean[self.primary_score_column] = df_clean[self.primary_score_column].astype(str).str.strip()
                
                numeric_mask = df_clean[self.primary_score_column].str.match(r'^-?\d+\.?\d*$')
                non_numeric_count = (~numeric_mask).sum()
                
                if non_numeric_count > 0:
                    df_clean = df_clean[numeric_mask]
                    warnings.append(f"Discarded {non_numeric_count} rows with non-numeric grades")
                
                df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')
            
            missing_scores = df_clean[self.primary_score_column].isna().sum()
            if missing_scores > 0:
                warnings.append(f"{self.primary_score_column} has {missing_scores} missing values")
            
            # Check numeric conversion
            valid_scores = df_clean[self.primary_score_column].dropna()
            if len(valid_scores) > 0:
                numeric_scores = pd.to_numeric(valid_scores, errors='coerce')
                non_numeric_count = numeric_scores.isna().sum()
                if non_numeric_count > 0:
                    errors.append(f"{self.primary_score_column} has {non_numeric_count} non-numeric values")
                else:
                    # Update the column with numeric values
                    df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "primary_score_column": self.primary_score_column,
            "cleaned_df": df_clean
        }

class ASAPAESValidator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id", "domain1_score"], "domain1_score", "essay_id")

class ASAP2Validator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id", "score"], "score", "essay_id")

class ASAPSASValidator(BaseValidator):
    def __init__(self):
        super().__init__(["Id", "Score1"], "Score1", "Id")

class ASAPPlusPlusValidator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id", "overall_score"], "overall_score", "essay_id")

class CSEEValidator(BaseValidator):
    def __init__(self):
        super().__init__(["index", "overall_score"], "overall_score", "index")

class Persuade2Validator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id_comp", "holistic_essay_score"], "holistic_essay_score", "essay_id_comp")

class RiceChemValidator(BaseValidator):
    def __init__(self, question_number):
        super().__init__(["sis_id", "Score"], "Score", "sis_id")
        self.question_number = question_number

class MohlarValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "grade"], "grade", "ID")

class IELTSWritingValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "Overall_Score"], "Overall_Score", "ID")

class IELTSTask2Validator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "band_score"], "band_score", "ID")

class RegradingDatasetJ2CValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "grade"], "grade", "ID")

class GradeLikeHumanValidator(BaseValidator):
    def __init__(self, question_number):
        super().__init__(["ID", "score_1"], "score_1", "ID")
        self.question_number = question_number

class BEEtlE2WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect"])

class BEEtlE3WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect", "contradictory"])

class SciEntSBank2WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect"])

class SciEntSBank3WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect", "contradictory"])

class RealEvaluationEngine:
    def __init__(self):
        self.ground_truth_cache = {}

        rice_chem_validators = create_rice_chem_validators()
        grade_like_human_validators = create_grade_like_human_validators()

        # First create the base validators
        self.validators = {
            "ASAP-AES": ASAPAESValidator(),
            "D_ASAP-AES": ASAPAESValidator(),
            "ASAP-AES_Set1": ASAPAESValidator(),
            "D_ASAP-AES_Set1": ASAPAESValidator(),
            "ASAP-AES_Set2_Domain1": ASAPAESValidator(),
            "D_ASAP-AES_Set2_Domain1": ASAPAESValidator(),
            "ASAP-AES_Set2_Domain2": ASAPAESValidator(),
            "D_ASAP-AES_Set2_Domain2": ASAPAESValidator(),
            "ASAP-AES_Set3": ASAPAESValidator(),
            "D_ASAP-AES_Set3": ASAPAESValidator(),
            "ASAP-AES_Set4": ASAPAESValidator(),
            "D_ASAP-AES_Set4": ASAPAESValidator(),
            "ASAP-AES_Set5": ASAPAESValidator(),
            "D_ASAP-AES_Set5": ASAPAESValidator(),
            "ASAP-AES_Set6": ASAPAESValidator(),
            "D_ASAP-AES_Set6": ASAPAESValidator(),
            "ASAP-AES_Set7": ASAPAESValidator(),
            "D_ASAP-AES_Set7": ASAPAESValidator(),
            "ASAP-AES_Set8": ASAPAESValidator(),
            "D_ASAP-AES_Set8": ASAPAESValidator(),
            "ASAP-SAS": ASAPSASValidator(),
            "D_ASAP-SAS": ASAPSASValidator(),
            "ASAP2": ASAP2Validator(),
            "D_ASAP2": ASAP2Validator(),
            "ASAP_plus_plus": ASAPPlusPlusValidator(),
            "D_ASAP_plus_plus": ASAPPlusPlusValidator(),
            "BEEtlE_2way": BEEtlE2WayValidator(),
            "D_BEEtlE_2way": BEEtlE2WayValidator(),
            "BEEtlE_3way": BEEtlE3WayValidator(),
            "D_BEEtlE_3way": BEEtlE3WayValidator(),
            "SciEntSBank_2way": SciEntSBank2WayValidator(),
            "D_SciEntSBank_2way": SciEntSBank2WayValidator(),
            "SciEntSBank_3way": SciEntSBank3WayValidator(),
            "D_SciEntSBank_3way": SciEntSBank3WayValidator(),
            "Mohlar": MohlarValidator(),
            "D_Mohlar": MohlarValidator(),
            "CSEE": CSEEValidator(),
            "D_CSEE": CSEEValidator(),
            "persuade_2": Persuade2Validator(),
            "D_persuade_2": Persuade2Validator(),
            "Regrading_Dataset_J2C": RegradingDatasetJ2CValidator(),
            "D_Regrading_Dataset_J2C": RegradingDatasetJ2CValidator(),
            "Ielts_Writing_Dataset": IELTSWritingValidator(),
            "D_Ielts_Writing_Dataset": IELTSWritingValidator(),
            "Ielst_Writing_Task_2_Dataset": IELTSTask2Validator(),
            "D_Ielst_Writing_Task_2_Dataset": IELTSTask2Validator(),
            **rice_chem_validators,
            **grade_like_human_validators,
        }

        # Then add D_ versions for grade_like_a_human datasets
        for q in ["q1", "q2", "q3", "q4", "q5"]:
            base_name = f"grade_like_a_human_dataset_os_{q}"
            d_name = f"D_grade_like_a_human_dataset_os_{q}"
            if base_name in self.validators:
                self.validators[d_name] = self.validators[base_name]
    
        # Add D_ versions for Rice_Chem validators
        for Q in ["Q1", "Q2", "Q3", "Q4"]:
            base_name = f"Rice_Chem_{Q}"
            d_name = f"D_Rice_Chem_{Q}"
            if base_name in self.validators:
                self.validators[d_name] = self.validators[base_name]
                
        self.SUBMISSION_REQUIREMENTS = {
            "ASAP-AES": ["essay_id", "domain1_score"],
            "ASAP2": ["essay_id", "score"], 
            "ASAP-SAS": ["Id", "Score1"],
            "ASAP_plus_plus": ["essay_id", "overall_score"],
            "CSEE": ["index", "overall_score"],
            "persuade_2": ["essay_id_comp", "holistic_essay_score"],
            "Rice_Chem_Q1": ["sis_id", "Score"],
            "Rice_Chem_Q2": ["sis_id", "Score"],
            "Rice_Chem_Q3": ["sis_id", "Score"],
            "Rice_Chem_Q4": ["sis_id", "Score"],
            "BEEtlE_2way": ["ID", "label"],
            "BEEtlE_3way": ["ID", "label"],
            "SciEntSBank_2way": ["ID", "label"],
            "SciEntSBank_3way": ["ID", "label"],
            "Mohlar": ["ID", "grade"],
            "Ielts_Writing_Dataset": ["ID", "Overall_Score"],
            "Ielst_Writing_Task_2_Dataset": ["ID", "band_score"],
            "Regrading_Dataset_J2C": ["ID", "grade"],
            "grade_like_a_human_dataset_os_q1": ["ID", "score_1"],
            "grade_like_a_human_dataset_os_q2": ["ID", "score_1"],
            "grade_like_a_human_dataset_os_q3": ["ID", "score_1"],
            "grade_like_a_human_dataset_os_q4": ["ID", "score_1"],
            "grade_like_a_human_dataset_os_q5": ["ID", "score_1"],
        }

        # Score column mapping
        self.SCORE_COLUMNS = {
            "ASAP-AES": "domain1_score",
            "ASAP2": "score",
            "ASAP-SAS": "Score1", 
            "ASAP_plus_plus": "overall_score",
            "BEEtlE_2way": "label",
            "BEEtlE_3way": "label",     
            "SciEntSBank_2way": "label",
            "SciEntSBank_3way": "label",
            "CSEE": "overall_score",
            "Mohlar": "grade",
            "Ielts_Writing_Dataset": "Overall_Score",
            "Ielst_Writing_Task_2_Dataset": "band_score",
            "persuade_2": "holistic_essay_score",
            "Regrading_Dataset_J2C": "grade",
            "grade_like_a_human_dataset_os_q1": "score_1",
            "grade_like_a_human_dataset_os_q2": "score_1",
            "grade_like_a_human_dataset_os_q3": "score_1", 
            "grade_like_a_human_dataset_os_q4": "score_1",
            "grade_like_a_human_dataset_os_q5": "score_1",
            "Rice_Chem_Q1": "Score",
            "Rice_Chem_Q2": "Score",
            "Rice_Chem_Q3": "Score",
            "Rice_Chem_Q4": "Score"
        }
        
        # ID columns for matching - use original column names
        self.ID_COLUMNS = {
            "ASAP-AES": "essay_id",
            "ASAP2": "essay_id", 
            "ASAP-SAS": "Id",
            "ASAP_plus_plus": "essay_id",
            "BEEtlE_2way": "ID",
            "BEEtlE_3way": "ID",
            "SciEntSBank_2way": "ID", 
            "SciEntSBank_3way": "ID",
            "CSEE": "index",
            "Mohlar": "ID",
            "Ielts_Writing_Dataset": "ID",
            "Ielst_Writing_Task_2_Dataset": "ID",
            "persuade_2": "essay_id_comp",
            "Regrading_Dataset_J2C": "ID",
            "grade_like_a_human_dataset_os_q1": "ID",
            "grade_like_a_human_dataset_os_q2": "ID",
            "grade_like_a_human_dataset_os_q3": "ID",
            "grade_like_a_human_dataset_os_q4": "ID",
            "grade_like_a_human_dataset_os_q5": "ID",
            "Rice_Chem_Q1": "sis_id",
            "Rice_Chem_Q2": "sis_id",
            "Rice_Chem_Q3": "sis_id", 
            "Rice_Chem_Q4": "sis_id"
        }

    def get_ground_truth(self, dataset_name: str) -> Dict[str, Any]:
        """Get ground truth data for a dataset (with caching)"""
        if dataset_name not in self.ground_truth_cache:
            print(f"Downloading ground truth for {dataset_name}")
            result = download_ground_truth_private(dataset_name)
            if result["status"] == "success":
                self.ground_truth_cache[dataset_name] = result["dataset"]
                print(f"Ground truth cached: {len(result['dataset'])} rows")
            return result
        else:
            return {"status": "success", "dataset": self.ground_truth_cache[dataset_name]}
    
    def get_score_column(self, dataset_name: str) -> str:
        return self.SCORE_COLUMNS.get(dataset_name, "score")
    
    def get_id_column(self, dataset_name: str) -> str:
        return self.ID_COLUMNS.get(dataset_name, "ID")

    def validate_full_structure(self, dataset_name: str, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate submission structure using dataset-specific validator"""
        try:
            print(f"Starting validation for {dataset_name}")
            
            if dataset_name in self.validators:
                validator = self.validators[dataset_name]
                validation_result = validator.validate(prediction_df)
                
                if not validation_result["valid"]:
                    return {
                        "valid": False,
                        "errors": validation_result["errors"],
                        "warnings": validation_result["warnings"],
                        "expected_columns": validator.required_columns,
                        "instruction": f"Fix validation errors for {dataset_name}"
                    }
                
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": validation_result["warnings"],
                    "score_column": validation_result["primary_score_column"],
                    "id_column": "ID",
                    "instruction": f"Validation passed for {dataset_name}"
                }
            else:
                return {
                    "valid": False,
                    "errors": [f"No validator found for dataset: {dataset_name}"],
                    "warnings": [],
                    "instruction": f"Dataset {dataset_name} is not supported"
                }
                
        except Exception as e:
            print(f"Validation error for {dataset_name}: {e}")
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "exception": True
            }

    def validate_submission_format(self, dataset_name: str, predictions_df: pd.DataFrame, testing_mode: bool = False) -> Dict[str, Any]:
        """Validate submission format using ground truth structure"""
        try:
            # Normalize dataset name - remove D_ prefix for validator lookup
            normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
            
            print(f"Validating submission for {dataset_name} using {normalized_name} validator")
            
            gt_result = self.get_ground_truth(dataset_name)
            if gt_result["status"] != "success":
                return {
                    "valid": False,
                    "errors": [f"Cannot load dataset schema: {gt_result.get('error')}"],
                    "warnings": []
                }
                        
            ground_truth_df = gt_result["dataset"]
            
            # ADD DEBUG CODE HERE
            if dataset_name == "D_BEEtlE_3way":
                print(f"DEBUG: Processing BEEtlE_3way validation")
                print(f"DEBUG: predictions_df shape: {predictions_df.shape}")
                print(f"DEBUG: predictions_df columns: {list(predictions_df.columns)}")
                print(f"DEBUG: predictions_df dtypes: {predictions_df.dtypes}")
                print(f"DEBUG: Sample prediction data:")
                print(predictions_df.head(2))
                print(f"DEBUG: ground_truth_df shape: {ground_truth_df.shape}")
                print(f"DEBUG: ground_truth_df columns: {list(ground_truth_df.columns)}")
            
            # Use normalized name to find validator
            if dataset_name in self.validators:
                validator = self.validators[normalized_name]
                validation_result = validator.validate(predictions_df, testing_mode=testing_mode)
                                
                return {
                    "valid": validation_result["valid"],
                    "errors": validation_result.get("errors", []),
                    "warnings": validation_result.get("warnings", []),
                    "expected_columns": validator.required_columns,
                    "score_column": self.get_score_column(normalized_name),
                    "matching_method": "id_based",
                    "cleaned_df": validation_result.get("cleaned_df", predictions_df)
                }
            else:
                return {
                    "valid": False,
                    "errors": [f"No validator found for dataset: {normalized_name}"],
                    "warnings": []
                }
                        
        except Exception as e:
            print(f"Validation error for {dataset_name}: {e}")
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }
        
    def match_predictions_to_ground_truth(self, dataset_name: str, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
        normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name

        score_col = self.get_score_column(normalized_name)
        id_col = self.get_id_column(normalized_name)
        
        print(f"Using ID-based matching for {dataset_name} with column '{id_col}'")
        
        # Debug: Show input data structure
        print(f"DEBUG: Prediction df columns: {list(prediction_df.columns)}")
        print(f"DEBUG: Ground truth df columns: {list(ground_truth_df.columns)}")
        print(f"DEBUG: Looking for score column: '{score_col}' and ID column: '{id_col}'")
        
        # Debug: Show sample data before processing
        print(f"DEBUG: Prediction df sample (first 3 rows):")
        print(prediction_df.head(3))
        print(f"DEBUG: Ground truth df sample (first 3 rows):")
        print(ground_truth_df.head(3))
        
        # Ensure ID columns are strings for consistent matching
        prediction_df[id_col] = prediction_df[id_col].astype(str)
        ground_truth_df[id_col] = ground_truth_df[id_col].astype(str)

        # Debug: Show ID conversion results
        print(f"DEBUG: Prediction IDs after conversion: {prediction_df[id_col].head(5).tolist()}")
        print(f"DEBUG: Ground truth IDs after conversion: {ground_truth_df[id_col].head(5).tolist()}")

        # Remove duplicates
        prediction_df_unique = prediction_df.drop_duplicates(subset=[id_col], keep='first')
        ground_truth_df_unique = ground_truth_df.drop_duplicates(subset=[id_col], keep='first')

        print(f"DEBUG: After deduplication - Predictions: {len(prediction_df_unique)}, Ground truth: {len(ground_truth_df_unique)}")

        # Merge on the correct ID column
        merged_df = prediction_df_unique.merge(
            ground_truth_df_unique[[id_col, score_col]],
            on=id_col, how="inner", suffixes=("_pred", "_true")
        )
        
        print(f"DEBUG: Merged df shape: {merged_df.shape}")
        print(f"DEBUG: Merged df columns: {list(merged_df.columns)}")
        
        if len(merged_df) == 0:
            print(f"ERROR: No matching {id_col} found between predictions and ground truth")
            print(f"DEBUG: Prediction IDs sample: {prediction_df_unique[id_col].head(10).tolist()}")
            print(f"DEBUG: Ground truth IDs sample: {ground_truth_df_unique[id_col].head(10).tolist()}")
            return {"status": "error", "error": f"No matching {id_col} found between predictions and ground truth"}
        
        # Debug for all datasets, not just specific ones
        print(f"DEBUG: Merged data sample:")
        score_pred_col = f"{score_col}_pred"
        score_true_col = f"{score_col}_true"
        
        if score_pred_col in merged_df.columns and score_true_col in merged_df.columns:
            print(merged_df[[id_col, score_pred_col, score_true_col]].head(10))
        else:
            print(f"DEBUG: Expected columns {score_pred_col} and {score_true_col} not found")
            print(f"DEBUG: Available columns: {list(merged_df.columns)}")

        # Debug for BEEtlE/SciEntSBank
        classification_datasets = ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]
        
        if normalized_name in classification_datasets:
            print(f"DEBUG: Processing classification dataset: {normalized_name}")
            
            # Check for any mismatches
            if score_pred_col in merged_df.columns and score_true_col in merged_df.columns:
                mismatches = merged_df[merged_df[score_pred_col] != merged_df[score_true_col]]
                matches = merged_df[merged_df[score_pred_col] == merged_df[score_true_col]]
                print(f"DEBUG: Found {len(matches)} matches and {len(mismatches)} mismatches out of {len(merged_df)} total")
                
                if len(mismatches) > 0:
                    print(f"DEBUG: First 5 mismatches:")
                    print(mismatches[[id_col, score_pred_col, score_true_col]].head())
        
        # Extract scores based on dataset type
        if normalized_name in classification_datasets:
            # For classification, get raw string values then convert
            pred_scores = merged_df[score_pred_col].values
            gt_scores = merged_df[score_true_col].values
            valid_mask = ~(pd.Series(pred_scores).isna() | pd.Series(gt_scores).isna())
            
            print(f"DEBUG: Classification - Pred scores sample: {pred_scores[:5]}")
            print(f"DEBUG: Classification - GT scores sample: {gt_scores[:5]}")
            print(f"DEBUG: Valid mask sum: {valid_mask.sum()}")
        else:
            # For regression, convert to numeric
            pred_scores_numeric = pd.to_numeric(merged_df[score_pred_col], errors='coerce')
            gt_scores_numeric = pd.to_numeric(merged_df[score_true_col], errors='coerce')
            valid_mask = ~(pred_scores_numeric.isna() | gt_scores_numeric.isna())
            pred_scores = pred_scores_numeric[valid_mask].values
            gt_scores = gt_scores_numeric[valid_mask].values
            
            print(f"DEBUG: Regression - Pred scores sample: {pred_scores[:5]}")
            print(f"DEBUG: Regression - GT scores sample: {gt_scores[:5]}")

        return {
            "status": "success",
            "y_pred": pred_scores[valid_mask] if normalized_name in classification_datasets else pred_scores,
            "y_true": gt_scores[valid_mask] if normalized_name in classification_datasets else gt_scores,
            "matched_count": int(valid_mask.sum()),
            "total_predictions": len(prediction_df),
            "total_ground_truth": len(ground_truth_df),
            "matching_method": "id_based"
        }
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        try:
            from sklearn.metrics import (
                mean_absolute_error, mean_squared_error, f1_score,
                precision_score, recall_score, cohen_kappa_score, accuracy_score
            )
            from scipy.stats import pearsonr

            print(f"DEBUG: Starting metrics calculation with {len(y_true)} samples")
            
            # Check if we have categorical data (strings)
            is_categorical = isinstance(y_true[0], str) if len(y_true) > 0 else False
            
            if is_categorical:
                print(f"DEBUG: Categorical data detected - using classification metrics")
                
                # Convert categorical labels to numeric for calculation
                unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
                label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
                
                y_true_numeric = np.array([label_to_num[label] for label in y_true])
                y_pred_numeric = np.array([label_to_num[label] for label in y_pred])
                
                print(f"DEBUG: Label mapping: {label_to_num}")
                print(f"DEBUG: Converted y_true: {y_true_numeric}")
                print(f"DEBUG: Converted y_pred: {y_pred_numeric}")
                
                # Calculate classification metrics
                accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
                
                try:
                    qwk = cohen_kappa_score(y_true_numeric, y_pred_numeric, weights="quadratic")
                except:
                    qwk = cohen_kappa_score(y_true_numeric, y_pred_numeric)  # Regular kappa if quadratic fails
                    
                try:
                    f1 = f1_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
                    precision = precision_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
                    recall = recall_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
                except:
                    f1 = precision = recall = accuracy  # Fallback for binary case
                
                # For categorical data, correlation and MAE don't make sense, so set them based on accuracy
                correlation = accuracy  # Use accuracy as proxy for correlation
                mae = 1.0 - accuracy  # Error rate as proxy for MAE
                mse = (1.0 - accuracy) ** 2
                rmse = np.sqrt(mse)
                
            else:
                print(f"DEBUG: Numeric data detected - using regression metrics")
                # Handle single sample case
                if len(y_true) == 1:
                    perfect_match = abs(y_true[0] - y_pred[0]) < 1e-10
                    correlation = 1.0 if perfect_match else 0.0
                    qwk = 1.0 if perfect_match else 0.0
                    f1 = precision = recall = 1.0 if perfect_match else 0.0
                else:
                    correlation, p_value = pearsonr(y_true, y_pred)
                    try:
                        qwk = cohen_kappa_score(y_true.round(), y_pred.round(), weights="quadratic")
                    except:
                        qwk = 0.0
                    
                    try:
                        y_true_class = y_true.round().astype(int)
                        y_pred_class = y_pred.round().astype(int)
                        f1 = f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                        precision = precision_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                        recall = recall_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                        accuracy = accuracy_score(y_true_class, y_pred_class)  # Add this line
                    except:
                        f1 = precision = recall = accuracy = 0.0  # Add accuracy here

                    # Calculate numeric metrics
                    mae = mean_absolute_error(y_true, y_pred) if len(y_true) > 0 else 0.0
                    mse = mean_squared_error(y_true, y_pred) if len(y_true) > 0 else 0.0
                    rmse = np.sqrt(mse)

                    metrics = {
                        "quadratic_weighted_kappa": float(qwk) if not pd.isna(qwk) else 0.0,
                        "pearson_correlation": float(correlation) if not pd.isna(correlation) else 0.0,
                        "mean_absolute_error": float(mae) if not pd.isna(mae) else 0.0,
                        "mean_squared_error": float(mse) if not pd.isna(mse) else 0.0,
                        "root_mean_squared_error": float(rmse) if not pd.isna(rmse) else 0.0,
                        "f1_score": float(f1) if not pd.isna(f1) else 0.0,
                        "precision": float(precision) if not pd.isna(precision) else 0.0,
                        "recall": float(recall) if not pd.isna(recall) else 0.0,
                        "accuracy": float(accuracy) if not pd.isna(accuracy) else 0.0  # Add this line
                    }
            
            print(f"DEBUG: Final metrics: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"DEBUG: Metrics calculation failed with exception: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            raise e
        
    def evaluate_submission(self, dataset_name: str, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Main evaluation function"""
        # Initialize metrics with default values at the start
        metrics = {
            'quadratic_weighted_kappa': 0.0,
            'pearson_correlation': 0.0,
            'mean_absolute_error': 0.0,
            'mean_squared_error': 0.0,
            'root_mean_squared_error': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0
        }
        
        try:
            print(f"Starting evaluation for {dataset_name}")
            
            # Get ground truth
            gt_result = self.get_ground_truth(dataset_name)
            if gt_result["status"] != "success":
                return {
                    "status": "error",
                    "error": f"Failed to load ground truth: {gt_result.get('error')}"
                }
            ground_truth_df = gt_result["dataset"]
            normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
        
            # Validate structure
            validation_result = self.validate_full_structure(dataset_name, predictions_df, ground_truth_df)
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "error": "Structure validation failed",
                    "validation_details": validation_result
                }
            
            # Match predictions to ground truth
            matching_result = self.match_predictions_to_ground_truth(dataset_name, predictions_df, ground_truth_df)
            if matching_result["status"] != "success":
                return {
                    "status": "error", 
                    "error": "Matching failed",
                    "matching_details": matching_result
                }
            
            y_pred = matching_result["y_pred"]
            y_true = matching_result["y_true"]
            
            if len(y_pred) == 0 or len(y_true) == 0:
                return {
                    "status": "error",
                    "error": "No valid score pairs found for evaluation"
                }

          
            # Classification label conversion
            if normalized_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]:
                
                # Define label mappings with variations for robustness
                if "3way" in normalized_name:
                    label_map = {
                        'correct': 2, 'Correct': 2, 'CORRECT': 2,
                        'incorrect': 0, 'Incorrect': 0, 'INCORRECT': 0,
                        'contradictory': 1, 'Contradictory': 1, 'CONTRADICTORY': 1,
                        '2': 2, '1': 1, '0': 0, 2: 2, 1: 1, 0: 0
                    }
                else:  # 2way
                    label_map = {
                        'correct': 1, 'Correct': 1, 'CORRECT': 1,
                        'incorrect': 0, 'Incorrect': 0, 'INCORRECT': 0,
                        '1': 1, '0': 0, 1: 1, 0: 0,
                        'True': 1, 'False': 0, 'true': 1, 'false': 0
                    }
               
                if len(y_pred) > 0 and isinstance(y_pred[0], str):
                    y_pred_series = pd.Series(y_pred).str.strip()  
                    y_pred = y_pred_series.map(label_map).fillna(-999).astype(int).to_numpy()
                    if -999 in y_pred:
                        unmapped = y_pred_series[y_pred == -999].unique()
                        print(f"WARNING: Unmapped prediction labels: {unmapped}")
                    y_pred = np.where(y_pred == -999, 0, y_pred) 
                    
                if len(y_true) > 0 and isinstance(y_true[0], str):
                    y_true_series = pd.Series(y_true).str.strip() 
                    y_true = y_true_series.map(label_map).fillna(-999).astype(int).to_numpy()
                    if -999 in y_true:
                        unmapped = y_true_series[y_true == -999].unique()
                        print(f"WARNING: Unmapped ground truth labels: {unmapped}")
                    y_true = np.where(y_true == -999, 0, y_true) 

            # Calculate metrics with proper error handling
            try:
                calculated_metrics = self.calculate_metrics(y_true, y_pred)
                if calculated_metrics and isinstance(calculated_metrics, dict):
                    metrics.update(calculated_metrics)
                    print(f"DEBUG: Metrics calculation successful: {metrics}")
                else:
                    print("DEBUG: Metrics calculation returned empty or invalid result")
            except Exception as metrics_error:
                print(f"WARNING: Metrics calculation failed: {metrics_error}")
                print("DEBUG: Using default metrics values")
                # metrics already initialized with defaults at the top

            id_column = self.get_id_column(dataset_name)
        
            if normalized_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]:
                unique_pred = list(set(y_pred)) if len(y_pred) > 0 else []
                unique_true = list(set(y_true)) if len(y_true) > 0 else []
                score_range_pred = unique_pred
                score_range_true = unique_true
            else:
                score_range_pred = [float(np.nanmin(y_pred)), float(np.nanmax(y_pred))] if len(y_pred) > 0 else [0, 0]
                score_range_true = [float(np.nanmin(y_true)), float(np.nanmax(y_true))] if len(y_true) > 0 else [0, 0]

            return {
                "status": "success",
                "metrics": metrics,
                "evaluation_details": {
                    "dataset": dataset_name,
                    "matched_examples": int(len(y_pred)),
                    "total_predictions": int(matching_result["total_predictions"]),
                    "total_ground_truth": int(matching_result["total_ground_truth"]),
                    "matching_method": "id_based",
                    "score_column": self.get_score_column(dataset_name),
                    "score_range_pred": score_range_pred,
                    "score_range_true": score_range_true,
                    "id_column": id_column,
                    "validation_warnings": validation_result.get("warnings", [])
                }
            }
            
        except Exception as e:
            print(f"Evaluation failed for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "dataset": dataset_name,
                "metrics": metrics  # Now metrics is always defined
            }

real_evaluation_engine = RealEvaluationEngine()

def get_all_submissions_from_db():
    try:
        db = get_database()
        submissions = db.query(OutputSubmission).all()
        
        result = []
        for sub in submissions:
            evaluation_data = {}
            if sub.evaluation_result:
                try:
                    evaluation_data = json.loads(sub.evaluation_result)
                except:
                    pass
            
            submission_dict = {
                "id": sub.id,
                "model_name": sub.submitter_name,
                "dataset_name": sub.dataset_name,
                "contact_email": sub.submitter_email,
                "institution": "",
                "description": sub.description or "",
                "has_real_evaluation": "real_evaluation" in evaluation_data,
                "metrics": evaluation_data.get("real_evaluation", {}).get("metrics", {})
            }
            result.append(submission_dict)
        
        return result
    except Exception as e:
        print(f"Database error: {e}")
        return []

def store_submission_in_db(dataset_name, model_name, metrics, description="", 
                          institution="", contact_email="", filename="",
                          file_content: bytes = None, ip_address="", user_agent=""):
    try:
        db = get_database()
        
        # Store the actual file
        if file_content:
            storage_info = file_storage.store_file(
                file_content, dataset_name, contact_email, filename
            )
        else:
            raise ValueError("File content is required for storage")
        
        submission = OutputSubmission(
            dataset_name=dataset_name,
            submitter_name=model_name,
            submitter_email=contact_email,
            original_filename=filename,
            stored_file_path=storage_info["stored_file_path"],
            file_hash=storage_info["file_hash"],
            file_size=storage_info["file_size"],
            upload_timestamp=storage_info["upload_timestamp"],
            ip_address=ip_address,
            user_agent=user_agent,
            status="completed",
            description=description
        )
        
        evaluation_data = {
            "real_evaluation": {
                "status": "success",
                "metrics": metrics,
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "file_hash_at_evaluation": storage_info["file_hash"]
            }
        }
        
        submission.evaluation_result = json.dumps(evaluation_data)
        
        db.add(submission)
        db.commit()
        db.refresh(submission)
        
        return {"status": "success", "submission_id": submission.id}
        
    except Exception as e:
        print(f"Database storage failed: {e}")
        return {"status": "error", "error": str(e)}

def clean_for_json(obj):
    """Convert object to JSON-safe format"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def clean_dataframe_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for processing"""
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    return df_clean

def decode_file_content(content: bytes) -> tuple:
    """Decode file content with multiple encoding attempts"""
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None

router = APIRouter()

@router.get("/format/{dataset_name}", response_model=DatasetFormatResponse)
async def get_dataset_format(dataset_name: str):
    """Get specific format requirements for a dataset"""
    try:
        if dataset_name not in real_evaluation_engine.validators:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
        
        required_columns = real_evaluation_engine.SUBMISSION_REQUIREMENTS.get(dataset_name, [])
        score_column = real_evaluation_engine.get_score_column(dataset_name)
        
        return clean_for_json({
            "dataset_name": dataset_name,
            "required_columns": required_columns,
            "score_column": score_column,
            "id_column": real_evaluation_engine.get_id_column(dataset_name),
            "matching_method": "id_based"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/available-datasets", response_model=AvailableDatasetsResponse)
async def get_available_datasets():
    try:
        datasets = list(real_evaluation_engine.validators.keys())
        return clean_for_json({
            "datasets": datasets,
            "total": len(datasets)
        })
    except Exception as e:
        return clean_for_json({
            "error": str(e),
            "datasets": []
        })

@router.post("/upload-single", response_model=SubmissionResponse)
@limiter.limit("5/minute")
async def upload_single_submission(
    request: Request, 
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    model_name: Optional[str] = Form(None),
    description: str = Form("Individual dataset test"),
    institution: str = Form(""),
    contact_email: str = Form("")
):
    validator = CSVSecurityValidator()
    
    if model_name is None:
        model_name = f"test_model_{dataset_name}"
    
    try:
        content = await file.read()
        csv_content, encoding_used = decode_file_content(content)
        
        if csv_content is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid file encoding. Please save your CSV as UTF-8."
            )
        
        # SECURITY VALIDATION
        security_validation = validator.validate_csv_content(csv_content)
        if not security_validation["valid"]:
            return clean_for_json({
                "success": False,
                "error": f"Security validation failed: {security_validation['error']}",
                "dataset": dataset_name,
                "filename": file.filename
            })
        
        # Sanitize filename
        safe_filename = validator.sanitize_filename(file.filename or "upload.csv")
        
        df = pd.read_csv(io.StringIO(csv_content))
        df_clean = clean_dataframe_safe(df)
        
        validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df_clean, testing_mode=False)
        
        if not validation_result["valid"]:
            return clean_for_json({
                "success": False,
                "validation_errors": validation_result["errors"],
                "validation_warnings": validation_result.get("warnings", []),
                "dataset": dataset_name,
                "filename": safe_filename
            })
        
        evaluation_result = real_evaluation_engine.evaluate_submission(
            dataset_name=dataset_name,
            predictions_df=df_clean
        )
        
        if evaluation_result["status"] != "success":
            return clean_for_json({
                "success": False,
                "evaluation_error": evaluation_result.get("error"),
                "dataset": dataset_name,
                "filename": safe_filename
            })
        
        db_result = store_submission_in_db(
            dataset_name=dataset_name,
            model_name=model_name,
            metrics=evaluation_result["metrics"],
            description=description,
            institution=institution,
            contact_email=contact_email,
            filename=safe_filename,
            file_content=content,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", "")
        )
        
        return clean_for_json({
            "success": True,
            "evaluation": evaluation_result,
            "database": db_result,
            "dataset": dataset_name,
            "model_name": model_name,
            "filename": safe_filename,
            "encoding_used": encoding_used,
            "security_info": f"File validated: {security_validation.get('info', 'Security check passed')}",
            "note": "REAL EVALUATION: Metrics calculated from actual ground truth comparisons"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        return clean_for_json({
            "success": False,
            "error": str(e),
            "dataset": dataset_name,
            "filename": validator.sanitize_filename(file.filename) if file else "unknown"
        })
    
@router.post("/upload-batch", response_model=BatchSubmissionResponse)
@limiter.limit("2/minute")
async def upload_batch_submissions(
    request: Request,
    files: List[UploadFile] = File(...),
    dataset_names: List[str] = Form(...),
    model_name: str = Form(...),
    description: str = Form(""),
    institution: str = Form(""),
    contact_email: str = Form("")
):
    validator = CSVSecurityValidator()
    
    print(f"DEBUG: Received batch upload with {len(files)} files")
    print(f"DEBUG: Dataset names: {dataset_names}")
    print(f"DEBUG: Model name: {model_name}")
    results = []
    successful_uploads = 0
    failed_uploads = 0

    if len(dataset_names) != len(files):
        return clean_for_json({
            "success": False,
            "error": f"Dataset names length ({len(dataset_names)}) does not match files length ({len(files)})"
        })

    for idx, file in enumerate(files):
        try:
            dataset_name = dataset_names[idx].strip()
           
            content = await file.read()
            csv_content = None
            encoding_used = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    csv_content = content.decode(encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
                        
            if csv_content is None:
                results.append({
                    "filename": file.filename,
                    "dataset": dataset_name,
                    "success": False,
                    "error": "Invalid file encoding"
                })
                failed_uploads += 1
                continue

            # SECURITY VALIDATION
            security_validation = validator.validate_csv_content(csv_content)
            if not security_validation["valid"]:
                results.append({
                    "filename": file.filename,
                    "dataset": dataset_name,
                    "success": False,
                    "error": f"Security validation failed: {security_validation['error']}"
                })
                failed_uploads += 1
                continue
            
            # Sanitize filename
            safe_filename = validator.sanitize_filename(file.filename or "upload.csv")

            df = pd.read_csv(io.StringIO(csv_content))
            df_clean = clean_dataframe_safe(df)
            
            validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df_clean)
            
            if not validation_result["valid"]:
                results.append({
                    "filename": safe_filename,
                    "dataset": dataset_name,
                    "success": False,
                    "validation_errors": validation_result["errors"],
                    "validation_warnings": validation_result.get("warnings", [])
                })
                failed_uploads += 1
                continue

            evaluation_result = real_evaluation_engine.evaluate_submission(
                dataset_name=dataset_name,
                predictions_df=df_clean
            )

            if evaluation_result.get("status") != "success":
                results.append({
                    "filename": safe_filename,
                    "dataset": dataset_name,
                    "success": False,
                    "evaluation_error": evaluation_result.get("error", "Evaluation failed")
                })
                failed_uploads += 1
                continue

            db_result = store_submission_in_db(
                dataset_name=dataset_name,
                model_name=model_name,
                metrics=evaluation_result["metrics"],
                description=description,
                institution=institution,
                contact_email=contact_email,
                filename=safe_filename,
                file_content=content,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "")
            )

            results.append({
                "filename": safe_filename,
                "dataset": dataset_name,
                "success": True,
                "evaluation": evaluation_result,
                "database": db_result,
                "encoding_used": encoding_used,
                "security_info": f"File validated: {security_validation.get('info', 'Security check passed')}"
            })
            successful_uploads += 1

        except Exception as e:
            safe_filename = validator.sanitize_filename(getattr(file, "filename", "unknown"))
            results.append({
                "filename": safe_filename,
                "dataset": dataset_name if "dataset_name" in locals() else "unknown",
                "success": False,
                "error": str(e)
            })
            failed_uploads += 1

    return clean_for_json({
        "success": successful_uploads > 0,
        "total_files": len(files),
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads,
        "model_name": model_name,
        "results": results,
        "note": "REAL EVALUATION: All metrics calculated from actual ground truth comparisons"
    })

def normalize_dataset_name_for_evaluation(dataset_name: str) -> str:
    if dataset_name.startswith("D_"):
        return dataset_name[2:] 
    return dataset_name

@router.post("/test-single-dataset")
async def test_single_dataset(
    request:Request,
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    model_name: str = Form(...),
    submitter_name: str = Form(...),
    submitter_email: str = Form(...),
    description: str = Form(...)
):
    validator = CSVSecurityValidator()
    try:
        content = await file.read()
        csv_content, encoding_used = decode_file_content(content)
        
        if csv_content is None:
            return clean_for_json({
                "success": False,
                "error": "Invalid file encoding. Please save your CSV as UTF-8.",
                "testing_mode": True
            })
        
        df = pd.read_csv(io.StringIO(csv_content))
        df_clean = clean_dataframe_safe(df)
        
        validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df_clean, testing_mode=True)
        
        if not validation_result["valid"]:
            return clean_for_json({
                "success": False,
                "validation_errors": validation_result["errors"],
                "validation_warnings": validation_result.get("warnings", []),
                "dataset": dataset_name,
                "filename": file.filename,
                "testing_mode": True
            })
        
        df_cleaned = validation_result.get("cleaned_df", df_clean)
        
        evaluation_result = real_evaluation_engine.evaluate_submission(
            dataset_name=dataset_name,
            predictions_df=df_cleaned
        )
        
        if evaluation_result["status"] != "success":
            return clean_for_json({
                "success": False,
                "evaluation_error": evaluation_result.get("error"),
                "dataset": dataset_name,
                "filename": file.filename,
                "testing_mode": True
            })
        
        # ADD THIS SECTION TO STORE THE SUBMISSION
        try:
            db_result = store_submission_in_db(
                dataset_name=dataset_name,
                model_name=f"Zero_Shot_Model_{dataset_name}",
                metrics=evaluation_result.get("metrics", {}),
                description=f"Zero-shot evaluation on {dataset_name}",
                institution="",
                contact_email="test@zeroshot.com",
                filename=file.filename,
                file_content=content,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "")
            )
            
            if db_result.get("status") == "success":
                print(f"✓ Submission stored in database with ID: {db_result.get('submission_id')}")
            else:
                print(f"✗ Failed to store submission: {db_result.get('error')}")
                
        except Exception as storage_error:
            print(f"✗ Database storage failed: {storage_error}")
            # Continue anyway - evaluation was successful
        
        response_data = {
            "success": True,
            "testing_mode": False,  # Change to False since we're storing it
            "dataset": dataset_name,
            "filename": file.filename,
            "evaluation": evaluation_result,
            "metrics": evaluation_result.get("metrics", {}),
            "encoding_used": encoding_used,
            "validation_warnings": validation_result.get("warnings", []),
        }
        
        return clean_for_json(response_data)
        
    except Exception as e:
        return clean_for_json({
            "success": False,
            "error": str(e),
            "dataset": dataset_name,
            "filename": file.filename if file else "unknown",
            "testing_mode": True
        })
    
@router.get("/leaderboard", response_model=Dict[str, Any]) 
async def get_leaderboard(
    dataset: str = "All Datasets",
    metric: str = "avg_quadratic_weighted_kappa",
    limit: int = 20,
    min_datasets: int = 21 
):
    try:
        if min_datasets is None:
            available_datasets = list(real_evaluation_engine.validators.keys())
            d_datasets = [ds for ds in available_datasets if ds.startswith('D_')]
            min_datasets = len(d_datasets)
            
        current_time = datetime.now().isoformat()
        submissions = get_all_submissions_from_db()
    
        real_submissions = [s for s in submissions if s.get('has_real_evaluation', False)]
        
        if not real_submissions:
            return clean_for_json({
                "dataset": dataset,
                "metric": metric,
                "total_entries": 0,
                "last_updated": current_time,
                "rankings": [],
                "summary_stats": {},
                "note": "No complete benchmarks found"
            })

        model_aggregates = {}
        for submission in real_submissions:
            model_name = submission['model_name']
            
            if model_name not in model_aggregates:
                model_aggregates[model_name] = {
                    'datasets': [],
                    'metrics': {},
                    'total_submissions': 0,
                    'institution': submission.get('institution', ''),
                    'description': submission.get('description', ''),
                    'contact_email': submission.get('contact_email', '')
                }
            
            model_aggregates[model_name]['datasets'].append(submission['dataset_name'])
            model_aggregates[model_name]['total_submissions'] += 1
            
            for metric_name, value in submission.get('metrics', {}).items():
                if metric_name not in model_aggregates[model_name]['metrics']:
                    model_aggregates[model_name]['metrics'][metric_name] = []
                model_aggregates[model_name]['metrics'][metric_name].append(value)

        complete_models = {}
        for model_name, data in model_aggregates.items():
            unique_datasets = list(set(data['datasets']))
            
            if len(unique_datasets) >= min_datasets:
                complete_models[model_name] = data
                complete_models[model_name]['unique_datasets_count'] = len(unique_datasets)
        
        if not complete_models:
            return clean_for_json({
                "dataset": dataset,
                "metric": metric,
                "total_entries": 0,
                "last_updated": current_time,
                "rankings": [],
                "summary_stats": {},
                "note": f"No complete benchmarks found. Models must test on at least {min_datasets} datasets to appear on leaderboard."
            })
        
        rankings = []
        for model_name, data in complete_models.items():
            # Calculate averages for the 8 specific metrics only
            avg_metrics = {}
            for metric_name, values in data['metrics'].items():
                if values:
                    avg_metrics[f"avg_{metric_name}"] = statistics.mean(values)
            
            # Include only the 8 specified metrics
            ranking_entry = {
                "model_name": model_name,
                "institution": data['institution'],
                "description": data['description'],
                "contact_email": data['contact_email'],
                "datasets_evaluated": list(set(data['datasets'])),
                "unique_datasets_count": data['unique_datasets_count'],
                "total_submissions": data['total_submissions'],
                "complete_benchmark": True,
                
                # Only the 8 specified metrics
                "avg_quadratic_weighted_kappa": avg_metrics.get("avg_quadratic_weighted_kappa", 0),
                "avg_pearson_correlation": avg_metrics.get("avg_pearson_correlation", 0),
                "avg_mean_absolute_error": avg_metrics.get("avg_mean_absolute_error", 0),
                "avg_root_mean_squared_error": avg_metrics.get("avg_root_mean_squared_error", 0),
                "avg_f1_score": avg_metrics.get("avg_f1_score", 0),
                "avg_precision": avg_metrics.get("avg_precision", 0),
                "avg_recall": avg_metrics.get("avg_recall", 0),
                "avg_accuracy": avg_metrics.get("avg_accuracy_within_1.0", avg_metrics.get("avg_accuracy", 0))
            }
            
            rankings.append(ranking_entry)
        
        # Sort by the selected metric
        if metric in ["avg_mean_absolute_error", "avg_root_mean_squared_error"]:
            rankings.sort(key=lambda x: x.get(metric, float('inf')))  # Lower is better
        else:
            rankings.sort(key=lambda x: x.get(metric, 0), reverse=True)  # Higher is better
        
        rankings = rankings[:limit]
        
        # Calculate summary stats for the 8 metrics only
        summary_stats = {}
        if rankings:
            metric_keys = [
                'avg_quadratic_weighted_kappa', 'avg_pearson_correlation', 'avg_mean_absolute_error',
                'avg_root_mean_squared_error', 'avg_f1_score', 'avg_precision', 'avg_recall', 'avg_accuracy'
            ]
            
            for metric_key in metric_keys:
                values = [r.get(metric_key, 0) for r in rankings if r.get(metric_key) is not None]
                if values:
                    summary_stats[metric_key] = {
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values)
                    }
            
            summary_stats['total_researchers'] = len(rankings)
            summary_stats['complete_benchmarks'] = len([r for r in rankings if r['complete_benchmark']])
        
        return clean_for_json({
            "dataset": dataset,
            "metric": metric,
            "total_entries": len(rankings),
            "last_updated": current_time,
            "rankings": rankings,
            "summary_stats": summary_stats,
            "available_metrics": metric_keys,
            "note": f"Complete benchmarks with {min_datasets}+ datasets showing 8 core metrics"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate leaderboard: {str(e)}")
    
@router.get("/stats", response_model=SubmissionsStatsResponse)
async def get_platform_stats():
    """Get platform statistics"""
    try:
        submissions = get_all_submissions_from_db()
        real_submissions = [s for s in submissions if s.get('has_real_evaluation', False)]
        
        unique_researchers = len(set(s.get('contact_email', '') for s in real_submissions if s.get('contact_email')))
        
        model_datasets = {}
        for submission in real_submissions:
            model = submission['model_name']
            if model not in model_datasets:
                model_datasets[model] = set()
            model_datasets[model].add(submission['dataset_name'])
        
        complete_benchmarks = sum(1 for datasets in model_datasets.values() if len(datasets) >= 20)
        
        return clean_for_json({
            "total_submissions": len(real_submissions),
            "total_researchers": unique_researchers,
            "complete_benchmarks": complete_benchmarks,
            "available_datasets": 24,
            "evaluation_type": "REAL_GROUND_TRUTH",
            "matching_method": "ID_based"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
    
@router.get("/admin/audit/{submission_id}")
async def get_submission_audit(submission_id: int, admin_key: str = ""):
    """Get complete audit trail for a submission"""
    if admin_key != os.getenv("ADMIN_AUDIT_KEY"):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        db = get_database()
        submission = db.query(OutputSubmission).filter(
            OutputSubmission.id == submission_id
        ).first()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Verify file integrity
        file_integrity_ok = file_storage.verify_file_integrity(
            submission.stored_file_path, submission.file_hash
        )
        
        return {
            "submission_id": submission.id,
            "dataset_name": submission.dataset_name,
            "submitter_name": submission.submitter_name,
            "submitter_email": submission.submitter_email,
            "original_filename": submission.original_filename,
            "upload_timestamp": submission.upload_timestamp,
            "file_hash": submission.file_hash,
            "file_size": submission.file_size,
            "ip_address": submission.ip_address,
            "user_agent": submission.user_agent,
            "file_integrity_verified": file_integrity_ok,
            "evaluation_result": json.loads(submission.evaluation_result or "{}"),
            "status": submission.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/download-original/{submission_id}")
async def download_original_file(submission_id: int, admin_key: str = ""):
    """Download the original submitted file"""
    if admin_key != os.getenv("ADMIN_AUDIT_KEY"):
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        db = get_database()
        submission = db.query(OutputSubmission).filter(
            OutputSubmission.id == submission_id
        ).first()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        file_content = file_storage.retrieve_file(submission.stored_file_path)
        
        return Response(
            content=file_content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={submission.original_filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/list-submissions")
async def list_submissions(admin_key: str = "test123"):
    """List all submissions in database"""
    try:
        db = get_database()
        submissions = db.query(OutputSubmission).all()
        
        result = []
        for sub in submissions:
            result.append({
                "id": sub.id,
                "dataset_name": sub.dataset_name,
                "submitter_name": sub.submitter_name,
                "submitter_email": sub.submitter_email,
                "upload_timestamp": getattr(sub, 'upload_timestamp', 'N/A'),
                "status": sub.status,
                "evaluation_result": sub.evaluation_result
            })
        
        return {"total_submissions": len(result), "submissions": result}
    except Exception as e:
        return {"error": str(e)}