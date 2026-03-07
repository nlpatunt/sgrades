# app/api/routes/output_submissions.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import Response
from typing import List, Dict, Any, Optional
from app.api.routes.dataset_ranges import get_score_range_for_dataset
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
from app.models.database import OutputSubmission, Dataset, EvaluationResult, LeaderboardCache
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

# ---------------------------------------------------------------------------
# Dataset type classification
# BEEtlE and SciEntSBank variants are classification tasks.
# Everything else is treated as regression.
# ---------------------------------------------------------------------------
CLASSIFICATION_DATASETS = {
    "BEEtlE_2way", "BEEtlE_3way",
    "SciEntSBank_2way", "SciEntSBank_3way",
}

def is_classification_dataset(dataset_name: str) -> bool:
    """Return True if the dataset uses categorical labels, not numeric scores."""
    normalized = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    return normalized in CLASSIFICATION_DATASETS


def download_ground_truth_private(dataset_name: str) -> Dict[str, Any]:
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    
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
        "OS_Dataset_q1": "ID",
        "OS_Dataset_q2": "ID",
        "OS_Dataset_q3": "ID",
        "OS_Dataset_q4": "ID",
        "OS_Dataset_q5": "ID",
        "Rice_Chem_Q1": "sis_id",
        "Rice_Chem_Q2": "sis_id",
        "Rice_Chem_Q3": "sis_id", 
        "Rice_Chem_Q4": "sis_id"
    }
        
    try:
        if normalized_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way", "ASAP-SAS"]:
            import pandas as pd
            import requests
            from io import StringIO
            
            if "BEEtlE" in normalized_name:
                suffix = "2way" if "2way" in normalized_name else "3way"
                urls = [
                    f"https://huggingface.co/datasets/nlpatunt/BEEtlE/resolve/main/test_{suffix}.csv",
                    f"https://huggingface.co/datasets/nlpatunt/BEEtlE/raw/main/test_{suffix}.csv"
                ]
            elif "SciEntSBank" in normalized_name:
                suffix = "2way" if "2way" in normalized_name else "3way"
                urls = [
                    f"https://huggingface.co/datasets/nlpatunt/SciEntSBank/resolve/main/test_{suffix}.csv",
                    f"https://huggingface.co/datasets/nlpatunt/SciEntSBank/raw/main/test_{suffix}.csv"
                ]
            elif normalized_name == "ASAP-SAS":
                urls = [
                    "https://huggingface.co/datasets/nlpatunt/ASAP-SAS/resolve/main/test.csv",
                    "https://huggingface.co/datasets/nlpatunt/ASAP-SAS/raw/main/test.csv"
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
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    df = pd.read_csv(StringIO(response.text))
                    columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
                    if columns_to_drop:
                        print(f"Dropping unnamed columns from {normalized_name}: {columns_to_drop}")
                        df = df.drop(columns=columns_to_drop)
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
            
            return {"status": "error", "error": f"All download URLs failed for {normalized_name}"}

        elif normalized_name in ["Rice_Chem_Q1", "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"]:
            q_num = normalized_name.split("_")[-1]
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files=f"{q_num}/test.csv")
            dataset = dataset["train"]
        elif normalized_name.startswith("OS_Dataset_q"):
            q_num = normalized_name.split("_q")[-1]
            dataset = load_dataset("nlpatunt/OS_Dataset", 
                                data_files=f"q{q_num}/test.csv",
                                trust_remote_code=True)
            dataset = dataset["train"]
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
        
        if 'dataset' in locals():
            df = dataset.to_pandas()
        else:
            return {"status": "error", "error": f"No dataset loaded for {normalized_name}"}
        
        if normalized_name.startswith("OS_Dataset") and "ID" not in df.columns:
            df["ID"] = range(1, len(df) + 1)
            print(f"Added ID column to {normalized_name}")

        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        id_column = id_columns_map.get(normalized_name, "ID")

        if id_column in df.columns and (df[id_column] == df.index).all():
            print("ERROR: ID column was replaced with DataFrame index!")
            df[id_column] = df.index + 1
        
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

def create_OS_Dataset_validators():
    return {
        "OS_Dataset_q1": GradeLikeHumanValidator("1"),
        "OS_Dataset_q2": GradeLikeHumanValidator("2"),
        "OS_Dataset_q3": GradeLikeHumanValidator("3"),
        "OS_Dataset_q4": GradeLikeHumanValidator("4"),
        "OS_Dataset_q5": GradeLikeHumanValidator("5")
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
            label_mapping.update({'contradictory': 'contradictory'})
        
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
        
        print(f"DEBUG: validate() called with testing_mode={testing_mode}")

        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {sorted(list(missing_cols))}")
        
        df_clean = df.copy()
        
        if self.id_column in df_clean.columns:
            duplicate_mask = df_clean[self.id_column].duplicated(keep='first')
            duplicate_count = duplicate_mask.sum()
            if duplicate_count > 0:
                print(f"DEBUG: Found {duplicate_count} duplicate IDs, testing_mode={testing_mode}")
                if testing_mode:
                    df_clean = df_clean[~duplicate_mask].copy()
                    warnings.append(f"Removed {duplicate_count} duplicate {self.id_column} rows (kept first occurrence)")
                    print(f"DEBUG: After removing duplicates: {len(df_clean)} rows remaining")
                else:
                    errors.append(f"Found {duplicate_count} duplicate {self.id_column} values - IDs must be unique")
        
        if self.primary_score_column not in df_clean.columns:
            errors.append(f"{self.primary_score_column} column is required for evaluation")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "primary_score_column": self.primary_score_column
            }
        
        missing_scores_mask = df_clean[self.primary_score_column].isna()
        missing_scores_count = missing_scores_mask.sum()
        if missing_scores_count > 0:
            print(f"DEBUG: Found {missing_scores_count} missing scores, testing_mode={testing_mode}")
            if testing_mode:
                df_clean = df_clean[~missing_scores_mask].copy()
                warnings.append(f"Removed {missing_scores_count} rows with missing {self.primary_score_column} values")
                print(f"DEBUG: After removing missing scores: {len(df_clean)} rows remaining")
            else:
                warnings.append(f"{self.primary_score_column} has {missing_scores_count} missing values")
        
        if len(df_clean) == 0:
            errors.append("No valid rows remaining after cleanup")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "primary_score_column": self.primary_score_column
            }
        
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
                print(f"DEBUG IELTS: Processing {validator_class}")
                print(f"DEBUG IELTS: Original values sample: {df_clean[self.primary_score_column].head(10).tolist()}")
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
                df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')
                print(f"DEBUG IELTS: Final values sample: {df_clean[self.primary_score_column].head(10).tolist()}")
                print(f"DEBUG IELTS: Final dtype: {df_clean[self.primary_score_column].dtype}")
        
            if validator_class == "MohlarValidator":
                df_clean[self.primary_score_column] = df_clean[self.primary_score_column].astype(str).str.strip()
                numeric_mask = df_clean[self.primary_score_column].str.match(r'^-?\d+\.?\d*$')
                non_numeric_count = (~numeric_mask).sum()
                if non_numeric_count > 0:
                    df_clean = df_clean[numeric_mask].copy()
                    warnings.append(f"Discarded {non_numeric_count} rows with non-numeric grades")
                df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')
            
            valid_scores = df_clean[self.primary_score_column].dropna()
            if len(valid_scores) > 0:
                numeric_scores = pd.to_numeric(valid_scores, errors='coerce')
                non_numeric_count = numeric_scores.isna().sum()
                if non_numeric_count > 0:
                    if testing_mode:
                        numeric_mask = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce').notna()
                        df_clean = df_clean[numeric_mask].copy()
                        warnings.append(f"Removed {non_numeric_count} non-numeric rows")
                        print(f"DEBUG: After removing non-numeric: {len(df_clean)} rows remaining")
                    else:
                        errors.append(f"{self.primary_score_column} has {non_numeric_count} non-numeric values")
                else:
                    df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')
        
        if len(errors) > 0:
            print(f"DEBUG VALIDATION: Errors found:")
            for i, error in enumerate(errors):
                print(f"  Error {i+1}: {error}")
        if len(warnings) > 0:
            print(f"DEBUG VALIDATION: Warnings:")
            for i, warning in enumerate(warnings):
                print(f"  Warning {i+1}: {warning}")
        print(f"DEBUG: Returning valid={len(errors) == 0}, {len(df_clean)} cleaned rows")
        
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
        OS_Dataset_validators = create_OS_Dataset_validators()

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
            "Ielts_Writing_Task_2_Dataset": IELTSTask2Validator(),
            "D_Ielts_Writing_Task_2_Dataset": IELTSTask2Validator(),
            **rice_chem_validators,
            **OS_Dataset_validators,
        }
        for q in ["q1", "q2", "q3", "q4", "q5"]:
            base_name = f"OS_Dataset_{q}"
            d_name = f"D_OS_Dataset_{q}"
            if base_name in self.validators:
                self.validators[d_name] = self.validators[base_name]
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
            "Ielts_Writing_Task_2_Dataset": ["ID", "band_score"],
            "Regrading_Dataset_J2C": ["ID", "grade"],
            "OS_Dataset_q1": ["ID", "score_1"],
            "OS_Dataset_q2": ["ID", "score_1"],
            "OS_Dataset_q3": ["ID", "score_1"],
            "OS_Dataset_q4": ["ID", "score_1"],
            "OS_Dataset_q5": ["ID", "score_1"],
        }

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
            "Ielts_Writing_Task_2_Dataset": "band_score",
            "persuade_2": "holistic_essay_score",
            "Regrading_Dataset_J2C": "grade",
            "OS_Dataset_q1": "score_1",
            "OS_Dataset_q2": "score_1",
            "OS_Dataset_q3": "score_1", 
            "OS_Dataset_q4": "score_1",
            "OS_Dataset_q5": "score_1",
            "Rice_Chem_Q1": "Score",
            "Rice_Chem_Q2": "Score",
            "Rice_Chem_Q3": "Score",
            "Rice_Chem_Q4": "Score"
        }
        
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
            "Ielts_Writing_Task_2_Dataset": "ID",
            "persuade_2": "essay_id_comp",
            "Regrading_Dataset_J2C": "ID",
            "OS_Dataset_q1": "ID",
            "OS_Dataset_q2": "ID",
            "OS_Dataset_q3": "ID",
            "OS_Dataset_q4": "ID",
            "OS_Dataset_q5": "ID",
            "Rice_Chem_Q1": "sis_id",
            "Rice_Chem_Q2": "sis_id",
            "Rice_Chem_Q3": "sis_id", 
            "Rice_Chem_Q4": "sis_id"
        }
    
    def get_ground_truth(self, dataset_name: str) -> Dict[str, Any]:
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

    def validate_full_structure(self, dataset_name: str, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame, testing_mode: bool = True) -> Dict[str, Any]:
        try:
            print(f"Starting validation for {dataset_name}")
            if dataset_name in self.validators:
                validator = self.validators[dataset_name]
                validation_result = validator.validate(prediction_df, testing_mode=testing_mode)
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
        try:
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
            
            if dataset_name == "D_BEEtlE_3way":
                print(f"DEBUG: Processing BEEtlE_3way validation")
                print(f"DEBUG: predictions_df shape: {predictions_df.shape}")
                print(f"DEBUG: predictions_df columns: {list(predictions_df.columns)}")
                print(f"DEBUG: predictions_df dtypes: {predictions_df.dtypes}")
                print(f"DEBUG: Sample prediction data:")
                print(predictions_df.head(2))
                print(f"DEBUG: ground_truth_df shape: {ground_truth_df.shape}")
                print(f"DEBUG: ground_truth_df columns: {list(ground_truth_df.columns)}")
            
            if "Ielts" in dataset_name or "IELTS" in dataset_name:
                print(f"DEBUG IELTS: predictions_df columns: {list(predictions_df.columns)}")
                print(f"DEBUG IELTS: predictions_df shape: {predictions_df.shape}")
                print(f"DEBUG IELTS: ground_truth_df columns: {list(ground_truth_df.columns)}")
                print(f"DEBUG IELTS: Expected validator: {normalized_name}")
                if normalized_name in self.validators:
                    validator = self.validators[normalized_name]
                    print(f"DEBUG IELTS: Required columns: {validator.required_columns}")
                    print(f"DEBUG IELTS: Score column: {validator.primary_score_column}")
                    
            if normalized_name in self.validators:
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
        print(f"DEBUG: Prediction df columns: {list(prediction_df.columns)}")
        print(f"DEBUG: Ground truth df columns: {list(ground_truth_df.columns)}")
        print(f"DEBUG: Looking for score column: '{score_col}' and ID column: '{id_col}'")
        print(f"DEBUG: Prediction df sample (first 3 rows):")
        print(prediction_df.head(3))
        print(f"DEBUG: Ground truth df sample (first 3 rows):")
        print(ground_truth_df.head(3))
        
        prediction_df[id_col] = prediction_df[id_col].astype(str)
        ground_truth_df[id_col] = ground_truth_df[id_col].astype(str)
        print(f"DEBUG: Prediction IDs after conversion: {prediction_df[id_col].head(5).tolist()}")
        print(f"DEBUG: Ground truth IDs after conversion: {ground_truth_df[id_col].head(5).tolist()}")

        prediction_df_unique = prediction_df.drop_duplicates(subset=[id_col], keep='first')
        ground_truth_df_unique = ground_truth_df.drop_duplicates(subset=[id_col], keep='first')
        print(f"DEBUG: After deduplication - Predictions: {len(prediction_df_unique)}, Ground truth: {len(ground_truth_df_unique)}")

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
        
        score_pred_col = f"{score_col}_pred"
        score_true_col = f"{score_col}_true"
        print(f"DEBUG: Merged data sample:")
        if score_pred_col in merged_df.columns and score_true_col in merged_df.columns:
            print(merged_df[[id_col, score_pred_col, score_true_col]].head(10))
        else:
            print(f"DEBUG: Expected columns {score_pred_col} and {score_true_col} not found")
            print(f"DEBUG: Available columns: {list(merged_df.columns)}")

        classification_datasets = ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]
        
        if normalized_name in classification_datasets:
            print(f"DEBUG: Processing classification dataset: {normalized_name}")
            if score_pred_col in merged_df.columns and score_true_col in merged_df.columns:
                mismatches = merged_df[merged_df[score_pred_col] != merged_df[score_true_col]]
                matches = merged_df[merged_df[score_pred_col] == merged_df[score_true_col]]
                print(f"DEBUG: Found {len(matches)} matches and {len(mismatches)} mismatches out of {len(merged_df)} total")
                if len(mismatches) > 0:
                    print(f"DEBUG: First 5 mismatches:")
                    print(mismatches[[id_col, score_pred_col, score_true_col]].head())
        
        if normalized_name in classification_datasets:
            pred_scores = merged_df[score_pred_col].values
            gt_scores = merged_df[score_true_col].values
            valid_mask = ~(pd.Series(pred_scores).isna() | pd.Series(gt_scores).isna())
            print(f"DEBUG: Classification - Pred scores sample: {pred_scores[:5]}")
            print(f"DEBUG: Classification - GT scores sample: {gt_scores[:5]}")
            print(f"DEBUG: Valid mask sum: {valid_mask.sum()}")
        else:
            pred_scores_numeric = pd.to_numeric(merged_df[score_pred_col], errors='coerce')
            gt_scores_numeric = pd.to_numeric(merged_df[score_true_col], errors='coerce')
            valid_mask = ~(pred_scores_numeric.isna() | gt_scores_numeric.isna())
            pred_scores = pred_scores_numeric[valid_mask].values
            gt_scores = gt_scores_numeric[valid_mask].values
            print(f"DEBUG: Regression - Pred scores sample: {pred_scores[:5]}")
            print(f"DEBUG: Regression - GT scores sample: {gt_scores[:5]}")

        essay_sets = None
        if normalized_name in ["ASAP-AES", "ASAP_plus_plus"]:
            if "essay_set" in merged_df.columns:
                essay_sets = merged_df["essay_set"].values

        return {
            "status": "success",
            "y_pred": pred_scores[valid_mask] if normalized_name in classification_datasets else pred_scores,
            "y_true": gt_scores[valid_mask] if normalized_name in classification_datasets else gt_scores,
            "essay_sets": essay_sets,
            "matched_count": int(valid_mask.sum()),
            "total_predictions": len(prediction_df),
            "total_ground_truth": len(ground_truth_df),
            "matching_method": "id_based"
        }
    
    def calculate_mae_percentage(self, mae: float, dataset_name: str, essay_set: int = 1) -> float:
        score_range = get_score_range_for_dataset(dataset_name, essay_set)
        range_size = score_range[1] - score_range[0]
        if range_size == 0:
            return 0.0
        mae_percent = (mae / range_size) * 100
        return round(mae_percent, 2)

    def calculate_mae_percentage_by_set(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        essay_sets: np.ndarray, dataset_name: str) -> Dict[str, Any]:
        from sklearn.metrics import mean_absolute_error
        results = {}
        per_set_mae_percents = []
        for essay_set in sorted(np.unique(essay_sets)):
            mask = essay_sets == essay_set
            y_true_set = y_true[mask]
            y_pred_set = y_pred[mask]
            if len(y_true_set) > 0:
                mae_set = mean_absolute_error(y_true_set, y_pred_set)
                mae_percent_set = self.calculate_mae_percentage(mae_set, dataset_name, int(essay_set))
                results[f"set_{int(essay_set)}"] = {
                    "mae": round(mae_set, 3),
                    "mae_percent": mae_percent_set,
                    "samples": len(y_true_set)
                }
                per_set_mae_percents.append(mae_percent_set)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          is_classification: bool = False) -> Dict[str, float]:
        """
        Classification datasets (BEEtlE / SciEntSBank):
            → f1_score, precision, recall, accuracy  (no QWK/Pearson/MAE/RMSE)

        Regression datasets (everything else):
            → quadratic_weighted_kappa, pearson_correlation,
              mean_absolute_error, mean_squared_error, root_mean_squared_error,
              accuracy  (no F1/precision/recall)

        The `is_classification` flag must be passed explicitly by the caller
        because by the time this function is called, labels have already been
        converted to integers, so isinstance(y_true[0], str) would be False
        even for classification datasets.
        """
        try:
            from sklearn.metrics import (
                mean_absolute_error, mean_squared_error, f1_score,
                precision_score, recall_score, cohen_kappa_score, accuracy_score
            )
            from scipy.stats import pearsonr

            def _safe(v):
                return float(v) if not pd.isna(v) else 0.0

            # ── CLASSIFICATION branch ─────────────────────────────────────────
            if is_classification:
                unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
                label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
                y_true_n = np.array([label_to_num[label] for label in y_true])
                y_pred_n = np.array([label_to_num[label] for label in y_pred])
                print(f"DEBUG: Label mapping: {label_to_num}")

                accuracy = accuracy_score(y_true_n, y_pred_n)
                try:
                    f1        = f1_score(y_true_n, y_pred_n, average="weighted", zero_division=0)
                    precision = precision_score(y_true_n, y_pred_n, average="weighted", zero_division=0)
                    recall    = recall_score(y_true_n, y_pred_n, average="weighted", zero_division=0)
                except Exception:
                    f1 = precision = recall = accuracy

                # Classification: only F1 / precision / recall / accuracy
                metrics = {
                    "f1_score":  _safe(f1),
                    "precision": _safe(precision),
                    "recall":    _safe(recall),
                    "accuracy":  _safe(accuracy),
                }

            # ── REGRESSION branch ─────────────────────────────────────────────
            else:
                print(f"DEBUG: Numeric data detected - using regression metrics")
                y_true = np.array(y_true, dtype=np.float64)
                y_pred = np.array(y_pred, dtype=np.float64)

                if len(y_true) == 1:
                    perfect_match = abs(y_true[0] - y_pred[0]) < 1e-10
                    qwk         = 1.0 if perfect_match else 0.0
                    correlation = 1.0 if perfect_match else 0.0
                    accuracy    = 1.0 if perfect_match else 0.0
                    mae  = 0.0 if perfect_match else abs(y_true[0] - y_pred[0])
                    mse  = 0.0 if perfect_match else (y_true[0] - y_pred[0]) ** 2
                    rmse = np.sqrt(mse)
                else:
                    correlation, _ = pearsonr(y_true, y_pred)
                    mae  = mean_absolute_error(y_true, y_pred)
                    mse  = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)

                    y_true_class = np.round(y_true).astype(np.int64)
                    y_pred_class = np.round(y_pred).astype(np.int64)
                    accuracy = accuracy_score(y_true_class, y_pred_class)
                    print(f"DEBUG: Accuracy - {np.sum(y_true_class == y_pred_class)}/{len(y_true_class)} = {accuracy:.4f}")

                    try:
                        qwk = cohen_kappa_score(y_true_class, y_pred_class, weights="quadratic")
                    except Exception:
                        qwk = 0.0

                # Regression: QWK / Pearson / MAE / MSE / RMSE / accuracy  (no F1/precision/recall)
                metrics = {
                    "quadratic_weighted_kappa": _safe(qwk),
                    "pearson_correlation":      _safe(correlation),
                    "mean_absolute_error":      _safe(mae),
                    "mean_squared_error":       _safe(mse),
                    "root_mean_squared_error":  _safe(rmse),
                    "accuracy":                 _safe(accuracy),
                }

            print(f"DEBUG: Final metrics: {metrics}")
            return metrics

        except Exception as e:
            print(f"DEBUG: Metrics calculation failed with exception: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            raise e
        
    def evaluate_submission(self, dataset_name: str, predictions_df: pd.DataFrame) -> Dict[str, Any]:
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
            gt_result = self.get_ground_truth(dataset_name)
            if gt_result["status"] != "success":
                return {
                    "status": "error",
                    "error": f"Failed to load ground truth: {gt_result.get('error')}"
                }
            ground_truth_df = gt_result["dataset"]
            normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
        
            validation_result = self.validate_full_structure(dataset_name, predictions_df, ground_truth_df, testing_mode=True)
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "error": "Structure validation failed",
                    "validation_details": validation_result
                }
            
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

            if normalized_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]:
                if "3way" in normalized_name:
                    label_map = {
                        'correct': 2, 'Correct': 2, 'CORRECT': 2,
                        'incorrect': 0, 'Incorrect': 0, 'INCORRECT': 0,
                        'contradictory': 1, 'Contradictory': 1, 'CONTRADICTORY': 1,
                        '2': 2, '1': 1, '0': 0, 2: 2, 1: 1, 0: 0
                    }
                else:
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

            try:
                calculated_metrics = self.calculate_metrics(
                    y_true, y_pred,
                    is_classification=is_classification_dataset(normalized_name)
                )
                if calculated_metrics and isinstance(calculated_metrics, dict):
                    metrics.update(calculated_metrics)
                    print(f"DEBUG: Metrics calculation successful: {metrics}")
                if normalized_name in ["ASAP-AES", "ASAP_plus_plus"]:
                    if matching_result.get("essay_sets") is not None:
                        mae_by_set = self.calculate_mae_percentage_by_set(
                            y_true, y_pred, 
                            matching_result["essay_sets"], 
                            dataset_name
                        )
                        metrics["mae_percentage_details"] = mae_by_set
                else:
                    mae = metrics.get("mean_absolute_error", 0)
                    mae_percent = self.calculate_mae_percentage(mae, dataset_name)
                    metrics["mae_percentage"] = mae_percent
            except Exception as metrics_error:
                print(f"WARNING: Metrics calculation failed: {metrics_error}")
                print("DEBUG: Using default metrics values")

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
                "metrics": metrics
            }

real_evaluation_engine = RealEvaluationEngine()

def get_all_submissions_from_db():
    try:
        db = get_database()
        db.rollback()
        submissions = db.query(OutputSubmission).all()
        result = []
        for sub in submissions:
            evaluation_data = {}
            if sub.evaluation_result:
                try:
                    evaluation_data = json.loads(sub.evaluation_result)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Invalid JSON for submission {sub.id} ({sub.submitter_name}): {e}")
                    print(f"Raw evaluation_result: {sub.evaluation_result[:200]}...")
                    continue
                except Exception as e:
                    print(f"ERROR: Unexpected error parsing submission {sub.id}: {e}")
                    continue
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

def update_leaderboard_cache_for_model(model_name: str):
    """Update cached leaderboard data for a specific model.

    Regression datasets  → avg QWK, Pearson, MAE, MSE, RMSE
    Classification datasets (BEEtlE / SciEntSBank) → avg F1, Precision, Recall, Accuracy

    The two groups are averaged SEPARATELY — classification datasets never
    affect regression metrics and vice versa.
    """
    try:
        db = get_database()
        submissions = db.query(OutputSubmission).filter(
            OutputSubmission.submitter_name == model_name
        ).all()
        if not submissions:
            return
        
        researcher_name = submissions[0].submitter_email
        description = submissions[0].description
        dataset_names = set()

        # Separate metric buckets
        regression_metrics: Dict[str, list] = {}
        classification_metrics: Dict[str, list] = {}

        for sub in submissions:
            if not sub.evaluation_result:
                continue
            try:
                eval_data = json.loads(sub.evaluation_result)
            except json.JSONDecodeError:
                continue
            if "real_evaluation" not in eval_data or "metrics" not in eval_data["real_evaluation"]:
                continue

            dataset_names.add(sub.dataset_name)
            metrics = eval_data["real_evaluation"]["metrics"]
            bucket = (
                classification_metrics
                if is_classification_dataset(sub.dataset_name)
                else regression_metrics
            )
            for metric_name, value in metrics.items():
                bucket.setdefault(metric_name, [])
                try:
                    bucket[metric_name].append(float(value))
                except (ValueError, TypeError):
                    continue

        def _avg(bucket: Dict[str, list]) -> Dict[str, float]:
            return {f"avg_{k}": statistics.mean(v) for k, v in bucket.items() if v}

        reg_avg = _avg(regression_metrics)
        cls_avg = _avg(classification_metrics)

        # Accuracy averaged across ALL datasets (regression + classification)
        combined_accuracy = (
            regression_metrics.get("accuracy", []) +
            classification_metrics.get("accuracy", [])
        )
        avg_accuracy_all = statistics.mean(combined_accuracy) if combined_accuracy else 0.0

        is_complete = len(dataset_names) >= 21

        cache_entry = db.query(LeaderboardCache).filter(
            LeaderboardCache.model_name == model_name
        ).first()

        shared_fields = dict(
            researcher_name=researcher_name,
            description=description,
            dataset_count=len(dataset_names),
            total_submissions=len(submissions),
            # Regression datasets only
            avg_quadratic_weighted_kappa=reg_avg.get("avg_quadratic_weighted_kappa", 0.0),
            avg_pearson_correlation=reg_avg.get("avg_pearson_correlation", 0.0),
            avg_mean_absolute_error=reg_avg.get("avg_mean_absolute_error", 0.0),
            avg_root_mean_squared_error=reg_avg.get("avg_root_mean_squared_error", 0.0),
            # Classification datasets only
            avg_f1_score=cls_avg.get("avg_f1_score", 0.0),
            avg_precision=cls_avg.get("avg_precision", 0.0),
            avg_recall=cls_avg.get("avg_recall", 0.0),
            # All datasets
            avg_accuracy=avg_accuracy_all,
            last_updated=datetime.now(),
            is_complete_benchmark=is_complete,
        )

        if cache_entry:
            cache_entry.researcher_name = researcher_name
            cache_entry.description = description
            cache_entry.dataset_count = len(dataset_names)
            cache_entry.total_submissions = len(submissions)
            cache_entry.avg_quadratic_weighted_kappa = avg_metrics.get('avg_quadratic_weighted_kappa')
            cache_entry.avg_pearson_correlation = avg_metrics.get('avg_pearson_correlation')
            cache_entry.avg_mean_absolute_error = avg_metrics.get('avg_mean_absolute_error')
            cache_entry.avg_root_mean_squared_error = avg_metrics.get('avg_root_mean_squared_error')
            cache_entry.avg_f1_score = avg_metrics.get('avg_f1_score')
            cache_entry.avg_precision = avg_metrics.get('avg_precision')
            cache_entry.avg_recall = avg_metrics.get('avg_recall')
            cache_entry.avg_accuracy = avg_metrics.get('avg_accuracy', 0.0)
            cache_entry.last_updated = datetime.now()
            cache_entry.is_complete_benchmark = is_complete
        else:
            cache_entry = LeaderboardCache(
                model_name=model_name,
                researcher_name=researcher_name,
                description=description,
                dataset_count=len(dataset_names),
                total_submissions=len(submissions),
                avg_quadratic_weighted_kappa=avg_metrics.get('avg_quadratic_weighted_kappa'),
                avg_pearson_correlation=avg_metrics.get('avg_pearson_correlation'),
                avg_mean_absolute_error=avg_metrics.get('avg_mean_absolute_error'),
                avg_root_mean_squared_error=avg_metrics.get('avg_root_mean_squared_error'),
                avg_f1_score=avg_metrics.get('avg_f1_score'),
                avg_precision=avg_metrics.get('avg_precision'),
                avg_recall=avg_metrics.get('avg_recall'),
                avg_accuracy=avg_metrics.get('avg_accuracy', 0.0),
                last_updated=datetime.now(),
                is_complete_benchmark=is_complete
            )
            db.add(cache_entry)

        db.commit()
        print(f"Updated leaderboard cache for {model_name}")
    except Exception as e:
        print(f"Failed to update cache for {model_name}: {e}")
        db.rollback()

def store_submission_in_db(dataset_name, model_name, metrics, description="", 
                          institution="", contact_email="", filename="",
                          file_content: bytes = None, ip_address="", user_agent=""):
    try:
        db = get_database()
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
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif obj is None:
        return None
    elif isinstance(obj, str):
        return obj
    else:
        return obj


def clean_dataframe_safe(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    return df_clean

def decode_file_content(content: bytes) -> tuple:
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None

router = APIRouter()

@router.get("/format/{dataset_name}", response_model=DatasetFormatResponse)
async def get_dataset_format(dataset_name: str):
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
        return clean_for_json({"datasets": datasets, "total": len(datasets)})
    except Exception as e:
        return clean_for_json({"error": str(e), "datasets": []})

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
            raise HTTPException(status_code=400, detail="Invalid file encoding. Please save your CSV as UTF-8.")
        security_validation = validator.validate_csv_content(csv_content)
        if not security_validation["valid"]:
            return clean_for_json({
                "success": False,
                "error": f"Security validation failed: {security_validation['error']}",
                "dataset": dataset_name,
                "filename": file.filename
            })
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
        evaluation_result = real_evaluation_engine.evaluate_submission(dataset_name=dataset_name, predictions_df=df_clean)
        if evaluation_result["status"] != "success":
            return clean_for_json({
                "success": False,
                "evaluation_error": evaluation_result.get("error"),
                "dataset": dataset_name,
                "filename": safe_filename
            })
        db_result = store_submission_in_db(
            dataset_name=dataset_name, model_name=model_name,
            metrics=evaluation_result["metrics"], description=description,
            institution=institution, contact_email=contact_email,
            filename=safe_filename, file_content=content,
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
                results.append({"filename": file.filename, "dataset": dataset_name, "success": False, "error": "Invalid file encoding"})
                failed_uploads += 1
                continue
            security_validation = validator.validate_csv_content(csv_content)
            if not security_validation["valid"]:
                results.append({"filename": file.filename, "dataset": dataset_name, "success": False, "error": f"Security validation failed: {security_validation['error']}"})
                failed_uploads += 1
                continue
            safe_filename = validator.sanitize_filename(file.filename or "upload.csv")
            df = pd.read_csv(io.StringIO(csv_content))
            df_clean = clean_dataframe_safe(df)
            validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df_clean)
            if not validation_result["valid"]:
                results.append({"filename": safe_filename, "dataset": dataset_name, "success": False, "validation_errors": validation_result["errors"], "validation_warnings": validation_result.get("warnings", [])})
                failed_uploads += 1
                continue
            evaluation_result = real_evaluation_engine.evaluate_submission(dataset_name=dataset_name, predictions_df=df_clean)
            if evaluation_result.get("status") != "success":
                results.append({"filename": safe_filename, "dataset": dataset_name, "success": False, "evaluation_error": evaluation_result.get("error", "Evaluation failed")})
                failed_uploads += 1
                continue
            db_result = store_submission_in_db(
                dataset_name=dataset_name, model_name=model_name,
                metrics=evaluation_result["metrics"], description=description,
                institution=institution, contact_email=contact_email,
                filename=safe_filename, file_content=content,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "")
            )
            results.append({"filename": safe_filename, "dataset": dataset_name, "success": True, "evaluation": evaluation_result, "database": db_result, "encoding_used": encoding_used, "security_info": f"File validated: {security_validation.get('info', 'Security check passed')}"})
            successful_uploads += 1
        except Exception as e:
            safe_filename = validator.sanitize_filename(getattr(file, "filename", "unknown"))
            results.append({"filename": safe_filename, "dataset": dataset_name if "dataset_name" in locals() else "unknown", "success": False, "error": str(e)})
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
    request: Request,
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    model_name: str = Form("Test_Model"),
    submitter_name: str = Form("Test_User"),
    submitter_email: str = Form("test@example.com"),
    description: str = Form("Single dataset test")
):
    validator = CSVSecurityValidator()
    try:
        content = await file.read()
        csv_content, encoding_used = decode_file_content(content)
        if csv_content is None:
            return clean_for_json({"success": False, "error": "Invalid file encoding. Please save your CSV as UTF-8.", "testing_mode": True})
        df = pd.read_csv(io.StringIO(csv_content))
        df_clean = clean_dataframe_safe(df)
        validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df_clean, testing_mode=True)
        if not validation_result["valid"]:
            raw_errors = validation_result.get("errors", [])
            errors = [str(err) for err in raw_errors] if raw_errors else ["Validation failed - unknown error"]
            print(f"DEBUG /test-single-dataset: Validation failed with errors: {errors}")
            return clean_for_json({"success": False, "validation_errors": validation_result["errors"], "validation_warnings": validation_result.get("warnings", []), "dataset": dataset_name, "filename": file.filename, "testing_mode": True})
        df_cleaned = validation_result.get("cleaned_df", df_clean)
        evaluation_result = real_evaluation_engine.evaluate_submission(dataset_name=dataset_name, predictions_df=df_cleaned)
        if evaluation_result["status"] != "success":
            return clean_for_json({"success": False, "evaluation_error": evaluation_result.get("error"), "dataset": dataset_name, "filename": file.filename, "testing_mode": True})
        try:
            db_result = store_submission_in_db(
                dataset_name=dataset_name, model_name=f"Test_{dataset_name}",
                metrics=evaluation_result.get("metrics", {}), description=description,
                institution="", contact_email=submitter_email,
                filename=file.filename, file_content=content,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "")
            )
            if db_result.get("status") == "success":
                print(f"✓ Test submission stored with ID: {db_result.get('submission_id')}")
            else:
                print(f"✗ Failed to store: {db_result.get('error')}")
        except Exception as storage_error:
            print(f"✗ Database storage failed: {storage_error}")
        return clean_for_json({
            "success": True,
            "testing_mode": False,
            "dataset": dataset_name,
            "filename": file.filename,
            "evaluation": evaluation_result,
            "metrics": evaluation_result.get("metrics", {}),
            "encoding_used": encoding_used,
            "validation_warnings": validation_result.get("warnings", []),
        })
    except Exception as e:
        return clean_for_json({"success": False, "error": str(e), "dataset": dataset_name, "filename": file.filename if file else "unknown", "testing_mode": True})
    
@router.get("/leaderboard", response_model=Dict[str, Any]) 
async def get_leaderboard(
    dataset: str = "All Datasets",
    metric: str = "avg_quadratic_weighted_kappa",
    limit: int = 50,
    min_datasets: int = 0,
    complete_only: bool = False
):
    try:
        print("DEBUG: Starting leaderboard generation")
        
        if complete_only:
            min_datasets = 23
            print(f"DEBUG: Complete benchmark mode - min_datasets set to {min_datasets}")
        else:
            min_datasets = 0
            print(f"DEBUG: Show all evaluations mode")
        
        current_time = datetime.now().isoformat()
        submissions = get_all_submissions_from_db()
        print(f"DEBUG: Got {len(submissions)} total submissions")
    
        real_submissions = [s for s in submissions if s.get('has_real_evaluation', False)]
        print(f"DEBUG: {len(real_submissions)} have real_evaluation")
        
        if not real_submissions:
            return clean_for_json({
                "dataset": dataset, "metric": metric, "total_entries": 0,
                "last_updated": current_time, "rankings": [], "summary_stats": {},
                "complete_only": complete_only, "note": "No evaluations found"
            })

        print("DEBUG: Starting model aggregation")

        # Build per-model lists: parallel dataset_list + per_dataset_metrics
        model_aggregates: Dict[str, Any] = {}
        for submission in real_submissions:
            model_name = submission['model_name']
            if model_name not in model_aggregates:
                model_aggregates[model_name] = {
                    'dataset_list': [],
                    'per_dataset_metrics': [],
                    'total_submissions': 0,
                    'institution': submission.get('institution', ''),
                    'description': submission.get('description', ''),
                    'contact_email': submission.get('contact_email', '')
                }
            model_aggregates[model_name]['dataset_list'].append(submission['dataset_name'])
            model_aggregates[model_name]['per_dataset_metrics'].append(submission.get('metrics', {}))
            model_aggregates[model_name]['total_submissions'] += 1
=======
            
            metrics = submission.get('metrics', {})
            
            for metric_name, value in metrics.items():
                if metric_name not in model_aggregates[model_name]['metrics']:
                    model_aggregates[model_name]['metrics'][metric_name] = []
                
                try:
                    if value is None:
                        continue  # Skip nulls - metric not applicable
                    elif isinstance(value, str):
                        if value.strip() == '' or value.lower() in ['nan', 'null', 'none']:
                            continue  # Skip nulls
                        else:
                            numeric_value = float(value)
                    else:
                        numeric_value = float(value)
                    
                    model_aggregates[model_name]['metrics'][metric_name].append(numeric_value)
                    
                except (ValueError, TypeError):
                    pass  # Skip bad values
>>>>>>> Stashed changes

        print("DEBUG: Filtering models based on dataset count")
        filtered_models: Dict[str, Any] = {}
        for model_name, data in model_aggregates.items():
            unique_datasets = list(set(data['dataset_list']))
            dataset_count = len(unique_datasets)
            if dataset_count >= min_datasets:
                filtered_models[model_name] = data
                filtered_models[model_name]['unique_datasets_count'] = dataset_count
                filtered_models[model_name]['is_complete_benchmark'] = dataset_count >= 23
                print(f"DEBUG: {model_name} has {dataset_count} datasets - INCLUDED")
            else:
                print(f"DEBUG: {model_name} has {dataset_count} datasets - EXCLUDED (min: {min_datasets})")
        
        print(f"DEBUG: {len(filtered_models)} models after filtering")
        
        if not filtered_models:
            return clean_for_json({
                "dataset": dataset, "metric": metric, "total_entries": 0,
                "last_updated": current_time, "rankings": [], "summary_stats": {},
                "complete_only": complete_only, "min_datasets": min_datasets,
                "note": f"No models found with at least {min_datasets} datasets"
            })
        
        print("DEBUG: Starting rankings calculation")
        rankings = []

        for model_name, data in filtered_models.items():
            try:
                # ── Separate regression vs classification per-dataset ─────────
                reg_bucket: Dict[str, list] = {}
                cls_bucket: Dict[str, list] = {}

                for ds_name, ds_metrics in zip(data['dataset_list'], data['per_dataset_metrics']):
                    bucket = cls_bucket if is_classification_dataset(ds_name) else reg_bucket
                    for metric_name, value in ds_metrics.items():
                        bucket.setdefault(metric_name, [])
                        try:
                            if value is None or value == '':
                                numeric_val = 0.0
                            elif isinstance(value, str):
                                numeric_val = 0.0 if value.strip().lower() in ['nan', 'null', 'none', ''] else float(value.strip())
                            else:
                                numeric_val = float(value)
                            bucket[metric_name].append(numeric_val)
                        except (ValueError, TypeError):
                            bucket[metric_name].append(0.0)

                def _safe_avg(bucket: Dict[str, list], key: str) -> float:
                    vals = bucket.get(key, [])
                    if not vals:
                        return 0.0
                    try:
                        return statistics.mean(vals)
                    except Exception:
                        return 0.0

                # QWK/Pearson/MAE/RMSE: regression datasets only
                # F1/precision/recall: classification datasets only
                # Accuracy: all datasets (regression + classification)
                all_accuracy_vals = (
                    reg_bucket.get("accuracy", []) +
                    cls_bucket.get("accuracy", [])
                )

                def _safe_mean(vals):
                    return statistics.mean(vals) if vals else 0.0

                avg_metrics = {
                    "avg_quadratic_weighted_kappa": _safe_avg(reg_bucket, "quadratic_weighted_kappa"),
                    "avg_pearson_correlation":      _safe_avg(reg_bucket, "pearson_correlation"),
                    "avg_mean_absolute_error":      _safe_avg(reg_bucket, "mean_absolute_error"),
                    "avg_root_mean_squared_error":  _safe_avg(reg_bucket, "root_mean_squared_error"),
                    "avg_f1_score":                 _safe_avg(cls_bucket, "f1_score"),
                    "avg_precision":                _safe_avg(cls_bucket, "precision"),
                    "avg_recall":                   _safe_avg(cls_bucket, "recall"),
                    "avg_accuracy":                 _safe_mean(all_accuracy_vals),
                }

                ranking_entry = {
                    "model_name": model_name,
                    "institution": data['institution'],
                    "description": data['description'],
                    "contact_email": data['contact_email'],
                    "datasets_evaluated": list(set(data['dataset_list'])),
                    "unique_datasets_count": data['unique_datasets_count'],
                    "total_submissions": data['total_submissions'],
                    "complete_benchmark": data['is_complete_benchmark'],
                rankings.sort(key=lambda x: (x.get(metric) is None, x.get(metric) if x.get(metric) is not None else float('inf')))
            else:
                rankings.sort(key=lambda x: (x.get(metric) is None, -(x.get(metric) or 0)))
            
            rankings = rankings[:limit]
        except Exception as sort_error:
            print(f"ERROR: Sorting failed: {sort_error}")
        
        summary_stats = {}
        if rankings:
            for metric_key in metric_keys:
                try:
                    values = [r.get(metric_key, 0) for r in rankings if r.get(metric_key) is not None]
                    if values:
                        numeric_values = []
                        for v in values:
                            try:
                                numeric_values.append(float(v))
                            except (ValueError, TypeError):
                                numeric_values.append(0.0)
                        if numeric_values:
                            summary_stats[metric_key] = {
                                'mean': statistics.mean(numeric_values),
                                'std': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
                                'min': min(numeric_values),
                                'max': max(numeric_values)
                            }
                except Exception:
                    pass
            summary_stats['total_researchers'] = len(rankings)
            summary_stats['complete_benchmarks'] = len([r for r in rankings if r['complete_benchmark']])
            summary_stats['partial_benchmarks'] = len([r for r in rankings if not r['complete_benchmark']])
        
        return clean_for_json({
            "dataset": dataset,
            "metric": metric,
            "total_entries": len(rankings),
            "last_updated": current_time,
            "rankings": rankings,
            "summary_stats": summary_stats,
            "available_metrics": metric_keys,
            "complete_only": complete_only,
            "min_datasets": min_datasets,
            "note": (
                f"{'Complete benchmarks (23+ datasets)' if complete_only else 'All evaluations'} — "
                f"top {len(rankings)} models. "
                f"QWK/Pearson/MAE/RMSE averaged over regression datasets only; "
                f"F1/Precision/Recall averaged over classification datasets "
                f"(BEEtlE_2way, BEEtlE_3way, SciEntSBank_2way, SciEntSBank_3way) only; "
                f"Accuracy averaged over all datasets."
            )
        })
        
    except Exception as e:
        print(f"ERROR: Leaderboard generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate leaderboard: {str(e)}")
    
@router.get("/leaderboard-cached", response_model=Dict[str, Any]) 
async def get_cached_leaderboard(
    dataset: str = "All Datasets",
    metric: str = "avg_quadratic_weighted_kappa",
    limit: int = 20,
    min_datasets: int = 21 
):
    """Fast leaderboard using pre-calculated cache.
    QWK/Pearson/MAE/RMSE = regression datasets only.
    F1/Precision/Recall = classification datasets only.
    Accuracy = all datasets.
    """
    try:
        db = get_database()
        cached_results = db.query(LeaderboardCache).filter(
            LeaderboardCache.is_complete_benchmark == True,
            LeaderboardCache.dataset_count >= min_datasets
        ).all()
        if not cached_results:
            return {"total_entries": 0, "rankings": [], "summary_stats": {}, "note": "No complete benchmarks found in cache"}
        
        rankings = []
        for cache_entry in cached_results:
            rankings.append({
                "model_name": cache_entry.model_name,
                "contact_email": cache_entry.researcher_name,
                "description": cache_entry.description,
                "unique_datasets_count": cache_entry.dataset_count,
                "total_submissions": cache_entry.total_submissions,
                "avg_quadratic_weighted_kappa": cache_entry.avg_quadratic_weighted_kappa,
                "avg_pearson_correlation": cache_entry.avg_pearson_correlation,
                "avg_mean_absolute_error": cache_entry.avg_mean_absolute_error,
                "avg_root_mean_squared_error": cache_entry.avg_root_mean_squared_error,
                "avg_f1_score": cache_entry.avg_f1_score,
                "avg_precision": cache_entry.avg_precision,
                "avg_recall": cache_entry.avg_recall,
                "avg_accuracy": cache_entry.avg_accuracy,
                "last_updated": cache_entry.last_updated.isoformat() if cache_entry.last_updated else None
            })
        
        if metric in ["avg_mean_absolute_error", "avg_root_mean_squared_error"]:
            rankings.sort(key=lambda x: x.get(metric, float('inf')))
        else:
            rankings.sort(key=lambda x: x.get(metric, 0), reverse=True)
        rankings = rankings[:limit]

        if rankings:
            summary_stats = {
                "total_researchers": len(rankings),
                "complete_benchmarks": len(rankings),
                "avg_quadratic_weighted_kappa": {"mean": sum(r["avg_quadratic_weighted_kappa"] for r in rankings) / len(rankings)},
                "avg_f1_score": {"mean": sum(r["avg_f1_score"] for r in rankings) / len(rankings)},
                "avg_precision": {"mean": sum(r["avg_precision"] for r in rankings) / len(rankings)},
                "avg_recall": {"mean": sum(r["avg_recall"] for r in rankings) / len(rankings)}
            }
        else:
            summary_stats = {"total_researchers": 0}

        return {
            "total_entries": len(rankings),
            "rankings": rankings,
            "summary_stats": summary_stats,
            "last_updated": datetime.now().isoformat(),
            "note": (
                f"Cached results — {len(rankings)} complete benchmarks. "
                f"QWK/Pearson/MAE/RMSE = regression datasets only; "
                f"F1/Precision/Recall = classification datasets only; "
                f"Accuracy = all datasets."
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cached leaderboard failed: {str(e)}")

@router.post("/admin/refresh-leaderboard-cache")
async def refresh_leaderboard_cache():
    try:
        db = get_database()
        model_names = db.query(OutputSubmission.submitter_name).filter(
            OutputSubmission.submitter_name.like('zero-shot-%')
        ).distinct().all()
        updated_models = []
        for (model_name,) in model_names:
            update_leaderboard_cache_for_model(model_name)
            updated_models.append(model_name)
        return {"message": f"Cache updated for {len(updated_models)} models", "models": updated_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache refresh failed: {str(e)}")

@router.get("/stats", response_model=SubmissionsStatsResponse)
async def get_platform_stats():
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
    if admin_key != os.getenv("ADMIN_AUDIT_KEY"):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        db = get_database()
        submission = db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        file_integrity_ok = file_storage.verify_file_integrity(submission.stored_file_path, submission.file_hash)
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
    if admin_key != os.getenv("ADMIN_AUDIT_KEY"):
        raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        db = get_database()
        submission = db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        file_content = file_storage.retrieve_file(submission.stored_file_path)
        return Response(
            content=file_content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={submission.original_filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/admin/list-submissions")
async def list_submissions(admin_key: str = "test123"):
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