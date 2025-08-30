# app/api/routes/output_submissions.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import os
import zipfile
import tempfile
import hashlib
from datetime import datetime
import statistics
import numpy as np
import json
from app.models.database import OutputSubmission, Dataset, EvaluationResult
from app.services.database_service import get_database

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

def download_ground_truth_private(dataset_name: str) -> Dict[str, Any]:
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
        "EFL": "ID",
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
        print(f"Loading private ground truth dataset: nlpatunt/{dataset_name}")
        
        if dataset_name == "BEEtlE_2way":
            dataset = load_dataset("nlpatunt/BEEtlE", "2way", split="test", trust_remote_code=True)
        elif dataset_name == "BEEtlE_3way":
            dataset = load_dataset("nlpatunt/BEEtlE", "3way", split="test", trust_remote_code=True)
        elif dataset_name == "SciEntSBank_2way":
            dataset = load_dataset("nlpatunt/SciEntSBank", "2way", split="test", trust_remote_code=True)
        elif dataset_name == "SciEntSBank_3way":
            dataset = load_dataset("nlpatunt/SciEntSBank", "3way", split="test", trust_remote_code=True)
        elif dataset_name in ["Rice_Chem_Q1", "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"]:
            q_num = dataset_name.split("_")[-1]
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files=f"{q_num}/test.csv")
            dataset = dataset["train"]
        elif dataset_name.startswith("grade_like_a_human_dataset_os_q"):
            q_num = dataset_name.split("_q")[-1]
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", name=f"q{q_num}", split="test", trust_remote_code=True)
            
        elif dataset_name == "persuade_2":
            dataset = load_dataset("nlpatunt/persuade_2", data_files="test.csv")
            dataset = dataset["train"] 
        elif dataset_name == "EFL":
            dataset = load_dataset("nlpatunt/EFL", data_files="test.csv")
            dataset = dataset["train"]
        elif dataset_name == "Mohlar":
            dataset = load_dataset("nlpatunt/Mohlar", data_files="test.csv")
            dataset = dataset["train"]
        else:
            try:
                dataset = load_dataset(f"nlpatunt/{dataset_name}", split="test", trust_remote_code=True)
            except:
                dataset = load_dataset(f"nlpatunt/{dataset_name}")
                if hasattr(dataset, 'keys'):
                    first_split = list(dataset.keys())[0]
                    dataset = dataset[first_split]
        
        df = dataset.to_pandas()
        
        if dataset_name.startswith("grade_like_a_human_dataset_os") and "ID" not in df.columns:
            df["ID"] = range(1, len(df) + 1)
            print(f"Added ID column to {dataset_name}")

        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"DEBUG: Dropped empty columns: {columns_to_drop}")

        id_column = id_columns_map.get(dataset_name, "ID")
        
        print(f"DEBUG: After pandas conversion - {id_column} column: {df[id_column].head(10).tolist()}")
        print(f"DEBUG: {id_column} column dtype: {df[id_column].dtype}")
        print(f"DEBUG: DataFrame index: {df.index.tolist()[:10]}")

        if id_column in df.columns and (df[id_column] == df.index).all():
            print("ERROR: ID column was replaced with DataFrame index!")
            df[id_column] = df.index + 1
            print(f"FIXED: {id_column} column now: {df[id_column].head(10).tolist()}")
            
        if dataset_name == "BEEtlE_2way":
            print(f"DEBUG: Ground truth first 10 rows:")
            print(df[[id_column, 'label']].head(10))
            df.to_csv(f"ground_truth_{dataset_name}.csv", index=False)
            print(f"Saved ground truth to ground_truth_{dataset_name}.csv for comparison")
            
        print(f"Loaded private dataset: {len(df)} rows, columns: {list(df.columns)}")
        
        return {
            "status": "success",
            "dataset": df,
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        print(f"HuggingFace datasets library failed: {str(e)}")
        
        if dataset_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]:
            try:
                import pandas as pd
                import requests
                from io import StringIO
                
                if "BEEtlE" in dataset_name:
                    suffix = "2way" if "2way" in dataset_name else "3way"
                    url = f"https://huggingface.co/datasets/nlpatunt/BEEtlE/raw/main/test_{suffix}.csv"
                else:  # SciEntSBank
                    suffix = "2way" if "2way" in dataset_name else "3way"
                    url = f"https://huggingface.co/datasets/nlpatunt/SciEntSBank/raw/main/test_{suffix}.csv"
                
                print(f"Attempting direct CSV download from: {url}")
                response = requests.get(url)
                response.raise_for_status()
                
                df = pd.read_csv(StringIO(response.text))
                print(f"Loaded {dataset_name} via direct download: {len(df)} rows, columns: {list(df.columns)}")
                
                return {
                    "status": "success",
                    "dataset": df,
                    "rows": len(df),
                    "columns": list(df.columns)
                }
                
            except Exception as fallback_error:
                print(f"Direct CSV download also failed: {fallback_error}")
                return {"status": "error", "error": f"Both dataset API and direct download failed: {str(e)} | {str(fallback_error)}"}
        
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
            label_mapping.update({
                valid_label.lower(): valid_label,
                valid_label.upper(): valid_label,
                valid_label.capitalize(): valid_label,
                valid_label: valid_label
            })
        
        if 'correct' in self.valid_labels:
            label_mapping.update({
                '1': 'correct', 'true': 'correct', 'yes': 'correct', 'right': 'correct'
            })
        if 'incorrect' in self.valid_labels:
            label_mapping.update({
                '0': 'incorrect', 'false': 'incorrect', 'no': 'incorrect', 'wrong': 'incorrect'
            })
        if 'partial_correct' in self.valid_labels:
            label_mapping.update({
                'partial': 'partial_correct', 'partially_correct': 'partial_correct'
            })
        
        df_clean['_original_label'] = df_clean[self.primary_score_column].copy()
        df_clean[self.primary_score_column] = df_clean[self.primary_score_column].str.lower().map(label_mapping).fillna(df_clean[self.primary_score_column])
        
        valid_mask = df_clean[self.primary_score_column].isin(self.valid_labels)
        invalid_mask = ~valid_mask
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            invalid_examples = df_clean[invalid_mask]['_original_label'].unique()[:5].tolist()
            
            if handle_invalid == 'assign_fallback' or testing_mode:
                df_clean.loc[invalid_mask, self.primary_score_column] = '4'  # Invalid = 4
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

class EFLValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "_Human_Mean"], "_Human_Mean", "ID")

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

        self.validators = {
            "ASAP-AES": ASAPAESValidator(),
            "ASAP-SAS": ASAPSASValidator(),
            "ASAP2": ASAP2Validator(),
            "ASAP_plus_plus": ASAPPlusPlusValidator(),
            "BEEtlE_2way": BEEtlE2WayValidator(),
            "BEEtlE_3way": BEEtlE3WayValidator(),
            "SciEntSBank_2way": SciEntSBank2WayValidator(),
            "SciEntSBank_3way": SciEntSBank3WayValidator(),
            "EFL": EFLValidator(),
            "Mohlar": MohlarValidator(),
            "CSEE": CSEEValidator(),
            "persuade_2": Persuade2Validator(),
            **rice_chem_validators,
            "Regrading_Dataset_J2C": RegradingDatasetJ2CValidator(),
            "Ielts_Writing_Dataset": IELTSWritingValidator(),
            "Ielst_Writing_Task_2_Dataset": IELTSTask2Validator(),
            **grade_like_human_validators,
        }

        # Submission requirements using original ID column names
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
            "EFL": ["ID", "_Human_Mean"],
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
            "EFL": "_Human_Mean",
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
            "EFL": "ID",
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
            gt_result = self.get_ground_truth(dataset_name)
            if gt_result["status"] != "success":
                return {
                    "valid": False,
                    "errors": [f"Cannot load dataset schema: {gt_result.get('error')}"],
                    "warnings": []
                }
            
            ground_truth_df = gt_result["dataset"]
            
            # Use the enhanced validator with testing_mode parameter
            if dataset_name in self.validators:
                validator = self.validators[dataset_name]
                validation_result = validator.validate(predictions_df, testing_mode=testing_mode)
                
                return {
                    "valid": validation_result["valid"],
                    "errors": validation_result.get("errors", []),
                    "warnings": validation_result.get("warnings", []),
                    "expected_columns": validator.required_columns,
                    "score_column": self.get_score_column(dataset_name),
                    "matching_method": "id_based",
                    "cleaned_df": validation_result.get("cleaned_df", predictions_df)
                }
            else:
                return {
                    "valid": False,
                    "errors": [f"No validator found for dataset: {dataset_name}"],
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
   
        score_col = self.get_score_column(dataset_name)
        id_col = self.get_id_column(dataset_name)
        
        print(f"Using ID-based matching for {dataset_name} with column '{id_col}'")
        
        # Ensure ID columns are strings for consistent matching
        prediction_df[id_col] = prediction_df[id_col].astype(str)
        ground_truth_df[id_col] = ground_truth_df[id_col].astype(str)

        # Remove duplicates
        prediction_df_unique = prediction_df.drop_duplicates(subset=[id_col], keep='first')
        ground_truth_df_unique = ground_truth_df.drop_duplicates(subset=[id_col], keep='first')

        # Merge on the correct ID column
        merged_df = prediction_df_unique.merge(
            ground_truth_df_unique[[id_col, score_col]],
            on=id_col, how="inner", suffixes=("_pred", "_true")
        )
        
        if len(merged_df) == 0:
            return {"status": "error", "error": f"No matching {id_col} found between predictions and ground truth"}
        
        if dataset_name.startswith("grade_like_a_human_dataset_os"):
            print(f"DEBUG: Prediction IDs first 10: {prediction_df[id_col].head(10).tolist()}")
            print(f"DEBUG: Ground truth IDs first 10: {ground_truth_df[id_col].head(10).tolist()}")
            print(f"DEBUG: Merged df shape: {merged_df.shape}")
            print(f"DEBUG: Sample comparison:")
            print(merged_df[['ID', f'{score_col}_pred', f'{score_col}_true']].head(10))

        # Debug for BEEtlE/SciEntSBank
        classification_datasets = ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]
        
        if dataset_name in classification_datasets:
            print(f"DEBUG: First 10 rows of merge:")
            print(merged_df[[id_col, f"{score_col}_pred", f"{score_col}_true"]].head(10))
            
            # Check for any mismatches
            mismatches = merged_df[merged_df[f"{score_col}_pred"] != merged_df[f"{score_col}_true"]]
            if len(mismatches) > 0:
                print(f"DEBUG: Found {len(mismatches)} mismatches out of {len(merged_df)}")
                print(f"DEBUG: First 5 mismatches:")
                print(mismatches[[id_col, f"{score_col}_pred", f"{score_col}_true"]].head())
        
        # Extract scores based on dataset type
        if dataset_name in classification_datasets:
            # For classification, get raw string values then convert
            pred_scores = merged_df[f"{score_col}_pred"].values
            gt_scores = merged_df[f"{score_col}_true"].values
            valid_mask = ~(pd.Series(pred_scores).isna() | pd.Series(gt_scores).isna())
        else:
            # For regression, convert to numeric
            pred_scores_numeric = pd.to_numeric(merged_df[f"{score_col}_pred"], errors='coerce')
            gt_scores_numeric = pd.to_numeric(merged_df[f"{score_col}_true"], errors='coerce')
            valid_mask = ~(pred_scores_numeric.isna() | gt_scores_numeric.isna())
            pred_scores = pred_scores_numeric[valid_mask].values
            gt_scores = gt_scores_numeric[valid_mask].values

        return {
            "status": "success",
            "y_pred": pred_scores[valid_mask] if dataset_name in classification_datasets else pred_scores,
            "y_true": gt_scores[valid_mask] if dataset_name in classification_datasets else gt_scores,
            "matched_count": int(valid_mask.sum()),
            "total_predictions": len(prediction_df),
            "total_ground_truth": len(ground_truth_df),
            "matching_method": "id_based"
        }
    
    # In output_submissions.py, replace your existing function with this one

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        try:
            from sklearn.metrics import (
                mean_absolute_error, mean_squared_error, f1_score,
                precision_score, recall_score, cohen_kappa_score
            )
            from scipy.stats import pearsonr

            print(f"DEBUG: Starting metrics calculation with {len(y_true)} samples")
            
            # Calculate basic metrics regardless of sample size
            mae = mean_absolute_error(y_true, y_pred) if len(y_true) > 0 else 0.0
            mse = mean_squared_error(y_true, y_pred) if len(y_true) > 0 else 0.0
            rmse = np.sqrt(mse)
            print(f"DEBUG: MAE={mae}, MSE={mse}, RMSE={rmse}")
            
            # Handle single sample case
            if len(y_true) == 1:
                print(f"DEBUG: Single sample detected - using simplified metrics")
                # For single sample, correlation and QWK are either perfect (1.0) or failed (0.0)
                perfect_match = abs(y_true[0] - y_pred[0]) < 1e-10
                correlation = 1.0 if perfect_match else 0.0
                qwk = 1.0 if perfect_match else 0.0
                
                # Classification metrics for single sample
                try:
                    y_true_class = y_true.round().astype(int)
                    y_pred_class = y_pred.round().astype(int)
                    f1 = 1.0 if y_true_class[0] == y_pred_class[0] else 0.0
                    precision = recall = f1
                    print(f"DEBUG: Single sample - F1={f1}, Precision={precision}, Recall={recall}")
                except Exception as e:
                    print(f"DEBUG: Single sample classification metrics failed: {e}")
                    f1 = precision = recall = 1.0 if perfect_match else 0.0
                    
            elif len(y_true) < 1:
                print(f"DEBUG: No samples - returning zero metrics")
                return {
                    "quadratic_weighted_kappa": 0.0,
                    "pearson_correlation": 0.0,
                    "mean_absolute_error": 0.0,
                    "mean_squared_error": 0.0,
                    "root_mean_squared_error": 0.0,
                    "f1_score": 0.0, "precision": 0.0, "recall": 0.0
                }
            else:
                # Multiple samples - use standard calculations
                correlation, p_value = pearsonr(y_true, y_pred)
                print(f"DEBUG: Correlation={correlation}, p_value={p_value}")
                
                try:
                    qwk = cohen_kappa_score(y_true.round(), y_pred.round(), weights="quadratic")
                    print(f"DEBUG: QWK={qwk}")
                except Exception as e:
                    print(f"DEBUG: QWK calculation failed: {e}")
                    qwk = 0.0
                
                try:
                    y_true_class = y_true.round().astype(int)
                    y_pred_class = y_pred.round().astype(int)
                    f1 = f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                    precision = precision_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                    recall = recall_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                    print(f"DEBUG: F1={f1}, Precision={precision}, Recall={recall}")
                except Exception as e:
                    print(f"DEBUG: Classification metrics failed: {e}")
                    f1 = precision = recall = 0.0
            
            metrics = {
                "quadratic_weighted_kappa": float(qwk),
                "pearson_correlation": float(correlation),
                "mean_absolute_error": float(mae),
                "mean_squared_error": float(mse), 
                "root_mean_squared_error": float(rmse),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall)
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

            # Debug logging
            print(f"DEBUG: Before conversion - y_pred[0]: '{y_pred[0]}', type: {type(y_pred[0])}")
            print(f"DEBUG: Before conversion - y_true[0]: '{y_true[0]}', type: {type(y_true[0])}")

            # Convert string labels to numbers for BEEtlE and SciEntSBank datasets
            # Convert string labels to numbers for BEEtlE and SciEntSBank datasets
            if dataset_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]:
                
                # Debug: Check actual labels before conversion
                print(f"DEBUG: Unique prediction labels: {pd.Series(y_pred).unique()}")
                print(f"DEBUG: Unique ground truth labels: {pd.Series(y_true).unique()}")
                print(f"DEBUG: Sample predictions: {y_pred[:10] if len(y_pred) >= 10 else y_pred}")
                print(f"DEBUG: Sample ground truth: {y_true[:10] if len(y_true) >= 10 else y_true}")
                
                # Define label mappings with variations for robustness
                if "3way" in dataset_name:
                    label_map = {
                        'correct': 2, 'Correct': 2, 'CORRECT': 2,
                        'incorrect': 0, 'Incorrect': 0, 'INCORRECT': 0,
                        'partial_correct': 1, 'Partial_correct': 1, 'PARTIAL_CORRECT': 1, 'partial correct': 1,
                        'contradictory': 1, 'Contradictory': 1, 'CONTRADICTORY': 1,
                        # Add numeric versions if needed
                        '2': 2, '1': 1, '0': 0, 2: 2, 1: 1, 0: 0
                    }
                else:  # 2way
                    label_map = {
                        'correct': 1, 'Correct': 1, 'CORRECT': 1,
                        'incorrect': 0, 'Incorrect': 0, 'INCORRECT': 0,
                        # Add numeric versions if needed  
                        '1': 1, '0': 0, 1: 1, 0: 0,
                        # Add other possible variations
                        'True': 1, 'False': 0, 'true': 1, 'false': 0
                    }
                
                # Convert to numeric if they're strings
                if len(y_pred) > 0 and isinstance(y_pred[0], str):
                    y_pred_series = pd.Series(y_pred).str.strip()  # Remove whitespace
                    y_pred = y_pred_series.map(label_map).fillna(-999).astype(int).to_numpy()
                    if -999 in y_pred:
                        unmapped = y_pred_series[y_pred == -999].unique()
                        print(f"WARNING: Unmapped prediction labels: {unmapped}")
                    y_pred = np.where(y_pred == -999, 0, y_pred)  # Default unmapped to 0
                    
                if len(y_true) > 0 and isinstance(y_true[0], str):
                    y_true_series = pd.Series(y_true).str.strip()  # Remove whitespace
                    y_true = y_true_series.map(label_map).fillna(-999).astype(int).to_numpy()
                    if -999 in y_true:
                        unmapped = y_true_series[y_true == -999].unique()
                        print(f"WARNING: Unmapped ground truth labels: {unmapped}")
                    y_true = np.where(y_true == -999, 0, y_true)  # Default unmapped to 0

            print(f"DEBUG: After conversion - y_pred[:5]: {y_pred[:5] if len(y_pred) >= 5 else y_pred}")
            print(f"DEBUG: After conversion - y_true[:5]: {y_true[:5] if len(y_true) >= 5 else y_true}")
            print(f"DEBUG: Unique pred values: {np.unique(y_pred)}, Unique true values: {np.unique(y_true)}")

            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred)
            
            # Get the appropriate ID column for this dataset
            id_column = self.get_id_column(dataset_name)
            
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
                    "id_column": id_column,
                    "score_range_pred": [float(np.nanmin(y_pred)), float(np.nanmax(y_pred))] if len(y_pred) > 0 else [0, 0],
                    "score_range_true": [float(np.nanmin(y_true)), float(np.nanmax(y_true))] if len(y_true) > 0 else [0, 0],
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
                "dataset": dataset_name
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
                          validation_type="real", evaluation_engine="RealEvaluationEngine",
                          has_real_evaluation=True, ground_truth_source="private_huggingface"):
    try:
        db = get_database()
        
        submission = OutputSubmission(
            dataset_name=dataset_name,
            submitter_name=model_name,
            submitter_email=contact_email,
            file_path=f"uploads/{dataset_name}_{model_name}_{filename}",
            file_format="CSV",
            status="completed",
            description=description
        )
        
        evaluation_data = {
            "real_evaluation": {
                "status": "success",
                "metrics": metrics,
                "evaluation_engine": evaluation_engine,
                "ground_truth_source": ground_truth_source
            },
            "validation_type": validation_type,
            "has_real_evaluation": has_real_evaluation
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

@router.get("/format/{dataset_name}")
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

@router.get("/available-datasets")
async def get_available_datasets():
    """Get list of all available datasets"""
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

@router.post("/upload-single")
async def upload_single_submission(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    model_name: Optional[str] = Form(None),
    description: str = Form("Individual dataset test"),
    institution: str = Form(""),
    contact_email: str = Form("")
):
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
        
        df = pd.read_csv(io.StringIO(csv_content))
        df_clean = clean_dataframe_safe(df)
        
        validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df_clean, testing_mode=False)
        
        if not validation_result["valid"]:
            return clean_for_json({
                "success": False,
                "validation_errors": validation_result["errors"],
                "validation_warnings": validation_result.get("warnings", []),
                "dataset": dataset_name,
                "filename": file.filename
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
                "filename": file.filename
            })
        
        db_result = store_submission_in_db(
            dataset_name=dataset_name,
            model_name=model_name,
            metrics=evaluation_result["metrics"],
            description=description,
            institution=institution,
            contact_email=contact_email,
            filename=file.filename
        )
        
        return clean_for_json({
            "success": True,
            "evaluation": evaluation_result,
            "database": db_result,
            "dataset": dataset_name,
            "model_name": model_name,
            "filename": file.filename,
            "encoding_used": encoding_used,
            "note": "REAL EVALUATION: Metrics calculated from actual ground truth comparisons"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        return clean_for_json({
            "success": False,
            "error": str(e),
            "dataset": dataset_name,
            "filename": file.filename if file else "unknown"
        })

@router.post("/upload-batch")
async def upload_batch_submissions(
    files: List[UploadFile] = File(...),
    dataset_names: List[str] = Form(...),
    model_name: str = Form(...),
    description: str = Form(""),
    institution: str = Form(""),
    contact_email: str = Form("")
):
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
            csv_content, encoding_used = decode_file_content(content)
            
            if csv_content is None:
                results.append({
                    "filename": file.filename,
                    "dataset": dataset_name,
                    "success": False,
                    "error": "Invalid file encoding"
                })
                failed_uploads += 1
                continue

            df = pd.read_csv(io.StringIO(csv_content))
            df_clean = clean_dataframe_safe(df)
            
            validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df_clean, testing_mode=False)
            
            if not validation_result["valid"]:
                results.append({
                    "filename": file.filename,
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
                    "filename": file.filename,
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
                filename=file.filename
            )

            results.append({
                "filename": file.filename,
                "dataset": dataset_name,
                "success": True,
                "evaluation": evaluation_result,
                "database": db_result,
                "encoding_used": encoding_used
            })
            successful_uploads += 1

        except Exception as e:
            results.append({
                "filename": getattr(file, "filename", "unknown"),
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

@router.post("/test-single-dataset")
async def test_single_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...)
):
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
        
        response_data = {
            "success": True,
            "testing_mode": True,
            "dataset": dataset_name,
            "filename": file.filename,
            "evaluation": evaluation_result,
            "metrics": evaluation_result.get("metrics", {}),
            "encoding_used": encoding_used,
            "validation_warnings": validation_result.get("warnings", []),
            "note": "TESTING MODE: Invalid labels assigned fallback values (4=invalid, 5=missing). Results are temporary and not stored."
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

@router.get("/leaderboard")
async def get_leaderboard(
    dataset: str = "All Datasets",
    metric: str = "avg_quadratic_weighted_kappa",
    limit: int = 20
):
    """Get leaderboard with real evaluation results"""
    try:
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
                "note": "No real evaluations found"
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
        
        rankings = []
        for model_name, data in model_aggregates.items():
            avg_metrics = {}
            for metric_name, values in data['metrics'].items():
                if values:
                    avg_metrics[f"avg_{metric_name}"] = statistics.mean(values)
            
            rankings.append({
                "model_name": model_name,
                "institution": data['institution'],
                "description": data['description'],
                "contact_email": data['contact_email'],
                "datasets_evaluated": list(set(data['datasets'])),
                "total_submissions": data['total_submissions'],
                "complete_benchmark": len(set(data['datasets'])) >= 20,
                **avg_metrics
            })
        
        if metric == "avg_mae":
            rankings.sort(key=lambda x: x.get(metric, float('inf')))
        else:
            rankings.sort(key=lambda x: x.get(metric, 0), reverse=True)
        
        rankings = rankings[:limit]
        
        return clean_for_json({
            "dataset": dataset,
            "metric": metric,
            "total_entries": len(rankings),
            "last_updated": current_time,
            "rankings": rankings,
            "note": "REAL EVALUATION: All metrics calculated from actual ground truth comparisons"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate leaderboard: {str(e)}")

@router.get("/health")
async def health_check():
    return clean_for_json({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0 - Simplified ID-based matching",
        "supported_datasets": 24,
        "evaluation_type": "REAL_GROUND_TRUTH",
        "matching_method": "ID_based_only"
    })

@router.get("/stats")
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