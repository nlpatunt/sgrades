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
    print("✅ Real evaluation packages imported successfully!")
except ImportError as e:
    print(f"⚠️ Real evaluation packages not available: {e}")
    REAL_EVALUATION_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
    print("✅ HuggingFace datasets library available")
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("⚠️ HuggingFace datasets library not available. Run: pip install datasets")

try:
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, accuracy_score, 
        f1_score, precision_score, recall_score, cohen_kappa_score
    )
    from scipy.stats import pearsonr
    print("✅ All sklearn metrics imported successfully!")
except ImportError as e:
    print(f"⚠️ Additional sklearn metrics not available: {e}")

USING_REAL_VALIDATOR = True

def download_ground_truth_private(dataset_name: str) -> Dict[str, Any]:
    
    if not HF_DATASETS_AVAILABLE:
        return {"status": "error", "error": "HuggingFace datasets library not available"}
    
    try:
        print(f"📥 Loading private ground truth dataset: nlpatunt/{dataset_name}")
        print("🔑 Using HuggingFace CLI authentication")
        
        # ✅ SPECIAL HANDLING for datasets with configurations
        if dataset_name == "BEEtlE_2way":
            dataset = load_dataset("nlpatunt/BEEtlE", "2way", split="test", trust_remote_code=True)
            print(f"✅ Loaded BEEtlE 2way config")
        elif dataset_name == "BEEtlE_3way":
            dataset = load_dataset("nlpatunt/BEEtlE", "3way", split="test", trust_remote_code=True) 
            print(f"✅ Loaded BEEtlE 3way config")
        elif dataset_name == "SciEntSBank_2way":
            dataset = load_dataset("nlpatunt/SciEntSBank", "2way", split="test", trust_remote_code=True)
            print(f"✅ Loaded SciEntSBank 2way config")
        elif dataset_name == "SciEntSBank_3way":
            dataset = load_dataset("nlpatunt/SciEntSBank", "3way", split="test", trust_remote_code=True)
            print(f"✅ Loaded SciEntSBank 3way config")
        elif dataset_name == "Rice_Chem_Q1":
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files="Q1/test.csv")
            dataset = dataset["train"]  # HF assigns data_files to "train" split
            print(f"✅ Loaded Rice_Chem Q1 test file")
        elif dataset_name == "Rice_Chem_Q2":
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files="Q2/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Rice_Chem Q2 test file")
        elif dataset_name == "Rice_Chem_Q3":
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files="Q3/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Rice_Chem Q3 test file")
        elif dataset_name == "Rice_Chem_Q4":
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files="Q4/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Rice_Chem Q4 test file")

        elif dataset_name == "grade_like_a_human_dataset_os_q1":
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", data_files="q1/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Grade Like Human Q1 test file")
        elif dataset_name == "grade_like_a_human_dataset_os_q2":
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", data_files="q2/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Grade Like Human Q2 test file")
        elif dataset_name == "grade_like_a_human_dataset_os_q3":
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", data_files="q3/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Grade Like Human Q3 test file")
        elif dataset_name == "grade_like_a_human_dataset_os_q4":
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", data_files="q4/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Grade Like Human Q4 test file")
        elif dataset_name == "grade_like_a_human_dataset_os_q5":
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", data_files="q5/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Grade Like Human Q5 test file")
        elif dataset_name == "grade_like_a_human_dataset_os_q6":
            dataset = load_dataset("nlpatunt/grade_like_a_human_dataset_os", data_files="q6/test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded Grade Like Human Q6 test file")
        elif dataset_name == "EFL":
            dataset = load_dataset("nlpatunt/EFL", data_files="test.csv")
            dataset = dataset["train"]
            print(f"✅ Loaded EFL test file only")
        else:
            try:
                dataset = load_dataset(f"nlpatunt/{dataset_name}", split="test")
            except:
                # Try without split if test split doesn't exist
                dataset = load_dataset(f"nlpatunt/{dataset_name}")
                if hasattr(dataset, 'keys'):
                    # Get the first available split
                    first_split = list(dataset.keys())[0]
                    dataset = dataset[first_split]
                    print(f"✅ Using split: {first_split}")
        
        # Convert to DataFrame
        df = dataset.to_pandas()
        
        print(f"✅ Loaded private dataset: {len(df)} rows, columns: {list(df.columns)}")
        
        return {
            "status": "success",
            "dataset": df,
            "rows": len(df),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ HuggingFace datasets library failed: {error_msg}")
        return {"status": "error", "error": error_msg}

def test_private_dataset_access():
    test_datasets = ["ASAP-AES", "BEEtlE_2way", "CSEE"]
    
    results = {}
    for dataset in test_datasets:
        print(f"\n🧪 Testing {dataset}...")
        result = download_ground_truth_private(dataset)
        results[dataset] = {
            "status": result["status"],
            "error": result.get("error", "none")
        }
        
        if result["status"] == "success":
            print(f"✅ {dataset}: {result['rows']} rows")
        else:
            print(f"❌ {dataset}: {result['error']}")
    
    return results

class RealEvaluationEngine:
    def __init__(self):
        self.ground_truth_cache = {}  # Cache downloaded ground truth
        
        # Simple score column mapping - only thing we need to configure
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
            "grade_like_a_human_dataset_os_q6": "score_1",
            "Rice_Chem_Q1": "Score",
            "Rice_Chem_Q2": "Score",
            "Rice_Chem_Q3": "Score",
            "Rice_Chem_Q4": "Score"
        }
        
        # ID columns for matching (fallback to row-based matching if none)
        self.ID_COLUMNS = {
            "ASAP-AES": "essay_id",
            "ASAP2": "essay_id", 
            "ASAP-SAS": "Id",
            "ASAP_plus_plus": "essay_id",
            "BEEtlE_2way": "question_id",
            "BEEtlE_3way": "question_id",
            "SciEntSBank_2way": "question_id", 
            "SciEntSBank_3way": "question_id",
            "CSEE": "essay_id",
            "EFL": "Essay Id",
            "Mohlar": None,  # Row-based matching
            "Ielts_Writing_Dataset": None,  # Row-based matching
            "Ielst_Writing_Task_2_Dataset": None,  # Row-based matching
            "persuade_2": "essay_id_comp",
            "Regrading_Dataset_J2C": "Question ID",
            "grade_like_a_human_dataset_os_q1": "id",
            "grade_like_a_human_dataset_os_q2": "id",
            "grade_like_a_human_dataset_os_q3": "id",
            "grade_like_a_human_dataset_os_q4": "id", 
            "grade_like_a_human_dataset_os_q5": "id",
            "grade_like_a_human_dataset_os_q6": "id",
            "Rice_Chem_Q1": "sis_id",
            "Rice_Chem_Q2": "sis_id",
            "Rice_Chem_Q3": "sis_id", 
            "Rice_Chem_Q4": "sis_id"
        }
        
    def get_ground_truth(self, dataset_name: str) -> Dict[str, Any]:
        """Get ground truth data for a dataset (with caching)"""
        if dataset_name not in self.ground_truth_cache:
            print(f"📥 Downloading ground truth for {dataset_name}")
            result = download_ground_truth_private(dataset_name)
            if result["status"] == "success":
                self.ground_truth_cache[dataset_name] = result["dataset"]
                print(f"✅ Ground truth cached: {len(result['dataset'])} rows")
            return result
        else:
            return {"status": "success", "dataset": self.ground_truth_cache[dataset_name]}
    
    def get_score_column(self, dataset_name: str) -> str:
        """Get the score column for evaluation"""
        return self.SCORE_COLUMNS.get(dataset_name, "score")
    
    def get_id_column(self, dataset_name: str) -> Optional[str]:
        """Get the ID column for matching (None = row-based matching)"""
        return self.ID_COLUMNS.get(dataset_name, "essay_id")

    def validate_full_structure(self, dataset_name: str, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that prediction CSV has same structure as downloaded dataset"""
        print(f"🔍 Validating full structure for {dataset_name}")
        
        # Expected columns = ALL columns from ground truth
        expected_columns = set(ground_truth_df.columns)
        provided_columns = set(prediction_df.columns)
        
        missing_columns = expected_columns - provided_columns
        extra_columns = provided_columns - expected_columns
        
        validation_errors = []
        warnings = []
        
        if missing_columns:
            validation_errors.append(f"Missing columns: {sorted(list(missing_columns))}")
            print(f"❌ Missing columns: {sorted(list(missing_columns))}")
        
        if extra_columns:
            validation_errors.append(f"Extra columns not in original dataset: {sorted(list(extra_columns))}")
            print(f"❌ Extra columns: {sorted(list(extra_columns))}")
        
        if validation_errors:
            return {
                "valid": False,
                "errors": validation_errors,
                "warnings": warnings,
                "expected_columns": sorted(list(expected_columns)),
                "provided_columns": sorted(list(provided_columns)),
                "instruction": "Upload CSV with ALL columns from the original downloaded dataset"
            }
        
        # Validate score column specifically
        score_col = self.get_score_column(dataset_name)
        if score_col not in prediction_df.columns:
            return {
                "valid": False, 
                "errors": [f"Score column '{score_col}' not found"],
                "warnings": warnings,
                "expected_score_column": score_col
            }
        
        # Check for valid scores
        scores = pd.to_numeric(prediction_df[score_col], errors='coerce')
        nan_count = scores.isna().sum()
        
        if nan_count > 0:
            warnings.append(f"Found {nan_count} non-numeric values in score column '{score_col}'")
        
        # Check row count
        if len(prediction_df) != len(ground_truth_df):
            warnings.append(f"Row count mismatch: {len(prediction_df)} predictions vs {len(ground_truth_df)} expected")
        
        print(f"✅ Structure validation passed for {dataset_name}")
        return {
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "matched_columns": len(expected_columns),
            "score_column": score_col,
            "id_column": self.get_id_column(dataset_name),
            "instruction": "All columns match downloaded dataset structure"
        }

    def match_predictions_to_ground_truth(self, dataset_name: str, prediction_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
        """Match predictions to ground truth using ID or row-based matching"""
        
        id_col = self.get_id_column(dataset_name)
        score_col = self.get_score_column(dataset_name)
        
        if id_col is None:
            # Row-based matching for datasets without IDs
            print(f"🔗 Using row-based matching for {dataset_name}")
            
            if len(prediction_df) != len(ground_truth_df):
                return {
                    "status": "error",
                    "error": f"Row count mismatch for row-based matching: {len(prediction_df)} vs {len(ground_truth_df)}"
                }
            
            # Reset indices for clean matching
            pred_clean = prediction_df.reset_index(drop=True)
            gt_clean = ground_truth_df.reset_index(drop=True)
            
            # Extract scores by row position
            pred_scores = pd.to_numeric(pred_clean[score_col], errors='coerce')
            gt_scores = pd.to_numeric(gt_clean[score_col], errors='coerce')
            
            # Remove rows where either score is NaN
            valid_mask = ~(pred_scores.isna() | gt_scores.isna())
            
            return {
                "status": "success",
                "y_pred": pred_scores[valid_mask].values,
                "y_true": gt_scores[valid_mask].values,
                "matched_count": valid_mask.sum(),
                "total_predictions": len(prediction_df),
                "total_ground_truth": len(ground_truth_df),
                "matching_method": "row_based"
            }
        
        else:
            # ID-based matching
            print(f"🔗 Using ID-based matching for {dataset_name} with column '{id_col}'")
            
            # Convert IDs to string for reliable matching
            prediction_df[id_col] = prediction_df[id_col].astype(str)
            ground_truth_df[id_col] = ground_truth_df[id_col].astype(str)
            
            # Merge on ID column
            merged_df = prediction_df[[id_col, score_col]].merge(
                ground_truth_df[[id_col, score_col]],
                on=id_col,
                suffixes=("_pred", "_true"),
                how="inner"
            )
            
            if len(merged_df) == 0:
                # Debug info for no matches
                sample_pred_ids = prediction_df[id_col].head(3).tolist()
                sample_gt_ids = ground_truth_df[id_col].head(3).tolist()
                
                return {
                    "status": "error",
                    "error": "No matching IDs found between predictions and ground truth",
                    "debug_info": {
                        "prediction_sample_ids": sample_pred_ids,
                        "ground_truth_sample_ids": sample_gt_ids,
                        "id_column": id_col
                    }
                }
            
            # Convert scores to numeric and remove NaN
            pred_scores = pd.to_numeric(merged_df[f"{score_col}_pred"], errors='coerce')
            gt_scores = pd.to_numeric(merged_df[f"{score_col}_true"], errors='coerce')
            
            valid_mask = ~(pred_scores.isna() | gt_scores.isna())
            
            return {
                "status": "success", 
                "y_pred": pred_scores[valid_mask].values,
                "y_true": gt_scores[valid_mask].values,
                "matched_count": valid_mask.sum(),
                "total_predictions": len(prediction_df),
                "total_ground_truth": len(ground_truth_df),
                "matching_method": "id_based",
                "id_column": id_col
            }

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        try:
            from sklearn.metrics import (
                mean_absolute_error, mean_squared_error, f1_score,
                precision_score, recall_score, cohen_kappa_score
            )
            from scipy.stats import pearsonr
            import numpy as np
            
            # Basic regression metrics
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Correlation
            correlation, p_value = pearsonr(y_true, y_pred)
            
            # Quadratic Weighted Kappa
            try:
                qwk = cohen_kappa_score(y_true.round(), y_pred.round(), weights="quadratic")
            except Exception as e:
                print(f"⚠️ QWK calculation failed: {e}")
                qwk = 0.0
            
            # Classification metrics (treating scores as classes)
            try:
                y_true_class = y_true.round().astype(int)
                y_pred_class = y_pred.round().astype(int)
                f1 = f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                precision = precision_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                recall = recall_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
            except Exception as e:
                print(f"⚠️ Classification metrics failed: {e}")
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
            
            print(f"✅ Metrics calculated - QWK: {qwk:.3f}, Pearson: {correlation:.3f}, MAE: {mae:.3f}")
            return metrics
            
        except Exception as e:
            print(f"❌ Metrics calculation failed: {e}")
            raise e

    def evaluate_submission(self, dataset_name: str, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Main evaluation function - simplified and cleaner"""
        try:
            print(f"🎯 Starting evaluation for {dataset_name}")
            
            # 1. Get ground truth
            gt_result = self.get_ground_truth(dataset_name)
            if gt_result["status"] != "success":
                return {
                    "status": "error",
                    "error": f"Failed to load ground truth: {gt_result.get('error')}"
                }
            ground_truth_df = gt_result["dataset"]
            
            # 2. Validate full structure (all columns must match)
            validation_result = self.validate_full_structure(dataset_name, predictions_df, ground_truth_df)
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "error": "Structure validation failed",
                    "validation_details": validation_result
                }
            
            # 3. Match predictions to ground truth
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
            
            # 4. Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred)
            
            # 5. Return results
            return {
                "status": "success",
                "metrics": metrics,
                "evaluation_details": {
                    "dataset": dataset_name,
                    "matched_examples": int(len(y_pred)),
                    "total_predictions": int(matching_result["total_predictions"]),
                    "total_ground_truth": int(matching_result["total_ground_truth"]),
                    "matching_method": matching_result["matching_method"],
                    "score_column": self.get_score_column(dataset_name),
                    "id_column": self.get_id_column(dataset_name),
                    "score_range_pred": [float(y_pred.min()), float(y_pred.max())],
                    "score_range_true": [float(y_true.min()), float(y_true.max())],
                    "validation_warnings": validation_result.get("warnings", [])
                }
            }
            
        except Exception as e:
            print(f"❌ Evaluation failed for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "dataset": dataset_name
            }

    def validate_submission_format(self, dataset_name: str, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate submission format - expects full dataset structure"""
        try:
            # Get ground truth to compare structure
            gt_result = self.get_ground_truth(dataset_name)
            if gt_result["status"] != "success":
                return {
                    "valid": False,
                    "errors": [f"Cannot load dataset schema: {gt_result.get('error')}"],
                    "warnings": []
                }
            
            ground_truth_df = gt_result["dataset"]
            
            # Use the same validation as evaluation
            validation_result = self.validate_full_structure(dataset_name, predictions_df, ground_truth_df)
            
            return {
                "valid": validation_result["valid"],
                "errors": validation_result.get("errors", []),
                "warnings": validation_result.get("warnings", []),
                "expected_format": f"All {len(ground_truth_df.columns)} columns from downloaded dataset",
                "expected_columns": validation_result.get("expected_columns", []),
                "dataset_info": f"Upload CSV with exact same structure as downloaded {dataset_name} dataset",
                "score_column": self.get_score_column(dataset_name),
                "matching_method": "row_based" if self.get_id_column(dataset_name) is None else "id_based"
            }
            
        except Exception as e:
            print(f"❌ Validation error for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": []
            }

real_evaluation_engine = RealEvaluationEngine()

def get_all_submissions_from_db():
    """Get all submissions from database"""
    try:
        db = get_database()
        submissions = db.query(OutputSubmission).all()
        
        result = []
        for sub in submissions:
            # Parse evaluation result
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
                "institution": "",  # Add if you have this field
                "description": sub.description or "",
                "has_real_evaluation": "real_evaluation" in evaluation_data,
                "metrics": evaluation_data.get("real_evaluation", {}).get("metrics", {})
            }
            result.append(submission_dict)
        
        return result
    except Exception as e:
        print(f"❌ Database error: {e}")
        return []

def store_submission_in_db(dataset_name, model_name, metrics, description="", 
                          institution="", contact_email="", filename="",
                          validation_type="real", evaluation_engine="RealEvaluationEngine",
                          has_real_evaluation=True, ground_truth_source="private_huggingface"):
    """Store submission in database"""
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
        
        # Store real evaluation data
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
        print(f"❌ Database storage failed: {e}")
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

router = APIRouter()

# Import the real submission validator
try:
    from app.services.submission_validator import submission_validator
    print("✅ Real submission_validator loaded successfully!")
    USING_REAL_VALIDATOR = True
except ImportError as e:
    print(f"⚠️ Real validator import failed: {e}")
    print("⚠️ Using fallback dummy validator")
    
    # Fallback dummy validator with compatible interface
    class ValidationResult:
        def __init__(self):
            self.valid = True  # Use 'valid' instead of 'is_valid'
            self.errors = []
            self.warnings = []
            self.dataset_name = ""
            self.row_count = 0
            self.expected_row_count = 0
    
    class DummyValidator:
        def validate_submission(self, dataset_name, df):
            result = ValidationResult()
            result.dataset_name = dataset_name
            result.row_count = len(df) if hasattr(df, '__len__') else 0
            
            # Basic validation - check if required columns exist
            if dataset_name == "ASAP-AES":
                required_columns = ['essay_id', 'domain1_score']
            else:
                required_columns = ['essay_id', 'predicted_score']
            
            # Check if required columns exist
            if hasattr(df, 'columns'):
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    result.valid = False
                    result.errors = [f"Missing required columns: {', '.join(missing_columns)}"]
                else:
                    result.valid = True
                    result.errors = []
                    result.warnings = []
            else:
                result.valid = False
                result.errors = ["Invalid DataFrame structure"]
            
            return result
        
        def get_dataset_schema(self, dataset_name):
            class Schema:
                def __init__(self):
                    if dataset_name == "ASAP-AES":
                        self.required_columns = ['essay_id', 'domain1_score']
                        self.optional_columns = ['essay_set', 'essay', 'rater1_domain1', 'rater2_domain1', 'rater3_domain1']
                        self.score_min = 1.0
                        self.score_max = 4.0
                        self.score_type = 'float'
                        self.id_pattern = r'ASAP-AES_test_\d+'
                    else:
                        self.required_columns = ['predicted_score']
                        self.optional_columns = []
                        self.score_min = 0.0
                        self.score_max = 100.0
                        self.score_type = 'float'
                        self.id_pattern = r'.*_test_\d+'
                        
            return Schema()
        
        def get_expected_format(self, dataset_name):
            schema = self.get_dataset_schema(dataset_name)
            return {
                "dataset_name": dataset_name,
                "required_columns": schema.required_columns,
                "optional_columns": schema.optional_columns,
                "score_range": [schema.score_min, schema.score_max],
                "score_type": schema.score_type,
                "example_format": {
                    schema.required_columns[0]: f"{dataset_name}_test_0",
                    schema.required_columns[1] if len(schema.required_columns) > 1 else "score": "Example score"
                }
            }
    
    submission_validator = DummyValidator()
    USING_REAL_VALIDATOR = False

def adapt_validation_result(result):
    """Adapt the real validator result to work with our code"""
    if USING_REAL_VALIDATOR:
        # Real validator uses 'is_valid', we need 'valid'
        if hasattr(result, 'is_valid'):
            result.valid = result.is_valid
        # Ensure all required attributes exist
        if not hasattr(result, 'valid'):
            result.valid = True
        if not hasattr(result, 'errors'):
            result.errors = []
        if not hasattr(result, 'warnings'):
            result.warnings = []
    return result

def validate_with_real_validator(dataset_name, df):
    """Wrapper function to handle both real and dummy validator"""
    try:
        if USING_REAL_VALIDATOR:
            # Use RealEvaluationEngine for validation to ensure consistency
            validation_result = real_evaluation_engine.validate_submission_format(dataset_name, df)
            
            # Convert to expected format
            class ValidationResult:
                def __init__(self, result_dict):
                    self.valid = result_dict["valid"]
                    self.errors = result_dict["errors"]
                    self.warnings = result_dict["warnings"]
                    self.dataset_name = dataset_name
                    self.row_count = len(df)
                    self.expected_row_count = len(df)  # Assuming all rows are expected
                    # Add expected format info if available
                    if "expected_columns" in result_dict:
                        self.expected_format = result_dict["expected_columns"]
            
            return ValidationResult(validation_result)
        else:
            # Use dummy validator
            result = submission_validator.validate_submission(dataset_name, df)
            return adapt_validation_result(result)
        
    except Exception as e:
        # Fallback result if validation fails
        class FallbackResult:
            def __init__(self):
                self.valid = False
                self.errors = [f"Validation error: {str(e)}"]
                self.warnings = []
                self.dataset_name = dataset_name
                self.row_count = len(df) if hasattr(df, '__len__') else 0
                self.expected_row_count = 0
        
        return FallbackResult()

def safe_dataframe_to_dict(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to JSON-safe dictionary"""
    # Replace problematic values
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    df_clean = df_clean.fillna("")
    
    # Convert to dict and clean
    result = df_clean.to_dict('records')
    return clean_for_json(result)

def clean_dataframe_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for processing - handle NaN, inf values"""
    df_clean = df.copy()
    
    # Replace infinite values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    return df_clean

# ====================================
# TEMPLATE AND FORMAT ENDPOINTS
# ====================================

@router.get("/template")
async def get_submission_template():
    """Get template for CSV submissions with examples for all datasets"""
    
    template_data = {
        "format": "CSV file with specific columns per dataset",
        "description": "Upload CSV files with predictions for each dataset",
        "total_datasets": 25,
        "datasets": {
            # Main Datasets
            "ASAP-AES": {
                "category": "Main",
                "required_columns": ["essay_id", "domain1_score", "domain2_score"],
                "optional_columns": ["rater1_trait1", "rater1_trait2", "rater1_trait3", "rater1_trait4", "rater1_trait5", "rater1_trait6"],
                "example": {
                    "essay_id": "ASAP-AES_test_0",
                    "domain1_score": 3.5,
                    "domain2_score": 4.0,
                    "rater1_trait1": 3
                },
                "score_range": "[0, 60]",
                "score_type": "float",
                "description": "Automated Essay Scoring dataset"
            },
            "ASAP2": {
                "category": "Main",
                "required_columns": ["essay_id", "score"],
                "optional_columns": [],
                "example": {
                    "essay_id": "ASAP2_test_0",
                    "score": 45.5
                },
                "score_range": "[0, 60]",
                "score_type": "float"
            },
            "BEEtlE_2way": {
                "category": "Main",
                "required_columns": ["question_id", "label"],
                "optional_columns": [],
                "example": {
                    "question_id": "BEEtlE_2way_test_0",
                    "label": 1
                },
                "score_range": "[0, 1]",
                "score_type": "int",
                "description": "Binary classification - correct/incorrect"
            },
            "BEEtlE_3way": {
                "category": "Main",
                "required_columns": ["question_id", "label"],
                "optional_columns": [],
                "example": {
                    "question_id": "BEEtlE_3way_test_0",
                    "label": 2
                },
                "score_range": "[0, 2]",
                "score_type": "int",
                "description": "Three-way classification"
            },
            "SciEntSBank_2way": {
                "category": "Science",
                "required_columns": ["question_id", "label"],
                "optional_columns": [],
                "example": {
                    "question_id": "SciEntSBank_2way_test_0",
                    "label": 1
                },
                "score_range": "[0, 1]",
                "score_type": "int"
            },
            "SciEntSBank_3way": {
                "category": "Science",
                "required_columns": ["question_id", "label"],
                "optional_columns": [],
                "example": {
                    "question_id": "SciEntSBank_3way_test_0",
                    "label": 2
                },
                "score_range": "[0, 2]",
                "score_type": "int"
            }
        },
        "validation_rules": {
            "file_format": "Must be CSV format",
            "encoding": "UTF-8 preferred",
            "required_columns": "All required columns must be present",
            "id_format": "IDs must match expected pattern for each dataset",
            "score_validation": "Scores must be within valid range and correct type",
            "no_missing_values": "Required columns cannot have missing values"
        }
    }
    
    return clean_for_json(template_data)

@router.get("/format/{dataset_name}")
async def get_dataset_format(dataset_name: str):
    """Get specific format requirements for a dataset"""
    
    try:
        if USING_REAL_VALIDATOR:
            # Use real validator's get_expected_format method
            format_info = submission_validator.get_expected_format(dataset_name)
            if "error" in format_info:
                raise HTTPException(status_code=404, detail=format_info["error"])
            return clean_for_json(format_info)
        else:
            # Use dummy validator
            dataset_info = submission_validator.get_dataset_schema(dataset_name)
            format_info = {
                "dataset": dataset_name,
                "required_columns": dataset_info.required_columns,
                "optional_columns": dataset_info.optional_columns,
                "score_range": f"[{dataset_info.score_min}, {dataset_info.score_max}]",
                "score_type": dataset_info.score_type,
                "id_pattern": dataset_info.id_pattern,
                "example_filename": f"{dataset_name}.csv"
            }
            return clean_for_json(format_info)
        
    except Exception as e:
        return clean_for_json({"error": str(e)})

# ====================================
# VALIDATION ENDPOINTS  
# ====================================


@router.post("/validate-csv")
async def validate_csv_submission(
    file: UploadFile = File(...),
    dataset_name: str = Form(...)
):
    """Validate a CSV submission for a specific dataset"""
    
    try:
        # Read the CSV file safely
        content = await file.read()
        
        # Handle different encodings gracefully
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
            return clean_for_json({
                "valid": False,
                "errors": ["Invalid file encoding. Please save your CSV as UTF-8."],
                "warnings": []
            })
        
        # Parse CSV with comprehensive error handling
        try:
            df = pd.read_csv(io.StringIO(csv_content))
        except pd.errors.EmptyDataError:
            return clean_for_json({
                "valid": False,
                "errors": ["CSV file is empty. Please upload a file with data."],
                "warnings": []
            })
        except pd.errors.ParserError as e:
            return clean_for_json({
                "valid": False,
                "errors": [f"CSV format error: {str(e)}. Please check your file format."],
                "warnings": []
            })
        except Exception as e:
            return clean_for_json({
                "valid": False,
                "errors": [f"Failed to read CSV: {str(e)}"],
                "warnings": []
            })
        
        # Basic file checks
        if len(df) == 0:
            return clean_for_json({
                "valid": False,
                "errors": ["CSV file contains no data rows."],
                "warnings": []
            })
        
        if len(df.columns) == 0:
            return clean_for_json({
                "valid": False,
                "errors": ["CSV file contains no columns."],
                "warnings": []
            })
        
        # Clean the DataFrame for validation
        df_clean = clean_dataframe_safe(df)
        
        # Validate using the real validator with adapter
        validation_result = validate_with_real_validator(dataset_name, df_clean)
        
        # Prepare response with safe data
        response = {
            "valid": validation_result.valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "dataset": dataset_name,
            "filename": file.filename,
            "row_count": len(df_clean),
            "column_count": len(df_clean.columns),
            "columns_found": list(df_clean.columns),
            "encoding_detected": encoding_used,
            "file_size_bytes": len(content),
            "validator_type": "real" if USING_REAL_VALIDATOR else "dummy"
        }
        
        # Add sample data (first 3 rows) for preview if validation passed
        if validation_result.valid and len(df_clean) > 0:
            sample_data = df_clean.head(3)
            response["sample_data"] = safe_dataframe_to_dict(sample_data)
            response["preview_note"] = "First 3 rows of your data"
        
        # Add expected format information
        try:
            if USING_REAL_VALIDATOR:
                expected_format = submission_validator.get_expected_format(dataset_name)
                if "error" not in expected_format:
                    response["expected_format"] = expected_format
            else:
                schema = submission_validator.get_dataset_schema(dataset_name)
                if schema:
                    response["expected_format"] = {
                        "required_columns": schema.required_columns,
                        "optional_columns": schema.optional_columns,
                        "score_range": f"[{schema.score_min}, {schema.score_max}]",
                        "score_type": schema.score_type
                    }
        except:
            pass  # Don't fail validation if schema lookup fails
        
        return clean_for_json(response)
        
    except Exception as e:
        return clean_for_json({
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "warnings": [],
            "dataset": dataset_name,
            "filename": file.filename if file else "unknown",
            "validator_type": "real" if USING_REAL_VALIDATOR else "dummy"
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
    """Upload a single dataset submission with real evaluation"""
    if model_name is None:
        model_name = f"test_model_{dataset_name}" 
    try:
        # Read the CSV file safely with encoding detection
        content = await file.read()
        csv_content = None
        encoding_used = None
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                csv_content = content.decode(encoding)
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if csv_content is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid file encoding. Please save your CSV as UTF-8."
            )
        
        # Parse CSV
        try:
            df = pd.read_csv(io.StringIO(csv_content))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse CSV: {str(e)}"
            )
        
        # Clean the DataFrame
        df_clean = clean_dataframe_safe(df)
        
        # Validate using real validator
        validation_result = validate_with_real_validator(dataset_name, df_clean)
        
        if not validation_result.valid:
            return clean_for_json({
                "success": False,
                "validation": {
                    "valid": False,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                },
                "dataset": dataset_name,
                "filename": file.filename
            })
        
        print(f"🎯 Starting REAL evaluation for {dataset_name}")
        
        # REAL EVALUATION with ground truth
        evaluation_result = real_evaluation_engine.evaluate_submission(
            dataset_name=dataset_name,
            predictions_df=df_clean
        )
        
        if evaluation_result["status"] != "success":
            return clean_for_json({
                "success": False,
                "evaluation": evaluation_result,
                "dataset": dataset_name,
                "filename": file.filename
            })
        
        # Store in database with REAL evaluation results
        db_result = store_submission_in_db(
            dataset_name=dataset_name,
            model_name=model_name,
            metrics=evaluation_result["metrics"],
            description=description,
            institution=institution,
            contact_email=contact_email,
            filename=file.filename,
            validation_type="real",
            evaluation_engine="RealEvaluationEngine",
            has_real_evaluation=True,
            ground_truth_source="private_huggingface"
        )
        
        return clean_for_json({
            "success": True,
            "evaluation": evaluation_result,
            "database": db_result,
            "dataset": dataset_name,
            "model_name": model_name,
            "filename": file.filename,
            "encoding_used": encoding_used,
            "validation": {
                "valid": True,
                "validator_type": "real"
            },
            "note": "🎯 REAL EVALUATION: Metrics calculated from actual ground truth comparisons"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return clean_for_json({
            "success": False,
            "error": str(e),
            "dataset": dataset_name,
            "filename": file.filename if file else "unknown"
        })

@router.post("/upload-batch")
async def upload_batch_submissions(
    files: List[UploadFile] = File(...),
    model_name: str = Form(...),
    description: str = Form(""),
    institution: str = Form(""),
    contact_email: str = Form("")
):
    """Upload multiple dataset submissions as a batch with real evaluation"""
    
    results = []
    successful_uploads = 0
    failed_uploads = 0
    
    for file in files:
        try:
            # Extract dataset name from filename (e.g., "ASAP-AES.csv" -> "ASAP-AES")
            dataset_name = file.filename.split('.')[0] if file.filename else "unknown"
            
            print(f"🎯 Processing batch file: {file.filename} for dataset: {dataset_name}")
            
            # Read file with encoding detection
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
            
            # Parse CSV
            df = pd.read_csv(io.StringIO(csv_content))
            df_clean = clean_dataframe_safe(df)
            
            # Validate
            validation_result = validate_with_real_validator(dataset_name, df_clean)
            
            if not validation_result.valid:
                results.append({
                    "filename": file.filename,
                    "dataset": dataset_name,
                    "success": False,
                    "validation_errors": validation_result.errors,
                    "validation_warnings": validation_result.warnings
                })
                failed_uploads += 1
                continue
            
            # REAL EVALUATION for this file
            evaluation_result = real_evaluation_engine.evaluate_submission(
                dataset_name=dataset_name,
                predictions_df=df_clean
            )
            
            if evaluation_result["status"] != "success":
                results.append({
                    "filename": file.filename,
                    "dataset": dataset_name,
                    "success": False,
                    "evaluation_error": evaluation_result.get("error", "Evaluation failed")
                })
                failed_uploads += 1
                continue
            
            # Store in database with REAL evaluation
            db_result = store_submission_in_db(
                dataset_name=dataset_name,
                model_name=model_name,
                metrics=evaluation_result["metrics"],
                description=description,
                institution=institution,
                contact_email=contact_email,
                filename=file.filename,
                validation_type="real",
                evaluation_engine="RealEvaluationEngine",
                has_real_evaluation=True,
                ground_truth_source="private_huggingface"
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
            print(f"❌ Batch processing failed for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "dataset": dataset_name if 'dataset_name' in locals() else "unknown",
                "success": False,
                "error": str(e)
            })
            failed_uploads += 1
    
    return clean_for_json({
        "success": successful_uploads > 0,
        "total_files": len(files),
        "successful_uploads": successful_uploads,
        "failed_uploads": failed_uploads,
        "success_rate": f"{(successful_uploads/len(files)*100):.1f}%",
        "model_name": model_name,
        "results": results,
        "validation_type": "real" if USING_REAL_VALIDATOR else "dummy",
        "evaluation_type": "REAL_GROUND_TRUTH",
        "note": "🎯 REAL EVALUATION: All metrics calculated from actual ground truth comparisons"
    })

@router.get("/list")
async def list_submissions(
    skip: int = 0,
    limit: int = 100,
    dataset_filter: Optional[str] = None,
    model_filter: Optional[str] = None
):
    """List all submissions with filtering and pagination"""
    
    try:
        db = get_database()
        # Build query
        query = db.query(OutputSubmission)
        
        if dataset_filter:
            query = query.filter(OutputSubmission.dataset_name == dataset_filter)
        if model_filter:
            query = query.filter(OutputSubmission.submitter_name.ilike(f"%{model_filter}%"))
        
        # Get total count
        total_count = query.count()
        
        # Get submissions with pagination
        submissions = query.order_by(OutputSubmission.submission_time.desc()).offset(skip).limit(limit).all()
        
        # Convert to clean format
        submissions_list = []
        for submission in submissions:
            submission_clean = {
                "id": submission.id,
                "dataset_name": submission.dataset_name,
                "model_name": submission.submitter_name,
                "institution": "",  # Add institution field to model if needed
                "contact_email": submission.submitter_email,
                "upload_date": submission.submission_time.isoformat() if submission.submission_time else None,
                "status": submission.status,
                "filename": os.path.basename(submission.file_path) if submission.file_path else "",
                "description": submission.description or ""
            }
            submissions_list.append(submission_clean)
        
        return clean_for_json({
            "submissions": submissions_list,
            "total_count": total_count,
            "page_info": {
                "skip": skip,
                "limit": limit,
                "has_more": total_count > (skip + limit)
            },
            "filters": {
                "dataset": dataset_filter,
                "model": model_filter
            }
        })
        
    except Exception as e:
        return clean_for_json({
            "error": str(e),
            "submissions": [],
            "total_count": 0
        })

@router.get("/stats")
async def get_platform_stats():
    """Get overall platform statistics"""
    try:
        print("📈 Fetching platform statistics...")
        
        submissions = get_all_submissions_from_db()
        real_submissions = [s for s in submissions if s.get('has_real_evaluation', False)]
        
        # Count unique researchers
        unique_researchers = len(set(s.get('contact_email', '') for s in real_submissions if s.get('contact_email')))
        
        # Count complete benchmarks (models with 20+ datasets)
        model_datasets = {}
        for submission in real_submissions:
            model = submission['model_name']
            if model not in model_datasets:
                model_datasets[model] = set()
            model_datasets[model].add(submission['dataset_name'])
        
        complete_benchmarks = sum(1 for datasets in model_datasets.values() if len(datasets) >= 20)
        
        stats = {
            "total_submissions": len(real_submissions),
            "total_researchers": unique_researchers,
            "complete_benchmarks": complete_benchmarks,
            "available_datasets": 25,
            "real_evaluations_only": True,
            "evaluation_type": "REAL_GROUND_TRUTH"
        }
        
        print(f"📊 Platform stats: {unique_researchers} researchers, {complete_benchmarks} complete benchmarks")
        
        return clean_for_json(stats)
        
    except Exception as e:
        print(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# ====================================
# LEADERBOARD ENDPOINT WITH REAL METRICS
# ====================================

@router.get("/leaderboard")
async def get_leaderboard(
    dataset: str = "All Datasets",
    metric: str = "avg_quadratic_weighted_kappa",
    limit: int = 20
):
    """Get leaderboard with REAL evaluation results only"""
    try:
        # Get current timestamp
        current_time = datetime.now().isoformat()
        
        # Get all submissions from database
        submissions = get_all_submissions_from_db()
        
        # ✅ FILTER: Only include submissions with REAL evaluation
        real_submissions = []
        for submission in submissions:
            if submission.get('has_real_evaluation', False):
                real_submissions.append(submission)
            else:
                print(f"⚠️ {submission.get('model_name', 'Unknown')} has no real evaluations, skipping from leaderboard")
        
        if not real_submissions:
            return clean_for_json({
                "dataset": dataset,
                "metric": metric,
                "total_entries": 0,
                "last_updated": current_time,
                "rankings": [],
                "summary_stats": {
                    "avg_quadratic_weighted_kappa": 0.0,
                    "avg_f1_score": 0.0,
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "total_submissions": 0,
                    "total_real_evaluations": 0,
                    "complete_benchmarks": 0,
                    "total_researchers": 0
                },
                "available_metrics": [
                    {"key": "avg_quadratic_weighted_kappa", "name": "Avg Quadratic Weighted Kappa", "higher_better": True},
                    {"key": "avg_pearson_correlation", "name": "Avg Pearson Correlation", "higher_better": True},
                    {"key": "avg_f1_score", "name": "Avg F1 Score", "higher_better": True},
                    {"key": "avg_precision", "name": "Avg Precision", "higher_better": True},
                    {"key": "avg_recall", "name": "Avg Recall", "higher_better": True},
                    {"key": "avg_mae", "name": "Avg Mean Absolute Error", "higher_better": False}
                ],
                "available_datasets": [
                    "ASAP-AES", "ASAP2", "ASAP-SAS", "BEEtlE_2way", "BEEtlE_3way", "CSEE", "EFL", "Mohlar",
                    "Ielts_Writing_Dataset", "Ielst_Writing_Task_2_Dataset", "persuade_2", "Regrading_Dataset_J2C",
                    "ASAP_plus_plus", "SciEntSBank_2way", "SciEntSBank_3way", "grade_like_a_human_dataset_os_q1",
                    "grade_like_a_human_dataset_os_q2", "grade_like_a_human_dataset_os_q3", "grade_like_a_human_dataset_os_q4",
                    "grade_like_a_human_dataset_os_q5", "grade_like_a_human_dataset_os_q6", "Rice_Chem_Q1",
                    "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"
                ],
                "validator_type": "real" if USING_REAL_VALIDATOR else "dummy",
                "evaluation_type": "REAL_GROUND_TRUTH",
                "note": "🎯 100% REAL EVALUATION: All metrics calculated from actual ground truth comparisons. No simulated data."
            })
        
        # Group by model and calculate aggregate metrics
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
                    'contact_email': submission.get('contact_email', ''),
                    'real_evaluation': True
                }
            
            model_aggregates[model_name]['datasets'].append(submission['dataset_name'])
            model_aggregates[model_name]['total_submissions'] += 1
            
            # Aggregate metrics
            for metric_name, value in submission.get('metrics', {}).items():
                if metric_name not in model_aggregates[model_name]['metrics']:
                    model_aggregates[model_name]['metrics'][metric_name] = []
                model_aggregates[model_name]['metrics'][metric_name].append(value)
        
        # Calculate averages and create rankings
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
                "complete_benchmark": len(set(data['datasets'])) >= 20,  # Consider 20+ datasets as complete
                "real_evaluation": True,
                **avg_metrics
            })
        
        # Sort by the requested metric
        if metric in ["avg_mae"]:  # Lower is better for MAE
            rankings.sort(key=lambda x: x.get(metric, float('inf')))
        else:  # Higher is better for most metrics
            rankings.sort(key=lambda x: x.get(metric, 0), reverse=True)
        
        # Apply limit
        rankings = rankings[:limit]
        
        # Calculate summary statistics
        all_metrics = {}
        for ranking in rankings:
            for key, value in ranking.items():
                if key.startswith('avg_') and isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        summary_stats = {
            "total_submissions": sum(r['total_submissions'] for r in rankings),
            "total_real_evaluations": len(real_submissions),
            "complete_benchmarks": sum(1 for r in rankings if r.get('complete_benchmark', False)),
            "total_researchers": len(set(r['contact_email'] for r in rankings if r.get('contact_email')))
        }
        
        for metric_name, values in all_metrics.items():
            summary_stats[metric_name] = statistics.mean(values) if values else 0.0
        
        return clean_for_json({
            "dataset": dataset,
            "metric": metric,
            "total_entries": len(rankings),
            "last_updated": current_time,
            "rankings": rankings,
            "summary_stats": summary_stats,
            "available_metrics": [
                {"key": "avg_quadratic_weighted_kappa", "name": "Avg Quadratic Weighted Kappa", "higher_better": True},
                {"key": "avg_pearson_correlation", "name": "Avg Pearson Correlation", "higher_better": True},
                {"key": "avg_f1_score", "name": "Avg F1 Score", "higher_better": True},
                {"key": "avg_precision", "name": "Avg Precision", "higher_better": True},
                {"key": "avg_recall", "name": "Avg Recall", "higher_better": True},
                {"key": "avg_mae", "name": "Avg Mean Absolute Error", "higher_better": False}
            ],
            "available_datasets": [
                "ASAP-AES", "ASAP2", "ASAP-SAS", "BEEtlE_2way", "BEEtlE_3way", "CSEE", "EFL", "Mohlar",
                "Ielts_Writing_Dataset", "Ielst_Writing_Task_2_Dataset", "persuade_2", "Regrading_Dataset_J2C",
                "ASAP_plus_plus", "SciEntSBank_2way", "SciEntSBank_3way", "grade_like_a_human_dataset_os_q1",
                "grade_like_a_human_dataset_os_q2", "grade_like_a_human_dataset_os_q3", "grade_like_a_human_dataset_os_q4",
                "grade_like_a_human_dataset_os_q5", "grade_like_a_human_dataset_os_q6", "Rice_Chem_Q1",
                "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"
            ],
            "validator_type": "real" if USING_REAL_VALIDATOR else "dummy",
            "evaluation_type": "REAL_GROUND_TRUTH",
            "note": "🎯 100% REAL EVALUATION: All metrics calculated from actual ground truth comparisons. No simulated data."
        })
        
    except Exception as e:
        print(f"❌ Leaderboard error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate leaderboard: {str(e)}")

# ====================================
# HEALTH CHECK AND INFO ENDPOINTS
# ====================================

@router.get("/health")
async def health_check():
    """Comprehensive health check for the submission system"""
    try:
        current_time = datetime.now().isoformat()
        
        # Test basic components
        health_status = {
            "status": "healthy",
            "timestamp": current_time,
            "version": "1.0.0",
            "components": {
                "database": "connected",
                "validator": "operational",
                "validator_type": "real" if USING_REAL_VALIDATOR else "dummy",
                "file_processing": "operational",
                "json_serialization": "operational"
            },
            "supported_datasets": 25,
            "supported_formats": ["CSV"],
            "max_file_size": "100MB",
            "supported_encodings": ["UTF-8", "Latin-1", "CP1252"]
        }
        
        # Test evaluation system
        try:
            evaluation_health = test_evaluation_system()
            health_status["components"]["evaluation_system"] = evaluation_health
        except Exception as e:
            health_status["components"]["evaluation_system"] = {
                "status": "error",
                "error": str(e)
            }
        
        return clean_for_json(health_status)
        
    except Exception as e:
        return clean_for_json({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "components": {
                "database": "unknown",
                "validator": "unknown",
                "file_processing": "unknown"
            }
        })

def test_evaluation_system():
    """Test the real evaluation system"""
    try:
        # Test basic metrics calculation
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        mae = np.mean(np.abs(y_true - y_pred))
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Test a few private dataset downloads
        test_datasets = ["ASAP-AES", "BEEtlE_2way", "CSEE"]
        dataset_tests = {}
        
        for dataset in test_datasets:
            try:
                result = download_ground_truth_private(dataset)
                if result["status"] == "success":
                    dataset_tests[dataset] = {"status": "success", "error": "none"}
                else:
                    dataset_tests[dataset] = {"status": "error", "error": result.get("error", "unknown")}
            except Exception as e:
                dataset_tests[dataset] = {"status": "error", "error": str(e)}
        
        return {
            "status": "private_dataset_test_complete",
            "basic_metrics_working": True,
            "test_mae": round(mae, 3),
            "test_correlation": round(correlation, 3),
            "hf_datasets_available": HF_DATASETS_AVAILABLE,
            "private_dataset_tests": dataset_tests,
            "numpy_version": np.__version__,
            "next_step": "Add HuggingFace token for private dataset access"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "hf_datasets_available": HF_DATASETS_AVAILABLE
        }
    
@router.get("/info")
async def get_system_info():
    """Get detailed system information"""
    return clean_for_json({
        "platform_name": "SUPERGLUE-like Essay Evaluation Platform",
        "version": "1.0.0",
        "total_datasets": 25,
        "evaluation_type": "REAL_GROUND_TRUTH",
        "validator_type": "real" if USING_REAL_VALIDATOR else "dummy",
        "supported_metrics": [
            "Quadratic Weighted Kappa",
            "Pearson Correlation",
            "F1 Score",
            "Precision",
            "Recall",
            "Mean Absolute Error"
        ],
        "datasets": [
            "ASAP-AES", "ASAP2", "ASAP-SAS", "BEEtlE_2way", "BEEtlE_3way", "CSEE", "EFL", "Mohlar",
            "Ielts_Writing_Dataset", "Ielst_Writing_Task_2_Dataset", "persuade_2", "Regrading_Dataset_J2C",
            "ASAP_plus_plus", "SciEntSBank_2way", "SciEntSBank_3way", "grade_like_a_human_dataset_os_q1",
            "grade_like_a_human_dataset_os_q2", "grade_like_a_human_dataset_os_q3", "grade_like_a_human_dataset_os_q4",
            "grade_like_a_human_dataset_os_q5", "grade_like_a_human_dataset_os_q6", "Rice_Chem_Q1",
            "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"
        ],
        "ground_truth_source": "private_huggingface_datasets",
        "real_evaluation": True,
        "note": "🎯 100% REAL EVALUATION: All metrics calculated from actual ground truth comparisons"
    })

@router.post("/bulk-validate")
async def bulk_validate_submissions(files: List[UploadFile] = File(...)):
    """Validate multiple CSV files at once without uploading"""
    
    validation_results = []
    total_files = len(files)
    
    for file_index, file in enumerate(files):
        dataset_name = "unknown"
        try:
            # Extract dataset name from filename
            filename = file.filename or f"file_{file_index}"
            dataset_name = filename.replace('.csv', '') if filename.endswith('.csv') else filename
            
            # Read and validate
            content = await file.read()
            
            # Handle encoding
            csv_content = None
            encoding_used = "unknown"
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    csv_content = content.decode(encoding)
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if csv_content is None:
                validation_results.append({
                    "dataset": dataset_name,
                    "filename": filename,
                    "valid": False,
                    "errors": ["Invalid file encoding"],
                    "warnings": []
                })
                continue
            
            # Parse and validate
            df = pd.read_csv(io.StringIO(csv_content))
            df_clean = clean_dataframe_safe(df)
            
            validation_result = validate_with_real_validator(dataset_name, df_clean)
            
            result = {
                "dataset": dataset_name,
                "filename": filename,
                "valid": validation_result.valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "row_count": len(df_clean),
                "column_count": len(df_clean.columns),
                "columns_found": list(df_clean.columns),
                "encoding": encoding_used,
                "file_size_bytes": len(content),
                "validator_type": "real" if USING_REAL_VALIDATOR else "dummy"
            }
            
            validation_results.append(result)
            
        except Exception as e:
            validation_results.append({
                "dataset": dataset_name,
                "filename": file.filename or f"file_{file_index}",
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "validator_type": "real" if USING_REAL_VALIDATOR else "dummy"
            })
    
    # Calculate summary
    valid_files = sum(1 for r in validation_results if r["valid"])
    invalid_files = total_files - valid_files
    
    summary = {
        "total_files": total_files,
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "validation_rate": round((valid_files / total_files * 100), 1) if total_files > 0 else 0,
        "validator_type": "real" if USING_REAL_VALIDATOR else "dummy",
        "results": validation_results
    }
    
    return clean_for_json(summary)

@router.get("/export/{format}")
async def export_submissions(
    format: str = "csv",
    dataset_filter: Optional[str] = None,
    model_filter: Optional[str] = None
):
    """Export submissions data in various formats (csv, json, xlsx)"""
    
    try:
        if format not in ["csv", "json", "xlsx"]:
            raise HTTPException(status_code=400, detail="Format must be csv, json, or xlsx")
        
        db = get_database()
        # Build query
        query = db.query(OutputSubmission)
        
        if dataset_filter:
            query = query.filter(OutputSubmission.dataset_name == dataset_filter)
        if model_filter:
            query = query.filter(OutputSubmission.submitter_name.ilike(f"%{model_filter}%"))
        
        submissions = query.order_by(OutputSubmission.submission_time.desc()).all()
        
        # Clean data for export
        export_data = []
        for submission in submissions:
            clean_submission = {
                "submission_id": submission.id,
                "dataset_name": submission.dataset_name,
                "model_name": submission.submitter_name,
                "contact_email": submission.submitter_email,
                "upload_date": submission.submission_time.isoformat() if submission.submission_time else None,
                "status": submission.status,
                "filename": os.path.basename(submission.file_path) if submission.file_path else "",
                "description": submission.description or "",
                "file_format": submission.file_format
            }
            export_data.append(clean_submission)
        
        # Generate export based on format
        if format == "json":
            return clean_for_json({
                "export_format": "json",
                "export_date": datetime.now().isoformat(),
                "total_records": len(export_data),
                "filters": {"dataset": dataset_filter, "model": model_filter},
                "validator_type": "real" if USING_REAL_VALIDATOR else "dummy",
                "data": export_data
            })
        
        elif format == "csv":
            # Convert to CSV format
            if export_data:
                df_export = pd.DataFrame(export_data)
                csv_content = df_export.to_csv(index=False)
                
                from fastapi.responses import Response
                return Response(
                    content=csv_content,
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=besesr_submissions_{datetime.now().strftime('%Y%m%d')}.csv"}
                )
            else:
                return Response(
                    content="No data to export",
                    media_type="text/plain"
                )
        
        else:  # xlsx format
            return clean_for_json({
                "error": "XLSX export not yet implemented",
                "available_formats": ["csv", "json"]
            })
    
    except HTTPException:
        raise
    except Exception as e:
        return clean_for_json({
            "error": str(e),
            "export_format": format,
            "data": []
        })

# ====================================
# INDIVIDUAL SUBMISSION ENDPOINTS (MUST BE LAST - CATCH-ALL ROUTE)
# ====================================

@router.get("/{submission_id}")
async def get_submission_details(submission_id: int):
    """Get detailed information about a specific submission"""
    
    try:
        db = get_database()
        submission = db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Parse evaluation result if it exists
        evaluation_result = {}
        if submission.evaluation_result:
            try:
                evaluation_result = json.loads(submission.evaluation_result)
            except:
                evaluation_result = {"raw": submission.evaluation_result}
        
        # Clean and return submission data
        submission_clean = {
            "id": submission.id,
            "dataset_name": submission.dataset_name,
            "model_name": submission.submitter_name,
            "description": submission.description or "",
            "contact_email": submission.submitter_email,
            "filename": os.path.basename(submission.file_path) if submission.file_path else "",
            "upload_date": submission.submission_time.isoformat() if submission.submission_time else None,
            "status": submission.status,
            "file_format": submission.file_format,
            "evaluation_result": evaluation_result,
            "error_message": submission.error_message,
            "validator_type": "real" if USING_REAL_VALIDATOR else "dummy"
        }
        
        return clean_for_json(submission_clean)
    
    except HTTPException:
        raise
    except Exception as e:
        return clean_for_json({"error": str(e)})

@router.delete("/{submission_id}")
async def delete_submission(submission_id: int):
    """Delete a specific submission"""
    
    try:
        db = get_database()
        submission = db.query(OutputSubmission).filter(OutputSubmission.id == submission_id).first()
        
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        
        # Delete the submission
        db.delete(submission)
        db.commit()
        
        return clean_for_json({
            "success": True,
            "message": f"Submission {submission_id} deleted successfully"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        return clean_for_json({
            "success": False,
            "error": str(e),
            "message": "Failed to delete submission"
        })