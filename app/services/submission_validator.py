# app/services/submission_validator.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re

@dataclass
class ValidationResult:
    """Result of submission validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    dataset_name: str
    row_count: int
    expected_row_count: int

class SubmissionValidator:
    """Validates researcher submissions against original dataset formats"""
    
    def __init__(self):
        # Dataset validation schemas based on your HuggingFace formats
        self.DATASET_SCHEMAS = {
            'ASAP-AES': {
                'required_columns': ['essay_id', 'domain1_score', 'domain2_score'],
                'optional_score_columns': ['rater1_domain1', 'rater2_domain1', 'rater3_domain1', 
                                         'rater1_domain2', 'rater2_domain2',
                                         'rater1_trait1', 'rater1_trait2', 'rater1_trait3', 'rater1_trait4',
                                         'rater1_trait5', 'rater1_trait6', 'rater2_trait1', 'rater2_trait2',
                                         'rater2_trait3', 'rater2_trait4', 'rater2_trait5', 'rater2_trait6',
                                         'rater3_trait1', 'rater3_trait2', 'rater3_trait3', 'rater3_trait4',
                                         'rater3_trait5', 'rater3_trait6'],
                'essay_id_pattern': r'^ASAP-AES_test_\d+$',
                'score_range': (0, 60),
                'score_type': 'float',
                'primary_score': 'domain1_score'
            },
            
            'ASAP2': {
                'required_columns': ['essay_id', 'score'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^ASAP2_test_\d+$',
                'score_range': (0, 60),
                'score_type': 'float',
                'primary_score': 'score'
            },
            
            'ASAP-SAS': {
                'required_columns': ['Id', 'Score1'],
                'optional_score_columns': ['Score2'],
                'essay_id_pattern': r'^ASAP-SAS_test_\d+$',
                'score_range': (0, 60),
                'score_type': 'float',
                'primary_score': 'Score1'
            },
            
            'ASAP_plus_plus': {
                'required_columns': ['essay_id', 'overall_score'],
                'optional_score_columns': ['Content', 'Organization', 'Word Choice', 'Sentence Fluency', 
                                         'Conventions', 'Prompt Adherence', 'Language', 'Narrativity'],
                'essay_id_pattern': r'^ASAP_plus_plus_test_\d+$',
                'score_range': (0, 100),
                'score_type': 'float',
                'primary_score': 'overall_score'
            },
            
            'CSEE': {
                'required_columns': ['essay_id', 'overall_score'],
                'optional_score_columns': ['content_score', 'language_score', 'structure_score'],
                'essay_id_pattern': r'^CSEE_test_\d+$',
                'score_range': (0, 100),
                'score_type': 'float',
                'primary_score': 'overall_score'
            },
            
            'BEEtlE_2way': {
                'required_columns': ['question_id', 'label'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^BEEtlE_2way_test_\d+$',
                'score_range': (0, 1),
                'score_type': 'int',
                'primary_score': 'label'
            },
            
            'BEEtlE_3way': {
                'required_columns': ['question_id', 'label'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^BEEtlE_3way_test_\d+$',
                'score_range': (0, 2),
                'score_type': 'int',
                'primary_score': 'label'
            },
            
            'SciEntSBank_2way': {
                'required_columns': ['question_id', 'label'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^SciEntSBank_2way_test_\d+$',
                'score_range': (0, 1),
                'score_type': 'int',
                'primary_score': 'label'
            },
            
            'SciEntSBank_3way': {
                'required_columns': ['question_id', 'label'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^SciEntSBank_3way_test_\d+$',
                'score_range': (0, 2),
                'score_type': 'int',
                'primary_score': 'label'
            },
            
            'Mohlar': {
                'required_columns': ['question_id', 'grade'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^Mohlar_test_\d+$',
                'score_range': (0, 100),
                'score_type': 'float',
                'primary_score': 'grade'
            },
            
            'Ielts_Writing_Dataset': {
                'required_columns': ['essay_id', 'Overall_Score'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^Ielts_Writing_Dataset_test_\d+$',
                'score_range': (0, 9),
                'score_type': 'float',
                'primary_score': 'Overall_Score'
            },
            
            'Ielst_Writing_Task_2_Dataset': {
                'required_columns': ['essay_id', 'band_score'],
                'optional_score_columns': [],
                'essay_id_pattern': r'^Ielst_Writing_Task_2_Dataset_test_\d+$',
                'score_range': (0, 9),
                'score_type': 'float',
                'primary_score': 'band_score'
            },
            
            'persuade_2': {
                'required_columns': ['essay_id_comp', 'holistic_essay_score'],
                'optional_score_columns': ['grade_level'],
                'essay_id_pattern': r'^persuade_2_test_\d+$',
                'score_range': (0, 100),
                'score_type': 'float',
                'primary_score': 'holistic_essay_score'
            },
            
            'Regrading_Dataset_J2C': {
                'required_columns': ['Question ID', 'grade'],
                'optional_score_columns': ['llm_grade', 'llm_total_grade'],
                'essay_id_pattern': r'^Regrading_Dataset_J2C_test_\d+$',
                'score_range': (0, 100),
                'score_type': 'float',
                'primary_score': 'grade'
            }
        }
        
        for q in ['q1', 'q2', 'q3', 'q4', 'q5']:
            self.DATASET_SCHEMAS[f'grade_like_a_human_dataset_os_{q}'] = {
                'required_columns': ['id', 'score_1'],
                'optional_score_columns': ['score_2', 'score_3', 'score_outlier'],
                'essay_id_pattern': f'^grade_like_a_human_dataset_os_{q}_test_\\d+$',
                'score_range': (0, 100),
                'score_type': 'float',
                'primary_score': 'score_1'
            }
        
        # Add Rice_Chem configs (Q1-Q4)
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            self.DATASET_SCHEMAS[f'Rice_Chem_{q}'] = {
                'required_columns': ['sis_id', 'Score'],
                'optional_score_columns': [],
                'essay_id_pattern': f'^Rice_Chem_{q}_test_\\d+$',
                'score_range': (0, 100),
                'score_type': 'float',
                'primary_score': 'Score'
            }

    def validate_submission(self, dataset_name: str, csv_file_path: str, 
                          expected_test_size: Optional[int] = None) -> ValidationResult:
        """
        Validate a researcher's submission against the original dataset format
        
        Args:
            dataset_name: Name of the dataset (e.g., 'ASAP-AES', 'BEEtlE_2way')
            csv_file_path: Path to the uploaded CSV file
            expected_test_size: Expected number of test examples (optional)
        
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        
        try:
            # 1. Check if dataset schema exists
            if dataset_name not in self.DATASET_SCHEMAS:
                errors.append(f"Unknown dataset '{dataset_name}'. Available datasets: {list(self.DATASET_SCHEMAS.keys())}")
                return ValidationResult(False, errors, warnings, dataset_name, 0, expected_test_size or 0)
            
            schema = self.DATASET_SCHEMAS[dataset_name]
            
            # 2. Load and parse CSV
            try:
                df = pd.read_csv(csv_file_path)
            except Exception as e:
                errors.append(f"Failed to read CSV file: {str(e)}")
                return ValidationResult(False, errors, warnings, dataset_name, 0, expected_test_size or 0)
            
            if df.empty:
                errors.append("CSV file is empty")
                return ValidationResult(False, errors, warnings, dataset_name, 0, expected_test_size or 0)
            
            # 3. Validate column structure
            required_cols = schema['required_columns']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
            
            extra_cols = [col for col in df.columns if col not in required_cols + schema.get('optional_score_columns', [])]
            if extra_cols:
                warnings.append(f"Extra columns found (will be ignored): {extra_cols}")
            
            # 4. Validate essay IDs
            id_column = required_cols[0]  # First column is always the ID
            if id_column in df.columns:
                self._validate_essay_ids(df, id_column, schema['essay_id_pattern'], errors, warnings)
            
            # 5. Validate scores
            primary_score_col = schema['primary_score']
            if primary_score_col in df.columns:
                self._validate_scores(df, primary_score_col, schema, errors, warnings)
            
            # 6. Validate data completeness
            if expected_test_size:
                if len(df) != expected_test_size:
                    errors.append(f"Row count mismatch: got {len(df)}, expected {expected_test_size}")
            
            # 7. Check for duplicates
            if id_column in df.columns:
                duplicates = df[id_column].duplicated().sum()
                if duplicates > 0:
                    errors.append(f"Found {duplicates} duplicate essay IDs")
            
            # 8. Check for missing values
            missing_scores = df[primary_score_col].isnull().sum() if primary_score_col in df.columns else 0
            if missing_scores > 0:
                errors.append(f"Found {missing_scores} missing scores in {primary_score_col}")
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                dataset_name=dataset_name,
                row_count=len(df),
                expected_row_count=expected_test_size or len(df)
            )
            
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
            return ValidationResult(False, errors, warnings, dataset_name, 0, expected_test_size or 0)
    
    def _validate_dataframe(self, df: pd.DataFrame, dataset_name: str, 
                           expected_test_size: Optional[int] = None) -> ValidationResult:
        """Validate DataFrame directly (used by enhanced validation)"""
        
        errors = []
        warnings = []
        
        # Check if dataset schema exists
        if dataset_name not in self.DATASET_SCHEMAS:
            errors.append(f"Unknown dataset '{dataset_name}'. Available datasets: {list(self.DATASET_SCHEMAS.keys())}")
            return ValidationResult(False, errors, warnings, dataset_name, 0, expected_test_size or 0)
        
        schema = self.DATASET_SCHEMAS[dataset_name]
        
        if df.empty:
            errors.append("CSV data is empty")
            return ValidationResult(False, errors, warnings, dataset_name, 0, expected_test_size or 0)
        
        # Validate column structure
        required_cols = schema['required_columns']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        extra_cols = [col for col in df.columns if col not in required_cols + schema.get('optional_score_columns', [])]
        if extra_cols:
            warnings.append(f"Extra columns found (will be ignored): {extra_cols}")
        
        # Validate essay IDs
        id_column = required_cols[0]  # First column is always the ID
        if id_column in df.columns:
            self._validate_essay_ids(df, id_column, schema['essay_id_pattern'], errors, warnings)
        
        # Validate scores
        primary_score_col = schema['primary_score']
        if primary_score_col in df.columns:
            self._validate_scores(df, primary_score_col, schema, errors, warnings)
        
        # Validate data completeness
        if expected_test_size:
            if len(df) != expected_test_size:
                errors.append(f"Row count mismatch: got {len(df)}, expected {expected_test_size}")
        
        # Check for duplicates
        if id_column in df.columns:
            duplicates = df[id_column].duplicated().sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate essay IDs")
        
        # Check for missing values
        missing_scores = df[primary_score_col].isnull().sum() if primary_score_col in df.columns else 0
        if missing_scores > 0:
            errors.append(f"Found {missing_scores} missing scores in {primary_score_col}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            dataset_name=dataset_name,
            row_count=len(df),
            expected_row_count=expected_test_size or len(df)
        )
    
    def _validate_essay_ids(self, df: pd.DataFrame, id_column: str, pattern: str, 
                           errors: List[str], warnings: List[str]):
        """Validate essay ID format and uniqueness"""
        try:
            invalid_ids = []
            for idx, essay_id in enumerate(df[id_column]):
                if pd.isna(essay_id):
                    invalid_ids.append(f"Row {idx + 1}: Missing essay ID")
                elif not re.match(pattern, str(essay_id)):
                    invalid_ids.append(f"Row {idx + 1}: Invalid ID format '{essay_id}' (expected pattern: {pattern})")
            
            if invalid_ids:
                if len(invalid_ids) <= 5:
                    errors.extend(invalid_ids)
                else:
                    errors.append(f"Found {len(invalid_ids)} invalid essay IDs. First 5: {invalid_ids[:5]}")
                    
        except Exception as e:
            warnings.append(f"Could not validate essay ID format: {str(e)}")
    
    def _validate_scores(self, df: pd.DataFrame, score_column: str, schema: Dict, 
                        errors: List[str], warnings: List[str]):
        """Validate score values and ranges"""
        try:
            scores = df[score_column].dropna()
            
            if len(scores) == 0:
                errors.append(f"No valid scores found in {score_column}")
                return
            
            # Check data type
            score_type = schema['score_type']
            if score_type == 'int':
                non_integers = scores[~scores.astype(str).str.match(r'^\d+$')]
                if len(non_integers) > 0:
                    errors.append(f"Found {len(non_integers)} non-integer values in {score_column} (expected integers)")
            elif score_type == 'float':
                try:
                    pd.to_numeric(scores)
                except:
                    errors.append(f"Found non-numeric values in {score_column}")
            
            # Check score range
            score_min, score_max = schema['score_range']
            numeric_scores = pd.to_numeric(scores, errors='coerce').dropna()
            
            out_of_range = numeric_scores[(numeric_scores < score_min) | (numeric_scores > score_max)]
            if len(out_of_range) > 0:
                errors.append(f"Found {len(out_of_range)} scores outside valid range [{score_min}, {score_max}]")
            
            # Statistical sanity checks
            unique_scores = numeric_scores.nunique()
            if unique_scores == 1:
                warnings.append(f"All predictions have the same value ({numeric_scores.iloc[0]})")
            elif unique_scores < 3:
                warnings.append(f"Very few unique prediction values ({unique_scores})")
            
            # Check if using reasonable portion of score range
            used_range = numeric_scores.max() - numeric_scores.min()
            possible_range = score_max - score_min
            if used_range < possible_range * 0.1:
                warnings.append(f"Using only {used_range:.1f} of {possible_range} possible score range")
                
        except Exception as e:
            warnings.append(f"Could not validate scores: {str(e)}")
    
    def get_expected_format(self, dataset_name: str) -> Dict[str, Any]:
        """Get expected submission format for a dataset"""
        if dataset_name not in self.DATASET_SCHEMAS:
            return {"error": f"Unknown dataset: {dataset_name}"}
        
        schema = self.DATASET_SCHEMAS[dataset_name]
        return {
            "dataset_name": dataset_name,
            "required_columns": schema['required_columns'],
            "optional_columns": schema.get('optional_score_columns', []),
            "primary_score_column": schema['primary_score'],
            "essay_id_pattern": schema['essay_id_pattern'],
            "score_range": schema['score_range'],
            "score_type": schema['score_type'],
            "example_format": {
                schema['required_columns'][0]: f"{dataset_name}_test_0",
                schema['primary_score']: f"Example score within range {schema['score_range']}"
            }
        }
    
    def validate_all_submissions(self, submissions: Dict[str, str]) -> Dict[str, ValidationResult]:
        """Validate multiple dataset submissions at once"""
        results = {}
        for dataset_name, csv_path in submissions.items():
            results[dataset_name] = self.validate_submission(dataset_name, csv_path)
        return results

# Global validator instance
submission_validator = SubmissionValidator()