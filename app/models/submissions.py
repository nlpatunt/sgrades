# app/models/submission.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class SubmissionCreate(BaseModel):
    model_name: str = Field(..., description="Name of the submitted model")
    dataset_name: str = Field(..., description="Dataset used for evaluation")
    description: Optional[str] = Field(None, description="Optional model description")
    evaluation_results: Dict[str, Any] = Field(..., description="Evaluation metrics and scores")
    file_name: str = Field(..., description="Original filename of submitted CSV")

class SubmissionResponse(BaseModel):
    message: str
    model_name: str
    dataset_name: str
    evaluation_results: Dict[str, Any]
    submission_id: str
    timestamp: Optional[datetime] = None

class EvaluationResult(BaseModel):
    """Structure for evaluation results"""
    overall_score: float
    metrics: Dict[str, float]  # e.g., {"qwk": 0.85, "pearson": 0.78, "mse": 0.24}
    dataset_name: str
    total_essays: int
    processing_time: float
    details: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    """Structure for CSV validation results"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    row_count: int = 0
    column_count: int = 0
    missing_columns: List[str] = []
    extra_columns: List[str] = []

# app/services/evaluation_engine.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import mean_squared_error, f1_score
from scipy.stats import pearsonr
import time
from app.models.submission import ValidationResult, EvaluationResult

class EvaluationEngine:
    """Service for evaluating model submissions against reference data"""
    
    def __init__(self):
        self.supported_metrics = [
            'quadratic_weighted_kappa',
            'pearson_correlation', 
            'mean_squared_error',
            'mean_absolute_error',
            'f1_score'
        ]
    
    def validate_submission_format(
        self, 
        predictions_df: pd.DataFrame, 
        reference_data: pd.DataFrame, 
        dataset_name: str
    ) -> ValidationResult:
        """
        Validate that the submitted CSV has the correct format
        """
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = ['essay_id', 'predicted_score']  # Adjust based on your datasets
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for extra columns
        extra_columns = [col for col in predictions_df.columns if col not in required_columns]
        if extra_columns:
            warnings.append(f"Extra columns found (will be ignored): {extra_columns}")
        
        # Check if essay_ids match reference data
        if 'essay_id' in predictions_df.columns and 'essay_id' in reference_data.columns:
            pred_ids = set(predictions_df['essay_id'].astype(str))
            ref_ids = set(reference_data['essay_id'].astype(str))
            
            missing_ids = ref_ids - pred_ids
            extra_ids = pred_ids - ref_ids
            
            if missing_ids:
                errors.append(f"Missing predictions for essay IDs: {list(missing_ids)[:10]}...")
            
            if extra_ids:
                warnings.append(f"Extra essay IDs found: {list(extra_ids)[:10]}...")
        
        # Check for null values
        if predictions_df.isnull().any().any():
            errors.append("CSV contains null/empty values")
        
        # Check data types
        if 'predicted_score' in predictions_df.columns:
            try:
                pd.to_numeric(predictions_df['predicted_score'])
            except ValueError:
                errors.append("predicted_score column contains non-numeric values")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            row_count=len(predictions_df),
            column_count=len(predictions_df.columns),
            missing_columns=missing_columns,
            extra_columns=extra_columns
        )
    
    async def evaluate_submission(
        self,
        predictions_df: pd.DataFrame,
        reference_data: pd.DataFrame,
        dataset_name: str,
        model_name: str
    ) -> EvaluationResult:
        """
        Evaluate model predictions against reference scores
        """
        start_time = time.time()
        
        # Merge predictions with reference data
        merged_df = pd.merge(
            predictions_df, 
            reference_data, 
            on='essay_id', 
            how='inner'
        )
        
        if len(merged_df) == 0:
            raise ValueError("No matching essay IDs found between predictions and reference data")
        
        # Extract predicted and actual scores
        y_pred = merged_df['predicted_score'].astype(float)
        y_true = merged_df['score'].astype(float)  # Adjust column name based on your data
        
        # Calculate metrics
        metrics = {}
        
        # Pearson correlation
        if len(y_pred) > 1:  # Need at least 2 points for correlation
            corr, _ = pearsonr(y_pred, y_true)
            metrics['pearson_correlation'] = float(corr) if not np.isnan(corr) else 0.0
        
        # Mean Squared Error
        metrics['mean_squared_error'] = float(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Error
        metrics['mean_absolute_error'] = float(np.mean(np.abs(y_true - y_pred)))
        
        # Quadratic Weighted Kappa (simplified version)
        metrics['quadratic_weighted_kappa'] = self._calculate_qwk(y_true, y_pred)
        
        # Overall score (you can customize this)
        overall_score = metrics['pearson_correlation'] * 0.6 + (1 - metrics['mean_squared_error'] / 100) * 0.4
        overall_score = max(0, min(1, overall_score))  # Clamp between 0 and 1
        
        processing_time = time.time() - start_time
        
        return EvaluationResult(
            overall_score=float(overall_score),
            metrics=metrics,
            dataset_name=dataset_name,
            total_essays=len(merged_df),
            processing_time=processing_time,
            details={
                'matched_essays': len(merged_df),
                'total_predictions': len(predictions_df),
                'total_reference': len(reference_data)
            }
        )
    
    def _calculate_qwk(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Quadratic Weighted Kappa (simplified implementation)
        """
        try:
            # Round predictions to nearest integer for QWK calculation
            y_true_rounded = np.round(y_true).astype(int)
            y_pred_rounded = np.round(y_pred).astype(int)
            
            # Simple implementation - you might want to use a proper QWK library
            from sklearn.metrics import cohen_kappa_score
            return float(cohen_kappa_score(y_true_rounded, y_pred_rounded, weights='quadratic'))
        except Exception:
            return 0.0