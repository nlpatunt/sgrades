from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class DatasetEvaluationResult(BaseModel):
    """Results for one model on one dataset"""
    dataset_name: str
    model_id: str
    total_essays: int
    
    # Core metrics (calculated vs human gold scores)
    metrics: Dict[str, float]  # e.g., {"quadratic_weighted_kappa": 0.73, "pearson_correlation": 0.82}
    
    # Per-essay results
    essay_results: List[Dict[str, Any]]  # Individual essay predictions vs gold scores
    
    # Evaluation metadata
    evaluation_time: datetime
    processing_time: float  # Total time taken
    status: str  # "completed", "failed", "in_progress"
    error_message: Optional[str] = None

class ModelEvaluationResult(BaseModel):
    """Complete evaluation results for one model across all datasets"""
    model_id: str
    model_name: str
    submitter_name: str
    
    # Results per dataset
    dataset_results: List[DatasetEvaluationResult]
    
    # Overall performance summary
    overall_metrics: Dict[str, float]  # Average metrics across datasets
    
    # Ranking information
    overall_rank: Optional[int] = None
    dataset_ranks: Dict[str, int] = {}  # Rank per dataset
    
    evaluation_start_time: datetime
    evaluation_end_time: Optional[datetime] = None
    total_evaluation_time: Optional[float] = None

class LeaderboardEntry(BaseModel):
    """Single entry for the leaderboard"""
    rank: int
    model_id: str
    model_name: str
    submitter_name: str
    
    # Key performance metrics
    avg_quadratic_weighted_kappa: float
    avg_pearson_correlation: float
    avg_accuracy: Optional[float] = None
    
    # Number of datasets evaluated
    datasets_completed: int
    total_datasets: int
    
    submission_time: datetime

class BenchmarkReport(BaseModel):
    """Comprehensive benchmark report"""
    report_id: str
    generated_time: datetime
    
    # Summary statistics
    total_models: int
    total_datasets: int
    total_evaluations: int
    
    # Leaderboard
    leaderboard: List[LeaderboardEntry]
    
    # Dataset-specific rankings
    dataset_leaderboards: Dict[str, List[LeaderboardEntry]]
    
    # Performance distributions
    metric_distributions: Dict[str, Dict[str, float]]  # e.g., {"qwk": {"mean": 0.75, "std": 0.12}}