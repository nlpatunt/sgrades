from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

# -----------------------------
# For individual essay grading (used during evaluation)
# -----------------------------
class EssayGradingRequest(BaseModel):
    """Request sent to a submitted model for grading an essay"""
    essay_text: str
    prompt: Optional[str] = None
    dataset_name: str
    essay_id: str

class EssayGradingResponse(BaseModel):
    """Response expected from a submitted model"""
    essay_id: str
    scores: Dict[str, float]  # e.g., {"holistic_score": 4.5, "grammar": 3.2}
    processing_time: Optional[float] = None
    model_confidence: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None

# -----------------------------
# For evaluation pipeline (internal use)
# -----------------------------
class EvaluationTask(BaseModel):
    """A single evaluation task"""
    task_id: str
    model_id: str
    dataset_name: str
    essay_ids: List[str]
    status: str = "pending"  # "pending", "running", "completed", "failed"
    created_time: datetime = Field(default_factory=datetime.now)
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    error_message: Optional[str] = None

class EssayComparison(BaseModel):
    """Comparison between model prediction and gold standard for one essay"""
    essay_id: str
    gold_scores: Dict[str, float]  # Human scores
    predicted_scores: Dict[str, float]  # Model scores
    differences: Dict[str, float]  # Absolute differences
    relative_errors: Dict[str, float]  # Relative errors

# -----------------------------
# Legacy models (keep for compatibility)
# -----------------------------
class GradingResponse(BaseModel):
    """Legacy grading response - kept for backward compatibility"""
    scores: Dict[str, float]
    feedback: str
    dataset_used: str
    model_used: str
    processing_time: float
    timestamp: datetime

class BenchmarkResult(BaseModel):
    """Legacy benchmark result - kept for compatibility"""
    dataset_name: str
    total_essays: int
    results: Dict[str, Any]