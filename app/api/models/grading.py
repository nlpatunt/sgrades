from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

# -----------------------------
# Response returned when grading one essay
# -----------------------------
class GradingResponse(BaseModel):
    scores: Dict[str, float]  # e.g., {"QWK": 0.78, "Accuracy": 0.9}
    feedback: str             # e.g., "The essay is coherent but lacks examples."
    dataset_used: str         # e.g., "asap-aes"
    model_used: str           # e.g., "gpt-4-openrouter"
    processing_time: float    # In seconds, e.g., 0.231
    timestamp: datetime       # When the response was generated


# -----------------------------
# Aggregated benchmark result for one dataset
# -----------------------------
class BenchmarkResult(BaseModel):
    dataset_name: str              # e.g., "asap-aes"
    total_essays: int              # total count evaluated
    results: Dict[str, Any]        # flexible structure: e.g., metric → average, std
