from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from datetime import datetime

# -----------------------------
# Represents metadata for a dataset in BESESR
# -----------------------------
class DatasetConfig(BaseModel):
    name: str  # e.g., "asap-aes"
    description: str  # e.g., "ASAP Automated Essay Scoring dataset"
    huggingface_id: Optional[str] = None  # if it's published to Hugging Face
    evaluation_metrics: List[str]  # e.g., ["QWK", "Accuracy"]
    
    # Each metric's valid range: e.g., {"QWK": (0.0, 1.0), "Accuracy": (0.0, 1.0)}
    score_ranges: Dict[str, Tuple[float, float]]


# -----------------------------
# Represents a single essay submission (e.g., from a student)
# -----------------------------
class EssaySubmission(BaseModel):
    essay_text: str = Field(..., max_length=5000)
    # The actual essay content submitted for scoring

    dataset_name: str
    # Name of the dataset this essay is associated with, e.g., "asap-aes"

    prompt: Optional[str] = None
    # The essay prompt (if shown to user or stored alongside)

    student_id: Optional[str] = None
    # You may use this if you plan to track submissions per student

    gold_scores: Optional[Dict[str, float]] = None
    # The human-assigned scores (if available), e.g., {"QWK": 0.73}

    submission_time: Optional[datetime] = None
    # Time the essay was submitted (useful for logging, auditing)

    submission_id: Optional[str] = None
    # Unique ID for tracking this submission
