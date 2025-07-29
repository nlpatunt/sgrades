from pydantic import BaseModel, Field
from typing import Dict, Union, Tuple, List, Optional, Any
from datetime import datetime
from enum import Enum

# Keep existing models...
class DatasetConfig(BaseModel):
    name: str
    description: str
    huggingface_id: str = None
    evaluation_metrics: List[str]
    score_ranges: Dict[str, Any]

class EssaySubmission(BaseModel):
    essay_text: str = Field(..., description="The essay text to be graded")
    dataset_name: str = Field(..., description="Name of the dataset to use for grading")
    prompt: Optional[str] = Field(None, description="The essay prompt")
    student_id: Optional[str] = Field(None, description="Student identifier")
    gold_scores: Optional[Dict[str, float]] = None
    # The human-assigned scores (if available), e.g., {"holistic_score": 4.5, "grammar": 3.0}
    submission_time: Optional[datetime] = None
    submission_id: Optional[str] = None

# NEW MODELS FOR BENCHMARKING:

class ModelType(str, Enum):
    API_ENDPOINT = "api_endpoint"
    HUGGINGFACE = "huggingface"
    OPENAI_COMPATIBLE = "openai_compatible"

class ModelSubmission(BaseModel):
    model_name: str = Field(..., description="Name of the submitted model")
    model_type: ModelType = Field(..., description="Type of model submission")
    
    # For API endpoints
    api_endpoint: Optional[str] = Field(None, description="API endpoint URL for the model")
    api_key: Optional[str] = Field(None, description="API key if required")
    
    # For HuggingFace models
    huggingface_model_id: Optional[str] = Field(None, description="HuggingFace model identifier")
    
    # Metadata
    submitter_name: str = Field(..., description="Name of the researcher/team")
    submitter_email: str = Field(..., description="Contact email")
    model_description: Optional[str] = Field(None, description="Description of the model")
    paper_url: Optional[str] = Field(None, description="Link to associated paper")
    
    submission_time: datetime = Field(default_factory=datetime.now)

class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    submitter_name: str
    submission_time: datetime
    status: str  # "submitted", "evaluating", "completed", "failed"
    model_type: ModelType