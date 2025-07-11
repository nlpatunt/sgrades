from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from datetime import datetime
import uuid

from app.api.models.essay import ModelSubmission, ModelInfo
from app.api.models.evaluation import ModelEvaluationResult, LeaderboardEntry
from app.config.datasets import get_all_datasets

router = APIRouter(prefix="/models", tags=["models"])

# In-memory storage for demo (replace with database later)
submitted_models: Dict[str, ModelSubmission] = {}
model_results: Dict[str, ModelEvaluationResult] = {}

@router.post("/submit", response_model=ModelInfo)
async def submit_model(submission: ModelSubmission):
    """Submit a new model for evaluation"""
    
    # Generate unique model ID
    model_id = f"model_{str(uuid.uuid4())[:8]}"
    
    # Validate submission
    if submission.model_type == "api_endpoint" and not submission.api_endpoint:
        raise HTTPException(status_code=400, detail="API endpoint required for API model type")
    
    if submission.model_type == "huggingface" and not submission.huggingface_model_id:
        raise HTTPException(status_code=400, detail="HuggingFace model ID required for HuggingFace model type")
    
    # Store submission
    submitted_models[model_id] = submission
    
    # Return model info
    return ModelInfo(
        model_id=model_id,
        model_name=submission.model_name,
        submitter_name=submission.submitter_name,
        submission_time=submission.submission_time,
        status="submitted",
        model_type=submission.model_type
    )

@router.get("/", response_model=List[ModelInfo])
async def list_models():
    """Get list of all submitted models"""
    return [
        ModelInfo(
            model_id=model_id,
            model_name=submission.model_name,
            submitter_name=submission.submitter_name,
            submission_time=submission.submission_time,
            status="submitted",  # TODO: Get real status from evaluation results
            model_type=submission.model_type
        )
        for model_id, submission in submitted_models.items()
    ]

@router.get("/{model_id}/info", response_model=ModelSubmission)
async def get_model_info(model_id: str):
    """Get detailed information about a submitted model"""
    if model_id not in submitted_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return submitted_models[model_id]

@router.post("/{model_id}/evaluate")
async def start_evaluation(model_id: str, background_tasks: BackgroundTasks):
    """Start evaluation of a model across all datasets"""
    if model_id not in submitted_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get all available datasets
    datasets = get_all_datasets()
    
    # Add evaluation task to background
    background_tasks.add_task(run_model_evaluation, model_id, list(datasets.keys()))
    
    return {
        "message": f"Evaluation started for model {model_id}",
        "model_id": model_id,
        "datasets_to_evaluate": list(datasets.keys()),
        "estimated_time": "5-10 minutes"
    }

@router.get("/{model_id}/results")
async def get_model_results(model_id: str):
    """Get evaluation results for a specific model"""
    if model_id not in submitted_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model_id not in model_results:
        return {
            "model_id": model_id,
            "status": "no_evaluation_started",
            "message": "No evaluation has been started for this model"
        }
    
    return model_results[model_id]

@router.get("/{model_id}/status")
async def get_evaluation_status(model_id: str):
    """Get current evaluation status for a model"""
    if model_id not in submitted_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # TODO: Implement real status tracking
    return {
        "model_id": model_id,
        "status": "submitted",
        "progress": "0/12 datasets completed",
        "estimated_remaining_time": "Unknown"
    }

# Background task function (placeholder)
async def run_model_evaluation(model_id: str, dataset_names: List[str]):
    """Run evaluation in background - placeholder implementation"""
    # TODO: Implement actual evaluation logic
    print(f"Starting evaluation for model {model_id} on datasets: {dataset_names}")
    
    # This will be implemented in the next step
    pass